from dataclasses import dataclass
from pathlib import Path
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.inference.batch import BatchGeneration, generate_batch
from src.datasets.codec import Codec, EncodedSequence
from src.datasets.dataset import SuffixDataset, SuffixItem
from src.model import TransformerCVAE

@dataclass(frozen=True)
class GenerationRow:
    """One row of a generations file: one (prefix, sample) pair, decoded and denormalized so
    it compares directly against the ground truth beside it.

    `sample_index` enumerates the generated suffixes of one prefix; it identifies a row rather
    than describing it. The model writes activities and a remaining time only, so there is no
    generated resource column; the ground-truth resources stay as what a run that does predict
    them would be scored against. Both remaining times are minutes from the end of the prefix
    to the end of the case.
    """
    case_id: str
    prefix_len: int
    truncated: bool     # whether the ground-truth suffix was cut short of its real ending
    sample_index: int   # which of the `num_samples` generated suffixes of this prefix this is
    generated_activities: list[str]
    generated_remaining_time_minutes: float
    true_activities: list[str]
    true_resources: list[str]
    true_remaining_time_minutes: float

def generations_path(generations_dir: str | Path, run_name: str) -> Path:
    """Where the generations of one run are kept: `<generations_dir>/<run_name>.parquet`.

    One file per run, named after it exactly as `best_model_path` names a run's checkpoint,
    so a run's generations are found without being told anything but its name.
    """
    return Path(generations_dir) / f'{run_name}.parquet'

def generate_suffixes(
    model: TransformerCVAE,
    loader: DataLoader,
    codec: Codec,
    *,
    num_samples: int,
    device: torch.device,
) -> pd.DataFrame:
    """Generate `num_samples` suffixes for every prefix in `loader`, as `GenerationRow`s.

    The generations are denormalized and decoded back into activity names and minutes, so
    what comes out is comparable against the log itself.

    Args:
        model: The trained model, already on `device`.
        loader: Batches of the split to generate for, from a `SuffixDataset`.
        codec: The codec the split was encoded with, used here in its decode direction.
        num_samples: How many suffixes to generate per prefix.
        device: Where to run the generation.
    Returns:
        One row per (prefix, sample), as a `GenerationRow`.
    """
    dataset: SuffixDataset = loader.dataset
    model.eval()

    rows: list[GenerationRow] = []
    for batch in loader:
        batch = batch.to(device)
        generation = generate_batch(model, batch, num_samples=num_samples)
        rows += _batch_rows(batch=batch, generation=generation, dataset=dataset, codec=codec)

    return pd.DataFrame(rows)


def _batch_rows(
    batch: SuffixItem,
    generation: BatchGeneration,
    dataset: SuffixDataset,
    codec: Codec,
) -> list[GenerationRow]:
    """Turn one batch's generation into rows: decode the activities, denormalize the times,
    and attach each prefix's case and cut point."""
    pair_indices = batch.pair_index.tolist()

    # Indexing one of these picks a single suffix out of the batch. A generated suffix has no
    # resource channel; the ground truth keeps the log's.
    generated_suffixes = EncodedSequence(
        activities=generation.activities,  # [batch_size, num_samples, steps]
    )
    true_suffixes = EncodedSequence(
        activities=generation.true_activities,  # [batch_size, seq_len]
        resources=batch.suffix.resources.cpu().numpy(),
    )

    # Both sides of the time comparison, back in minutes.
    generated_remaining = codec.denormalize_remaining_time(generation.remaining_time)
    true_remaining = codec.denormalize_remaining_time(generation.true_remaining_time)

    rows = []
    for position, pair_index in enumerate(pair_indices):
        info = dataset.pair_info(pair_index)
        ground_truth = codec.decode_suffix(
            true_suffixes[position], length=generation.true_lengths[position]
        )

        for sample_index in range(generation.activities.shape[1]):
            generated = codec.decode_suffix(
                generated_suffixes[position, sample_index],
                length=generation.lengths[position, sample_index],
            )
            rows.append(
                GenerationRow(
                    case_id=info.case_id,
                    prefix_len=info.prefix_len,
                    truncated=info.truncated,
                    sample_index=sample_index,
                    generated_activities=generated.activities,
                    generated_remaining_time_minutes=float(
                        generated_remaining[position, sample_index]
                    ),
                    true_activities=ground_truth.activities,
                    true_resources=ground_truth.resources,
                    true_remaining_time_minutes=float(true_remaining[position]),
                )
            )
    return rows
