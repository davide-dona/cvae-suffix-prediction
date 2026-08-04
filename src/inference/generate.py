from dataclasses import dataclass
from pathlib import Path
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.inference.batch import BatchGeneration, generate_batch
from src.datasets.codec import decode_sequence
from src.datasets.dataset import SuffixDataset, SuffixItem
from src.datasets.description import DatasetDescription
from src.model import TransformerCVAE

@dataclass(frozen=True)
class GenerationRow:
    """One row of a generations file: one (prefix, sample) pair and the ground truth beside it.

    One row of a table rather than a structure: what this becomes is a parquet file, which
    `src/evaluation` reads back by column name. A channel the model learns to write gains a
    column per side here.

    `sample_index` enumerates the generated suffixes of one prefix; it identifies a row rather
    than describing it. The `point_` and `true_` fields describe the prefix rather than the
    sample, so they repeat unchanged across its rows. Every remaining time is minutes from the
    end of the prefix to the end of the case.
    """
    case_id: str
    prefix_len: int
    truncated: bool     # whether the ground-truth suffix was cut short of its real ending
    sample_index: int   # which of the `num_samples` generated suffixes of this prefix this is
    generated_activities: list[str]
    generated_remaining_time_minutes: float
    # The suffix written from the mean of `p(z | prefix)`: the model's single answer, drawn once
    # per prefix and the only column comparable against a model that does not sample.
    point_activities: list[str]
    point_remaining_time_minutes: float
    true_activities: list[str]
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
    description: DatasetDescription,
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
        description: The description the split was encoded through, read here in the decode
            direction.
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
        rows += _batch_rows(
            batch=batch, generation=generation, dataset=dataset, description=description
        )

    return pd.DataFrame(rows)


def _batch_rows(
    batch: SuffixItem,
    generation: BatchGeneration,
    dataset: SuffixDataset,
    description: DatasetDescription,
) -> list[GenerationRow]:
    """Turn one batch's generation into rows: decode each suffix and the truth beside it, and
    attach the case and cut point the pair came from."""
    rows = []
    for position, pair_index in enumerate(batch.pair_index.tolist()):
        info = dataset.pair_info(pair_index)

        # Decoded once per pair, and repeated on each of its sample rows.
        truth = decode_sequence(
            description,
            activities=generation.true_activities[position],
            length=generation.true_lengths[position],
            remaining_time=generation.true_remaining_time[position],
        )
        point = decode_sequence(
            description,
            activities=generation.point_activities[position],
            length=generation.point_lengths[position],
            remaining_time=generation.point_remaining_time[position],
        )

        for sample_index in range(generation.activities.shape[1]):
            generated = decode_sequence(
                description,
                activities=generation.activities[position, sample_index],
                length=generation.lengths[position, sample_index],
                remaining_time=generation.remaining_time[position, sample_index],
            )
            rows.append(
                GenerationRow(
                    case_id=info.case_id,
                    prefix_len=info.prefix_len,
                    truncated=info.truncated,
                    sample_index=sample_index,
                    generated_activities=generated.activities,
                    generated_remaining_time_minutes=generated.remaining_time_minutes,
                    point_activities=point.activities,
                    point_remaining_time_minutes=point.remaining_time_minutes,
                    true_activities=truth.activities,
                    true_remaining_time_minutes=truth.remaining_time_minutes,
                )
            )
    return rows
