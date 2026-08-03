from dataclasses import dataclass
from pathlib import Path
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.inference.batch import BatchGeneration, generate_batch
from src.datasets.codec import DecodedSequence, EncodedSequence, decode_sequence
from src.datasets.dataset import PairInfo, SuffixDataset, SuffixItem
from src.datasets.description import DatasetDescription
from src.model import TransformerCVAE

@dataclass(frozen=True)
class GenerationRow:
    """One row of a generations file: one (prefix, sample) pair and the ground truth beside it.

    One row of a table rather than a structure, so each side's `DecodedSequence` is spread into
    columns of its own: what this becomes is a parquet file, which `src/evaluation` reads back
    by column name. A channel the model learns to write gains a column per side here.

    `sample_index` enumerates the generated suffixes of one prefix; it identifies a row rather
    than describing it. Both remaining times are minutes from the end of the prefix to the end
    of the case.
    """
    case_id: str
    prefix_len: int
    truncated: bool     # whether the ground-truth suffix was cut short of its real ending
    sample_index: int   # which of the `num_samples` generated suffixes of this prefix this is
    generated_activities: list[str]
    generated_remaining_time_minutes: float
    true_activities: list[str]
    true_remaining_time_minutes: float

    @classmethod
    def of(
        cls,
        info: PairInfo,
        *,
        sample_index: int,
        generated: DecodedSequence,
        truth: DecodedSequence,
    ) -> "GenerationRow":
        """Flatten one generated suffix and its ground truth into a row of the table.

        Args:
            info: Which cut of which case the pair is.
            sample_index: Which of the prefix's generated suffixes this row holds.
            generated: The suffix the model wrote, in the log's own units.
            truth: The suffix the log actually holds, decoded the same way.
        Returns:
            The row, ready to be collected into the generations frame.
        """
        return cls(
            case_id=info.case_id,
            prefix_len=info.prefix_len,
            truncated=info.truncated,
            sample_index=sample_index,
            generated_activities=generated.activities,
            generated_remaining_time_minutes=generated.remaining_time_minutes,
            true_activities=truth.activities,
            true_remaining_time_minutes=truth.remaining_time_minutes,
        )

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
    pair_indices = batch.pair_index.tolist()

    # Indexing one of these picks a single suffix out of the batch. Activities alone on both
    # sides: they are the only per-event channel the model writes.
    generated_suffixes = EncodedSequence(
        activities=generation.activities,  # [batch_size, num_samples, steps]
    )
    true_suffixes = EncodedSequence(
        activities=generation.true_activities,  # [batch_size, seq_len]
    )

    rows = []
    for position, pair_index in enumerate(pair_indices):
        info = dataset.pair_info(pair_index)
        truth = decode_sequence(
            description,
            true_suffixes[position],
            length=generation.true_lengths[position],
            remaining_time=generation.true_remaining_time[position],
        )

        for sample_index in range(generation.activities.shape[1]):
            generated = decode_sequence(
                description,
                generated_suffixes[position, sample_index],
                length=generation.lengths[position, sample_index],
                remaining_time=generation.remaining_time[position, sample_index],
            )
            rows.append(
                GenerationRow.of(
                    info, sample_index=sample_index, generated=generated, truth=truth
                )
            )
    return rows
