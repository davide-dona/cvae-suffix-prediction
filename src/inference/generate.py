from dataclasses import dataclass
from pathlib import Path
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.inference.batch import generate_predictions
from src.inference.prediction import PrefixPrediction
from src.datasets.codec import DecodedSequence
from src.datasets.dataset import SuffixDataset
from src.datasets.description import DatasetDescription
from src.model import TransformerCVAE

@dataclass(frozen=True)
class GenerationRow:
    """One row of a generations file: one (prefix, sample) pair and the ground truth beside it.

    One row of a table rather than a structure: what this becomes is a parquet file. The column
    names are this module's business in both directions, written by `_batch_rows` and read back by
    `prediction_from_rows`, so a reader of a generations file asks here rather than knowing the
    schema. A channel the model learns to write gains a column per side here.

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

    One file per run, named after it exactly as `checkpoint_path` names a run's checkpoint,
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
    for batch_index, batch in enumerate(loader):
        print(
            f'Batch {batch_index + 1}/{len(loader)}: '
            f'{batch.pair_index.size(dim=0)} prefixes, {num_samples} samples each',
            flush=True,
        )
        batch = batch.to(device)
        rows += _batch_rows(
            pair_indices=batch.pair_index.tolist(),
            predictions=generate_predictions(
                model, batch, num_samples=num_samples, description=description
            ),
            dataset=dataset,
        )

    return pd.DataFrame(rows)


def prediction_from_rows(rows: pd.DataFrame) -> PrefixPrediction:
    """Read one prefix's prediction back out of the rows a generations file holds it as.

    The inverse of `_batch_rows` over one prefix, and the reason `src/evaluation` never needs to
    know a column name.

    Args:
        rows: The rows of a single (case, cut point), one per generated sample. The point
            prediction and the ground truth describe the prefix rather than the sample and repeat
            across the rows, so both are read off the first of them.
    Returns:
        The same prediction `generate_predictions` produced. Parquet hands the activity columns
        back as arrays, so they are copied into lists: `DecodedSequence` promises lists, and the
        edit distance that reads them is quicker to index for it.
    """
    ground_truth = rows.iloc[0]
    return PrefixPrediction(
        samples=[
            DecodedSequence(
                activities=list(sample.generated_activities),
                remaining_time_minutes=float(sample.generated_remaining_time_minutes),
            )
            for sample in rows.itertuples()
        ],
        point=DecodedSequence(
            activities=list(ground_truth.point_activities),
            remaining_time_minutes=float(ground_truth.point_remaining_time_minutes),
        ),
        truth=DecodedSequence(
            activities=list(ground_truth.true_activities),
            remaining_time_minutes=float(ground_truth.true_remaining_time_minutes),
        ),
    )


def _batch_rows(
    pair_indices: list[int],
    predictions: list[PrefixPrediction],
    dataset: SuffixDataset,
) -> list[GenerationRow]:
    """Flatten one batch's predictions into rows, one per (prefix, sample).

    Args:
        pair_indices: Which pair of `dataset` each prediction was made for, in the batch's order.
        predictions: The model's answer for each of those pairs.
        dataset: The dataset the pairs were cut from, read for their case and cut point.
    Returns:
        One row per (prefix, sample). The point prediction and the truth describe the prefix rather
        than the sample, so they repeat unchanged across its rows.
    """
    rows = []
    # `strict` because the correspondence between a prediction and the pair it was made for is the
    # one invariant tying a generated row back to its case.
    for pair_index, prediction in zip(pair_indices, predictions, strict=True):
        info = dataset.pair_info(pair_index)
        for sample_index, sample in enumerate(prediction.samples):
            rows.append(
                GenerationRow(
                    case_id=info.case_id,
                    prefix_len=info.prefix_len,
                    truncated=info.truncated,
                    sample_index=sample_index,
                    generated_activities=sample.activities,
                    generated_remaining_time_minutes=sample.remaining_time_minutes,
                    point_activities=prediction.point.activities,
                    point_remaining_time_minutes=prediction.point.remaining_time_minutes,
                    true_activities=prediction.truth.activities,
                    true_remaining_time_minutes=prediction.truth.remaining_time_minutes,
                )
            )
    return rows
