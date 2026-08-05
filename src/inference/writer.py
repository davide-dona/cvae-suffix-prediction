from pathlib import Path
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src.datasets.codec import DecodedSequence
from src.datasets.dataset import PairInfo
from src.inference.generation import Generation


# The schema of the Parquet file that holds a run's generations.
# One row per generated sample, with the point prediction
_SCHEMA = pa.schema([
    ('case_id', pa.large_string()),
    ('prefix_len', pa.int64()),
    # Whether the ground-truth suffix was cut short of its real ending.
    ('truncated', pa.bool_()),
    ('sample_index', pa.int64()),
    ('generated_activities', pa.list_(pa.field(name='element', type=pa.string()))),
    ('generated_remaining_time_minutes', pa.float64()),
    # The suffix written from the mean of `p(z | prefix)`: the model's single answer, drawn once
    # per prefix and the only column comparable against a model that does not sample.
    ('point_activities', pa.list_(pa.field(name='element', type=pa.string()))),
    ('point_remaining_time_minutes', pa.float64()),
    ('true_activities', pa.list_(pa.field(name='element', type=pa.string()))),
    ('true_remaining_time_minutes', pa.float64()),
])


def generations_path(generations_dir: str | Path, run_name: str) -> Path:
    """Where the generations of one run are kept: `<generations_dir>/<run_name>.parquet`.

    One file per run, named after it exactly as `checkpoint_path` names a run's checkpoint,
    so a run's generations are found without being told anything but its name.
    """
    return Path(generations_dir) / f'{run_name}.parquet'


def open_generations(path: Path) -> pq.ParquetWriter:
    """Open a Parquet file for writing generations, creating its parent directories if needed.
    Args:
        path: The file to write, from `generations_path`. Overwritten if it already exists.
    Returns:
        A writer bound to the generations schema, to be used as a context manager: closing it is
        what writes the file's footer.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    return pq.ParquetWriter(where=path, schema=_SCHEMA)


def table_from_generations(
    generations: list[Generation], infos: list[PairInfo]
) -> pa.Table:
    """Flatten one batch's generations into a table, one row per (prefix, sample).

    Args:
        generations: The model's answers, in the order `generate_batch` returned them.
        infos: Which pair each of those generations answers, in the same order.
    Returns:
        The rows as one table, ready to be written as a row group. The point prediction and the
        truth describe the prefix rather than the sample, so they repeat unchanged across its rows.
    Raises:
        ValueError: If `generations` and `infos` are of different lengths, which would put every
            generated suffix beside the wrong case.
    """
    # Dicts here because they are what arrow's constructor takes, keyed by the schema's own names.
    rows = [
        {
            'case_id': info.case_id,
            'prefix_len': info.prefix_len,
            'truncated': info.truncated,
            'sample_index': sample_index,
            'generated_activities': sample.activities,
            'generated_remaining_time_minutes': sample.remaining_time_minutes,
            'point_activities': generation.point.activities,
            'point_remaining_time_minutes': generation.point.remaining_time_minutes,
            'true_activities': generation.truth.activities,
            'true_remaining_time_minutes': generation.truth.remaining_time_minutes,
        }
        for generation, info in zip(generations, infos, strict=True)
        for sample_index, sample in enumerate(generation.samples)
    ]
    return pa.Table.from_pylist(mapping=rows, schema=_SCHEMA)


def generation_from_rows(rows: pd.DataFrame) -> Generation:
    """Read one prefix's generation back out of the rows a generations file holds it as.

    The inverse of `table_from_generations` over one prefix, which is what lets a written file be
    scored through the same `score_prefix` a training run reports from.

    Args:
        rows: The rows of a single (case, cut point), one per generated sample. The point
            prediction and the ground truth describe the prefix rather than the sample and repeat
            across the rows, so both are read off the first of them.
    Returns:
        The same generation `generate_batch` produced. Parquet hands the activity columns back as
        arrays, so they are copied into lists: `DecodedSequence` promises lists, and the edit
        distance that reads them is quicker to index for it.
    """
    ground_truth = rows.iloc[0]
    return Generation(
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
