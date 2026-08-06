from collections.abc import Iterator
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src.inference.generation import DecodedEvents, Generation

# One run of activity names, the shape every activity column of the schema is built from.
_ACTIVITIES = pa.list_(pa.field(name='element', type=pa.string()))

# The schema of the Parquet file that holds a run's generations. One row per prefix: the samples
# nest inside it, so nothing describing the prefix is written once per sample.
_SCHEMA = pa.schema(
    [
        ('case_id', pa.large_string()),
        ('prefix_len', pa.int64()),
        # Whether the ground-truth suffix was cut short of its real ending.
        ('truncated', pa.bool_()),
        # The events before the cut, which a constraint over the whole trace is checked against.
        ('prefix_activities', _ACTIVITIES),
        # One entry per draw of z, in the order they were drawn: `hit_rate_at_k` reads the first k.
        ('generated_activities', pa.list_(pa.field(name='element', type=_ACTIVITIES))),
        ('generated_remaining_time_minutes', pa.list_(pa.field(name='element', type=pa.float64()))),
        # The suffix written from the mean of `p(z | prefix)`: the model's single answer, drawn once
        # per prefix and the only column comparable against a model that does not sample.
        ('point_activities', _ACTIVITIES),
        ('point_remaining_time_minutes', pa.float64()),
        ('true_activities', _ACTIVITIES),
        ('true_remaining_time_minutes', pa.float64()),
    ]
)


def open_generations(path: Path) -> pq.ParquetWriter:
    """Open a Parquet file for writing generations, creating its parent directories if needed.
    Args:
        path: The file to write, from `paths.generations_path`. Overwritten if it already exists.
    Returns:
        A writer bound to the generations schema, to be used as a context manager: closing it is
        what writes the file's footer.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    return pq.ParquetWriter(where=path, schema=_SCHEMA)


def table_from_generations(generations: list[Generation]) -> pa.Table:
    """Lay one batch's generations out as a table, one row per prefix.

    Args:
        generations: The model's answers, in the order `generate_batch` returned them.
    Returns:
        The rows as one table, ready to be written as a row group.
    """
    # Dicts here because they are what arrow's constructor takes, keyed by the schema's own names.
    rows = [
        {
            'case_id': generation.case_id,
            'prefix_len': generation.prefix_len,
            'truncated': generation.truncated,
            'prefix_activities': generation.prefix_activities,
            'generated_activities': [sample.activities for sample in generation.samples],
            'generated_remaining_time_minutes': [
                sample.remaining_time_minutes for sample in generation.samples
            ],
            'point_activities': generation.point.activities,
            'point_remaining_time_minutes': generation.point.remaining_time_minutes,
            'true_activities': generation.truth.activities,
            'true_remaining_time_minutes': generation.truth.remaining_time_minutes,
        }
        for generation in generations
    ]
    return pa.Table.from_pylist(mapping=rows, schema=_SCHEMA)


def read_generations(path: Path) -> Iterator[Generation]:
    """Read a generations file back one prefix at a time.

    The inverse of `table_from_generations` over a whole file. One row group is decoded at a time,
    so what it costs to read is set by the batch a run wrote rather than by the size of the split.
    A prefix cannot straddle a row group, since a row holds one, which is what lets a reader stop
    at any group boundary without buffering.

    Args:
        path: The generations file to read, from `paths.generations_path`.
    Yields:
        The generation for each prefix, in the order they were written.
    """
    with pq.ParquetFile(path) as parquet:
        for row_group in range(parquet.num_row_groups):
            frame = parquet.read_row_group(row_group).to_pandas()
            for _, row in frame.iterrows():
                yield _generation_from_row(row)


def _generation_from_row(row: pd.Series) -> Generation:
    """Read one prefix's generation back out of the row a generations file holds it as.

    What lets a written file be scored through the same `score_generation` a training run reports
    from.

    Args:
        row: One row of a generations file, holding a prefix and every suffix drawn for it.
    Returns:
        The same generation `generate_batch` produced. Parquet hands the activity columns back as
        arrays, so they are copied into lists: `DecodedEvents` promises lists, and the edit
        distance that reads them is quicker to index for it. Copying is also what lets the row
        group behind them be dropped once the prefix has been scored.
    """
    return Generation(
        case_id=str(row.case_id),
        prefix_activities=list(row.prefix_activities),
        truncated=bool(row.truncated),
        samples=[
            DecodedEvents(
                activities=list(activities),
                remaining_time_minutes=float(remaining_time_minutes),
            )
            for activities, remaining_time_minutes in zip(
                row.generated_activities, row.generated_remaining_time_minutes, strict=True
            )
        ],
        point=DecodedEvents(
            activities=list(row.point_activities),
            remaining_time_minutes=float(row.point_remaining_time_minutes),
        ),
        truth=DecodedEvents(
            activities=list(row.true_activities),
            remaining_time_minutes=float(row.true_remaining_time_minutes),
        ),
    )
