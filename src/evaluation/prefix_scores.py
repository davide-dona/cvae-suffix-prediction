from collections.abc import Iterable, Iterator, Sequence
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src import paths
from src.evaluation.scores import METRICS
from src.evaluation.summary import PrefixSummary, flatten_scores
from src.artifacts import group_by_model, read_metadata, with_metadata
from src.inference.generation_store import PrefixKey

# Prefixes buffered in each Parquet row group.
BLOCK = 16_384

# Per-prefix identity and reporting lengths.
PREFIX_SCORE_KEYS = ('case_id', 'prefix_len', 'suffix_len')

# Per-prefix score schema; metric columns derive from `METRICS`.
_SCHEMA = pa.schema(
    [
        ('case_id', pa.large_string()),
        ('prefix_len', pa.int64()),
        ('suffix_len', pa.int64()),
        *((key, pa.float64()) for key in METRICS.entries),
    ]
)


def stream_prefix_scores(
    summaries: Iterable[PrefixSummary],
    keys: Sequence[PrefixKey],
    *,
    path: Path,
    metadata: dict[str, str],
) -> Iterator[PrefixSummary]:
    """Write per-prefix scores while yielding the original summaries.

    Args:
        summaries: Prefix summaries in generation order.
        keys: Prefix identities in the same order.
        path: Destination Parquet path.
        metadata: Identity with_metadata into the Parquet schema.

    Yields:
        Each input summary unchanged.

    Raises:
        ValueError: If summaries and keys have different lengths.
    """
    columns: dict[str, list] = {name: [] for name in _SCHEMA.names}

    def flush(writer: pq.ParquetWriter) -> None:
        if not columns['case_id']:
            return
        writer.write_table(pa.Table.from_pydict(mapping=columns, schema=_SCHEMA))
        for values in columns.values():
            values.clear()

    with pq.ParquetWriter(where=path, schema=with_metadata(_SCHEMA, metadata)) as writer:
        for summary, (case_id, prefix_len) in zip(summaries, keys, strict=True):
            columns['case_id'].append(case_id)
            columns['prefix_len'].append(prefix_len)
            columns['suffix_len'].append(summary.suffix_len)
            for name, value in flatten_scores(summary).items():
                columns[name].append(value)
            if len(columns['case_id']) >= BLOCK:
                flush(writer)
            yield summary
        flush(writer)


def read_prefix_scores(path: Path, *, columns: Sequence[str] | None = None) -> pd.DataFrame:
    """Read all or selected per-prefix score columns.

    Args:
        path: Per-prefix score Parquet path.
        columns: Optional columns to read.

    Returns:
        Per-prefix score dataframe.
    """
    wanted = None if columns is None else list(columns)
    return pq.read_table(source=path, columns=wanted).to_pandas()


def require_columns(path: Path, columns: Sequence[str]) -> None:
    """Require the requested columns in a score file's schema.

    Args:
        path: Per-prefix score Parquet path.
        columns: Required column names.

    Raises:
        ValueError: If a required column is absent.
    """
    absent = [key for key in columns if key not in pq.read_schema(where=path).names]
    if absent:
        raise ValueError(
            f'{path} carries no {", ".join(absent)}, so it predates the scores now read. '
            'Score it again with `python -m pipelines.evaluate`.'
        )


def score_files(reports: Sequence[Path]) -> dict[str, dict[str, Path]]:
    """Find score files beside reports and group them by dataset.

    Args:
        reports: Evaluation report paths.

    Returns:
        Score files keyed by dataset and model.

    Raises:
        ValueError: If a score file is missing, invalid, or duplicates a model metadata.
    """
    files = [(report, report.with_name('prefix_scores.parquet')) for report in reports]
    missing = [str(report) for report, scores in files if not scores.exists()]
    if missing:
        raise ValueError(
            'no per-prefix scores beside these reports, so how far a metadata’s means could be off '
            'cannot be read:\n  '
            + '\n  '.join(missing)
            + '\nScore them again with `python -m pipelines.evaluate`, which writes them beside '
            'the report.'
        )

    runs: list[tuple[Path]] = []
    for _, scores in files:
        try:
            with pq.ParquetFile(scores) as parquet:
                runs.append((read_metadata(parquet), scores))
        except (ValueError, TypeError, KeyError) as error:
            # Preserve filesystem errors as-is.
            raise ValueError(f'{scores} is not a per-prefix scores file: {error}') from error
    return group_by_model(runs)
