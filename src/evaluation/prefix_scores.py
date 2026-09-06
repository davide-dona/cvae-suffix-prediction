from collections.abc import Iterable, Iterator, Sequence
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src import paths
from src.evaluation.scores import METRICS
from src.evaluation.summary import PrefixSummary, flatten_scores
from src.identity import RunIdentity, group_by_model, read_run_identity, stamped
from src.inference.generation_store import PrefixKey

# How many prefixes one row group holds. A split of a quarter of a million is then a handful of
# blocks, and nothing beyond one block is ever buffered.
BLOCK = 16_384

# Which prefix a row scores, and the two lengths its scores are read against. The same key the
# generations are written under, so the rows of two runs of one log are matched on it.
PREFIX_SCORE_KEYS = ('case_id', 'prefix_len', 'suffix_len')

# The schema of the Parquet file that holds a run's per-prefix scores, one row per prefix. The
# metric columns are built from `METRICS` rather than listed, so a score added to a family reaches
# this file with no change here.
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
    run: RunIdentity,
) -> Iterator[PrefixSummary]:
    """Write each prefix's scores as they pass through, and yield them on unchanged.

    Args:
        summaries: Each prefix's scores, in the order the generations file holds them.
        keys: Which prefix each of them answers, in that same order, from `Generations.prefix_keys`.
            Paired positionally rather than carried on a `PrefixSummary`, which holds floats alone
            so that a worker never sends the log's own strings back.
        path: Where to write, its directory already made, from `paths.PREFIX_SCORES.prepare`.
            Overwritten if it already exists.
        run: The run these scores belong to, stamped into the file so a reader knows what it holds
            rather than guessing at it from the path.
    Yields:
        Every summary given, unchanged and in order.
    Raises:
        ValueError: If the scores and the keys are of different lengths, which would pair a
            prefix's scores to another prefix.
    """
    columns: dict[str, list] = {name: [] for name in _SCHEMA.names}

    def flush(writer: pq.ParquetWriter) -> None:
        if not columns['case_id']:
            return
        writer.write_table(pa.Table.from_pydict(mapping=columns, schema=_SCHEMA))
        for values in columns.values():
            values.clear()

    with pq.ParquetWriter(where=path, schema=stamped(_SCHEMA, run)) as writer:
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
    """Read a run's per-prefix scores back.

    Args:
        path: The file `stream_prefix_scores` wrote, from `paths.PREFIX_SCORES`.
        columns: Which columns to read, or `None` for all of them. A reader after a handful of
            metrics pulls those alone off a file that holds every one of them.
    Returns:
        One row per prefix, under `PREFIX_SCORE_KEYS` and one float column per metric of `METRICS`,
        or the subset `columns` named.
    """
    wanted = None if columns is None else list(columns)
    return pq.read_table(source=path, columns=wanted).to_pandas()


def require_columns(path: Path, columns: Sequence[str]) -> None:
    """Check that a scores file carries every column a reader is about to ask for.

    Checked against the schema rather than after the read, which would fail on the missing column
    with nothing to say about why it is missing.

    Args:
        path: The scores file, from `paths.PREFIX_SCORES`.
        columns: The columns the reader needs.
    Raises:
        ValueError: If any of them is absent, naming them, since a run scored before a metric
            existed carries every other column and reads as a file that simply predates it.
    """
    absent = [key for key in columns if key not in pq.read_schema(where=path).names]
    if absent:
        raise ValueError(
            f'{path} carries no {", ".join(absent)}, so it predates the scores now read. '
            'Score it again with `python -m pipelines.evaluate`.'
        )


def score_files(reports: Sequence[Path]) -> dict[str, dict[str, Path]]:
    """Find the per-prefix scores beside each report and group them by the log they belong to.

    Args:
        reports: The evaluation reports being read, from `pipelines.evaluate`.
    Returns:
        The scores file of each model, keyed by the log's own name.
    Raises:
        ValueError: If a report has no scores beside it, if one is not a scores file, or if one log
            is given two runs of the same model.
    """
    files = [(report, paths.PREFIX_SCORES.beside(report)) for report in reports]
    missing = [str(report) for report, scores in files if not scores.exists()]
    if missing:
        raise ValueError(
            'no per-prefix scores beside these reports, so how far a run’s means could be off '
            'cannot be read:\n  '
            + '\n  '.join(missing)
            + '\nScore them again with `python -m pipelines.evaluate`, which writes them beside '
            'the report.'
        )

    runs: list[tuple[RunIdentity, Path]] = []
    for _, scores in files:
        try:
            with pq.ParquetFile(scores) as parquet:
                runs.append((read_run_identity(parquet), scores))
        except (ValueError, TypeError, KeyError) as error:
            # An `OSError` is an unreadable disk, a real failure, and is left to surface as itself
            # rather than being relabelled as the wrong file type.
            raise ValueError(f'{scores} is not a per-prefix scores file: {error}') from error
    return group_by_model(runs)
