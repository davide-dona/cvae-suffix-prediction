import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path
from typing import Self

import pandas as pd
from pydantic import TypeAdapter, ValidationError

from src.evaluation.scores import COMPARABLE_METRICS
from src.evaluation.summary import EvaluationSummary, LengthSummary, flatten_scores
from src.identity import RunIdentity, group_by_model


@dataclass(frozen=True)
class EvaluationReport:
    """Everything one evaluation produced, under the identity of the run it scored."""

    run: RunIdentity
    summary: EvaluationSummary

    @classmethod
    def read(cls, path: str | Path) -> Self:
        """Read a report back from the JSON `write` produced.

        Args:
            path: The report to read, e.g. from `paths.EVALUATION`.
        Returns:
            The report, validated against the current schema so a report written before a metric
            was added or renamed fails here rather than on a missing key further down.
        """
        return _ADAPTER.validate_json(Path(path).read_bytes())

    def write(self, path: str | Path) -> Path:
        """
        Write the report as JSON.

        Args:
            path: Where to write, its directory already made, from `paths.EVALUATION.prepare`.
        Returns:
            The path written to.
        """
        path = Path(path)
        path.write_text(json.dumps(asdict(self), indent=4))
        return path


# Built once, since a `TypeAdapter` compiles the schema it validates against.
_ADAPTER = TypeAdapter(EvaluationReport)


class Axis(StrEnum):
    """The three breakdowns a metric is read against.

    A figure names the one it draws, and it is what the `axis` column of `read_reports` holds:
    the evaluation as a whole, or one length at a time, cut either at the prefix or at the
    ground-truth suffix.
    """

    OVERALL = 'overall'
    PREFIX = 'prefix'
    SUFFIX = 'suffix'


# What every figure and every table reads. One row is one metric of one run, so a metric added to
# the scores reaches the figures without a change here. `prefixes` is how many prefixes that row's
# own mean was taken over, which for a metric of `COMPARABLE_METRICS` is fewer than the breakdown
# holds.
REPORT_COLUMNS = ('dataset', 'model', 'axis', 'length', 'prefixes', 'metric', 'value')


def _over(summary: EvaluationSummary | LengthSummary, metric: str) -> int:
    """How many prefixes one metric's mean was taken over.

    Args:
        summary: The breakdown the metric was read off.
        metric: Which of the report's numbers it is.
    Returns:
        The prefixes it holds, or the ones the distributional family is read over where the
        metric is one of that family's, which is what `summary.compared` counts.
    """
    return summary.compared if metric in COMPARABLE_METRICS else summary.prefixes


def _rows(report: EvaluationReport) -> list[dict[str, object]]:
    """Lay one report out as one row per metric per breakdown."""
    run, summary = report.run, report.summary
    identity = {'dataset': run.dataset, 'model': run.model}

    rows: list[dict[str, object]] = [
        identity
        | {
            'axis': Axis.OVERALL,
            'length': None,
            'prefixes': _over(summary, metric),
            'metric': metric,
            'value': value,
        }
        for metric, value in flatten_scores(summary).items()
    ]
    breakdowns = ((Axis.PREFIX, summary.by_prefix_length), (Axis.SUFFIX, summary.by_suffix_length))
    rows.extend(
        identity
        | {
            'axis': axis,
            'length': entry.length,
            'prefixes': _over(entry, metric),
            'metric': metric,
            'value': value,
        }
        for axis, breakdown in breakdowns
        for entry in breakdown
        for metric, value in flatten_scores(entry).items()
    )
    return rows


def read_reports(files: Sequence[Path]) -> pd.DataFrame:
    """Read a set of evaluation reports into the frame every figure and table is drawn from.

    Args:
        files: The reports to compare, from `pipelines.evaluate`. Each says which run wrote it, so
            they may come from any number of logs and models.
    Returns:
        One row per metric per breakdown, under `REPORT_COLUMNS`. `length` and `prefixes` are
        nullable integers, `length` being null on the `Axis.OVERALL` rows.
    Raises:
        ValueError: If a file is not an evaluation report, or if one log is given two runs of the
            same model, which would draw two lines under one name.
    """
    reports: list[tuple[Path, EvaluationReport]] = []
    for file in files:
        try:
            reports.append((file, EvaluationReport.read(file)))
        except ValidationError as error:
            # A swept directory reads files nobody typed, and the schema error names none of them.
            raise ValueError(f'{file} is not an evaluation report: {error}') from error

    # Called for the check alone: a log given two runs of one model would draw two lines under one
    # name. The rows below are laid out per report either way, so the grouping itself is not needed.
    group_by_model((report.run, file) for file, report in reports)

    rows = [row for _, report in reports for row in _rows(report)]
    frame = pd.DataFrame(rows, columns=list(REPORT_COLUMNS))
    return frame.astype({'length': 'Int64', 'prefixes': 'Int64', 'value': 'float64'})
