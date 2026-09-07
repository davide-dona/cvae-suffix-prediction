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
from src.artifacts import group_by_model


@dataclass(frozen=True)
class EvaluationReport:
    """Evaluation results for one metadata."""

    metadata: dict[str, str]
    summary: EvaluationSummary

    @classmethod
    def read(cls, path: str | Path) -> Self:
        """Read and validate a JSON evaluation report.

        Args:
            path: JSON report path.

        Returns:
            The validated report.
        """
        return _ADAPTER.validate_json(Path(path).read_bytes())

    def write(self, path: str | Path) -> Path:
        """Write the report as JSON.

        Args:
            path: Destination path.

        Returns:
            The written path.
        """
        path = Path(path)
        path.write_text(json.dumps(asdict(self), indent=4))
        return path


# Reuse the compiled validation schema.
_ADAPTER = TypeAdapter(EvaluationReport)


class Axis(StrEnum):
    """Metric aggregation levels."""

    OVERALL = 'overall'
    PREFIX = 'prefix'
    SUFFIX = 'suffix'


# Columns shared by report readers. Comparable metrics use fewer prefixes.
REPORT_COLUMNS = ('dataset', 'model', 'axis', 'length', 'prefixes', 'metric', 'value')


def _over(summary: EvaluationSummary | LengthSummary, metric: str) -> int:
    """Return the population used to aggregate a metric.

    Args:
        summary: Aggregate containing the metric.
        metric: Metric key.

    Returns:
        Prefix count used for the metric's mean.
    """
    return summary.compared if metric in COMPARABLE_METRICS else summary.prefixes


def _rows(report: EvaluationReport) -> list[dict[str, object]]:
    """Flatten a report into one row per metric and breakdown.

    Args:
        report: Evaluation report to flatten.

    Returns:
        Report rows in the shared tabular schema.
    """
    metadata, summary = report.metadata, report.summary
    identity = {'dataset': metadata['dataset'], 'model': metadata['model']}

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
    """Load reports into the dataframe consumed by tables and figures.

    Args:
        files: Evaluation report paths.

    Returns:
        One row per metric and aggregation level.

    Raises:
        ValueError: If a file is invalid or a dataset has duplicate model runs.
    """
    reports: list[tuple[Path, EvaluationReport]] = []
    for file in files:
        try:
            reports.append((file, EvaluationReport.read(file)))
        except ValidationError as error:
            # Add the path omitted by the schema error.
            raise ValueError(f'{file} is not an evaluation report: {error}') from error

    # Reject duplicate model runs for the same log.
    group_by_model((report.metadata, file) for file, report in reports)

    rows = [row for _, report in reports for row in _rows(report)]
    frame = pd.DataFrame(rows, columns=list(REPORT_COLUMNS))
    return frame.astype({'length': 'Int64', 'prefixes': 'Int64', 'value': 'float64'})
