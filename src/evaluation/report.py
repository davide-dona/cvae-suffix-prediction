import json
from dataclasses import asdict, dataclass
from pathlib import Path

from src.evaluation.accuracy import AccuracyMetrics
from src.evaluation.conformance import ConformanceMetrics


@dataclass(frozen=True)
class EvaluationReport:
    """Everything one evaluation produced, and what it was measured on.

    The counts are part of the result rather than a log line: an average is not comparable
    across runs without knowing how many prefixes went into it, and how many were left out.
    """
    run_name: str
    pairs: int
    cases: int
    samples_per_prefix: int
    truncated_pairs_excluded: int
    accuracy: AccuracyMetrics
    conformance: ConformanceMetrics

    def write(self, path: str | Path) -> Path:
        """
        Write the report as JSON, creating parent directories.

        Args:
            path: Where to write, e.g. the generations file's own path with a `.json` suffix.
        Returns:
            The path written to.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2))
        return path
