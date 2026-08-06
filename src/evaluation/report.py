import json
from dataclasses import asdict, dataclass
from pathlib import Path

from src.evaluation.metrics import EvaluationMetrics


@dataclass(frozen=True)
class EvaluationReport:
    """Everything one evaluation produced, under the name of the run it scored."""
    run_name: str
    metrics: EvaluationMetrics

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
        path.write_text(json.dumps(asdict(self), indent=4))
        return path
