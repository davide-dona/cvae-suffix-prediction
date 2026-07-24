from pathlib import Path
import yaml
from .schema import ExperimentConfig


def load_config(path: str | Path) -> ExperimentConfig:
    """Load and validate an experiment config from a YAML file."""
    with Path(path).open("r") as f:
        raw = yaml.safe_load(f)
    return ExperimentConfig.model_validate(raw)
