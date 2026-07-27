from pathlib import Path
import yaml
from .schema import ExperimentConfig


def _deep_merge(base: dict, override: dict) -> dict:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(merged.get(key), dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(path: str | Path) -> ExperimentConfig:
    """Load and validate an experiment config, merged over the sibling base.yaml."""
    path = Path(path)
    with (path.parent / "base.yaml").open("r") as f:
        base = yaml.safe_load(f)
    with path.open("r") as f:
        override = yaml.safe_load(f)
    return ExperimentConfig.model_validate(_deep_merge(base, override))
