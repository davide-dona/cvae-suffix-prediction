from pathlib import Path

import yaml

from .schema import ExperimentConfig


def _deep_merge(base: dict, override: dict) -> dict:
    """
    Recursively merge two dicts, with override taking precedence.
    Recursion allows nested dicts to be merged rather than replaced, so a config can override
    just one field of a nested section.
    Args:
        base: The base config dict.
        override: The override config dict.
    Returns:
        The merged config dict.
    """
    # Start with a copy of the base dict
    merged = dict(base)

    # For each key/value pair in the override dict
    for key, value in override.items():
        # If both are dicts, merge them recursively
        if isinstance(merged.get(key), dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            # Otherwise, override the base value with the override value
            merged[key] = value
    return merged


def load_config(path: str | Path) -> ExperimentConfig:
    """Load and validate an experiment config, merged over the sibling base.yaml.

    The filename is the dataset's identity: `config/sepsis.yaml` describes `sepsis`, and every
    path the pipelines read or write is derived from that name (see `src/paths.py`). It is filled
    into `data.name` here rather than written in the YAML, where it could disagree with the file
    holding it.

    Args:
        path: The dataset's config YAML.
    Returns:
        The validated config, with `data.name` set from `path`'s stem.
    """
    path = Path(path)
    # Load the base config and the override config
    with (path.parent / 'base.yaml').open('r') as f:
        base = yaml.safe_load(f)
    with path.open('r') as f:
        override = yaml.safe_load(f)
    # Merge the two configs, with the override taking precedence
    merged = _deep_merge(base, override)
    merged['data']['name'] = path.stem
    return ExperimentConfig.model_validate(merged)
