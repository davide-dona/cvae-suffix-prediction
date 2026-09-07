from pathlib import Path

import yaml

from .schema import DatasetConfig, ExperimentConfig, SamplingConfig


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge two dicts, with override taking precedence.
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


def _read(path: Path) -> dict:
    """Read one layer of a config."""
    with path.open('r') as f:
        return yaml.safe_load(f)


def load_config(model: Path, config: Path) -> ExperimentConfig:
    """Load and validate an experiment config, merging the model over the dataset.

    The model is a layer of its own because an architecture is chosen independently of the log it
    is run on: one file per architecture against one per dataset, rather than a copy of each
    model's widths in every dataset config. It also carries every setting that does not vary with
    the dataset - the training loop, the optimizer, and, for the CVAE, the loss - so a run is
    fully described by these two files together.

    Args:
        model: Path to the model config YAML, e.g. config/models/cvae.yaml. Its `model.kind` is
            what says which architecture the run builds.
        config: Path to the dataset config YAML, e.g. config/datasets/bpic17.yaml.
    Returns:
        The validated config.
    """
    merged = _deep_merge(_read(model), _read(config))
    return ExperimentConfig.model_validate(merged)


def load_dataset_config(config: Path) -> DatasetConfig:
    """Load and validate a dataset config, for pipelines that never read a model-dependent value.
    Args:
        config: Path to the dataset config YAML, e.g. config/datasets/bpic17.yaml.
    Returns:
        The validated `data`/`declare` sections.
    """
    return DatasetConfig.model_validate(_read(config))


def load_generation_config(
    experiment_config: dict,
    *,
    device: str | None,
    num_samples: int | None,
    sampling: SamplingConfig | None,
) -> ExperimentConfig:
    """Load and validate the config a checkpoint is generated from.
    Args:
        experiment_config: The run's config as the checkpoint stores it, from
            `checkpoint['experiment_config']`.
        device: Overrides the run's own `training.device`, e.g. to generate on a different
            machine than the one it trained on. `None` keeps it.
        num_samples: How many suffixes to draw per prefix, replacing the run's own
            `inference.evaluation_samples`, or `None` to keep it.
        sampling: Overrides the run's own `model.sampling`, which is how a sampler chosen after
            training by `pipelines.tune` reaches a generation without the checkpoint being
            rewritten. `None` keeps what the run trained under. An architecture whose config has
            no `sampling` section rejects this rather than ignoring it, there being no read of
            its heads for a temperature to shape.
    Returns:
        The validated config.
    Raises:
        pydantic.ValidationError: If the merged config is not a valid experiment: `num_samples`
            below `InferenceConfig`'s floor, or a `sampling` override against an architecture
            that declares none.
    """
    merged = experiment_config
    if device is not None:
        # Merged in rather than set on the config, which is frozen once validated.
        merged = _deep_merge(merged, {'training': {'device': device}})
    if num_samples is not None:
        merged = _deep_merge(merged, {'inference': {'evaluation_samples': num_samples}})
    if sampling is not None:
        merged = _deep_merge(merged, {'model': {'sampling': sampling.model_dump()}})
    return ExperimentConfig.model_validate(merged)
