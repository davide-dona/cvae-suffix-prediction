from .base import StrictModel
from .data import DataConfig, DeclareConfig
from .inference import InferenceConfig
from .model import ModelConfig
from .training import DataLoaderConfig, EarlyStoppingConfig, OptimizerConfig, TrainingConfig


class ExperimentConfig(StrictModel):
    """Top-level config, the single object loaded from YAML.

    Pass sub-sections (e.g. `cfg.model.encoder`, `cfg.optimizer`) into functions rather than this
    whole object, so each function only depends on the parameters it actually uses. There is no
    top-level `loss` section: it is CVAE-only, so it lives on `model` instead, alongside `prior`
    and `latent`.
    """

    seed: int

    data: DataConfig
    declare: DeclareConfig
    dataloader: DataLoaderConfig
    model: ModelConfig
    optimizer: OptimizerConfig
    training: TrainingConfig
    early_stopping: EarlyStoppingConfig
    inference: InferenceConfig


class DatasetConfig(StrictModel):
    """The model-independent parts of an experiment config, for pipelines that never read a
    model-dependent value.
    """

    data: DataConfig
    declare: DeclareConfig
