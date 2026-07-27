from pathlib import Path
from typing import Union

import torch
from torch import nn

from src.configs.dataset_info import DatasetInfo
from src.configs.schema import ModelConfig
from src.models.attention_cvae import AttentionCVAE

#: Subdirectory (relative to a run's directory) that per-epoch checkpoints are saved
#: under, one per improving best.
BEST_MODELS_DIRNAME = 'best-models'


def best_model_path_for_epoch(checkpoint_dir: Union[str, Path], epoch: int) -> Path:
    """The path `save_checkpoint` writes to for a given run and epoch."""
    return Path(checkpoint_dir) / BEST_MODELS_DIRNAME / f'best-model-epoch-{epoch}.pt'


def save_checkpoint(model: nn.Module, *, model_config: dict, path: Union[str, Path]) -> Path:
    """
    Save a model checkpoint in the schema `build_model_from_checkpoint` expects.

    The full config travels with the weights, so the same model can be rebuilt later without
    being told a single hyperparameter. The data-derived dimensions are deliberately not
    stored: they come from the `DatasetInfo` of the dataset the model is being used on, which
    cannot then drift from the weights.

    Args:
        model: The model whose weights to save.
        model_config: Its `ModelConfig`, dumped to plain data.
        path: Where to write, parent directories included.
    Returns:
        The path written to.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    torch.save({'model_config': model_config, 'model_state_dict': model.state_dict()}, path)
    return path


def load_checkpoint(model_path: Union[str, Path]) -> dict:
    """Read a checkpoint file written by `save_checkpoint`."""
    return torch.load(Path(model_path), map_location='cpu', weights_only=False)


def build_model_from_checkpoint(
    checkpoint: dict, dataset_info: DatasetInfo, *, device: str = 'cpu'
) -> AttentionCVAE:
    """
    Rebuild the model a checkpoint holds, with its weights loaded.

    Args:
        checkpoint: A checkpoint read by `load_checkpoint`.
        dataset_info: The dataset the model is to be used on, supplying the vocabulary
            sizes and sequence length it was built against.
        device: Where to place the model.
    Returns:
        The model, in evaluation mode.
    """
    missing = {'model_config', 'model_state_dict'} - checkpoint.keys()
    if missing:
        raise ValueError(f'checkpoint is missing {sorted(missing)}. Train the model again.')

    config = ModelConfig.model_validate(checkpoint['model_config'])

    model = AttentionCVAE(config, dataset_info).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model
