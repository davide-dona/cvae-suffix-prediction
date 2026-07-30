from pathlib import Path
from typing import Union
import torch
from torch import nn

from src.configs.dataset_info import DatasetInfo
from src.configs.schema import ModelConfig
from src.model.transformer_cvae import TransformerCVAE

def best_model_path(best_model_dir: Union[str, Path], run_name: str) -> Path:
    """
    Where the best step of a run is kept: `<best_model_dir>/<run_name>.pt`.

    One file per run, overwritten as the best improves, so the directory holds every run's
    result side by side and nothing has to be picked out of a history.
    """
    return Path(best_model_dir) / f'{run_name}.pt'


def latest_best_model_path(best_model_dir: Union[str, Path], run_prefix: str) -> Path:
    """
    The best model of the most recent run of one config.

    `pipelines/train.py` names a run `<dataset>/<experiment_name>-<timestamp>`, so every run of
    one config differs from the others only in that timestamp, and the format sorts them in
    start order. The last one is the run most recently started with this config, which is the
    one worth predicting with unless told otherwise.

    Args:
        best_model_dir: Where `best_model_path` writes.
        run_prefix: A run name without its timestamp, `<dataset>/<experiment_name>`.
    Returns:
        The path of the newest matching run's best model.
    Raises:
        FileNotFoundError: If this config has no trained model yet.
    """
    candidates = sorted(Path(best_model_dir).glob(f'{run_prefix}-*.pt'))
    if not candidates:
        raise FileNotFoundError(
            f'no trained model matching {Path(best_model_dir) / run_prefix}-<timestamp>.pt. '
            'Train one first, or name a checkpoint explicitly.'
        )
    return candidates[-1]


def save_checkpoint(
    model: nn.Module,
    *,
    model_config: dict,
    step: int,
    val_loss: float,
    path: Union[str, Path],
) -> Path:
    """
    Save a model checkpoint in the schema `build_model_from_checkpoint` expects.

    The full config travels with the weights, so the same model can be rebuilt later without
    being told a single hyperparameter. The data-derived dimensions are deliberately not
    stored: they come from the `DatasetInfo` of the dataset the model is being used on, which
    cannot then drift from the weights.

    The write goes to a temporary file and is then moved onto `path`, which is atomic within a
    directory. A best-model file is overwritten every time the best improves, so a run killed
    mid-write would otherwise leave a truncated file where the only copy of the best model was.

    Args:
        model: The model whose weights to save.
        model_config: Its `ModelConfig`, dumped to plain data.
        step: The optimizer step the weights are from. The best-model filename does not say,
            so the file has to.
        val_loss: That step's prior-path validation loss.
        path: Where to write, parent directories included.
    Returns:
        The path written to.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    temp_path = path.with_name(f'{path.name}.tmp')
    torch.save(
        obj={
            'model_config': model_config,
            'model_state_dict': model.state_dict(),
            'step': step,
            'val_loss': val_loss,
        },
        f=temp_path,
    )
    temp_path.replace(target=path)
    return path


def load_checkpoint(model_path: Union[str, Path]) -> dict:
    """Read a checkpoint file written by `save_checkpoint`."""
    return torch.load(f=Path(model_path), map_location='cpu', weights_only=False)


def build_model_from_checkpoint(
    checkpoint: dict, dataset_info: DatasetInfo, *, device: str = 'cpu'
) -> TransformerCVAE:
    """
    Rebuild the model a checkpoint holds, with its weights loaded.

    Only the config and the weights are read; the `step` and `val_loss` a checkpoint also
    carries describe the run it came from and say nothing about how to rebuild it.

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

    model = TransformerCVAE(config=config, dataset_info=dataset_info).to(device=device)
    model.load_state_dict(state_dict=checkpoint['model_state_dict'])
    model.eval()
    return model
