from pathlib import Path
import torch
from torch import nn


def best_model_path(best_model_dir: str | Path, run_name: str) -> Path:
    """
    Where the best step of a run is kept: `<best_model_dir>/<run_name>.pt`.

    One file per run, overwritten as the best improves, so the directory holds every run's
    result side by side and nothing has to be picked out of a history.
    """
    return Path(best_model_dir) / f'{run_name}.pt'


def latest_best_model_path(best_model_dir: str | Path, run_prefix: str) -> Path:
    """
    The best model of the most recent run of one config.

    `pipelines/train.py` names a run `<dataset>/<experiment_name>-<timestamp>`, so every run of
    one config differs from the others only in that timestamp, and the format sorts them in
    start order. The last one is the run most recently started with this config, which is the
    one worth generating with unless told otherwise.

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
    selection_score: float,
    path: str | Path,
) -> Path:
    """
    Save a model checkpoint in the schema `TransformerCVAE.from_checkpoint` expects.

    The full config travels with the weights, so the same model can be rebuilt later without
    being told a single hyperparameter.

    Args:
        model: The model whose weights to save.
        model_config: Its `ModelConfig`, dumped to plain data.
        step: The optimizer step the weights are from. The best-model filename does not say,
            so the file has to.
        selection_score: That step's generation score, the number it was chosen on.
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
            'selection_score': selection_score,
        },
        f=temp_path,
    )
    temp_path.replace(target=path)
    return path


def load_checkpoint(model_path: str | Path) -> dict:
    """Read a checkpoint file written by `save_checkpoint`."""
    return torch.load(f=Path(model_path), map_location='cpu', weights_only=False)
