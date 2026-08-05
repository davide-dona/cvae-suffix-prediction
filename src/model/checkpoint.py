from pathlib import Path
import torch
from torch import nn


def checkpoint_path(directory: str | Path, run_name: str) -> Path:
    """
    Where a run keeps a checkpoint: `<directory>/<run_name>.pt`.

    One file per run, overwritten as the run goes on, so a directory holds every run's
    checkpoint side by side and nothing has to be picked out of a history. Used with
    `training.best_model_dir` for the best step and `training.checkpoint_dir` for the last one.

    Args:
        directory: The directory the file belongs in.
        run_name: The run the checkpoint is of.
    Returns:
        The path to write to or read from.
    """
    return Path(directory) / f'{run_name}.pt'


def latest_best_model_path(best_model_dir: str | Path, run_prefix: str) -> Path:
    """
    The best model of the most recent run of one config.

    `pipelines/train.py` names a run `<dataset>/<experiment_name>-<timestamp>`, so every run of
    one config differs from the others only in that timestamp, and the format sorts them in
    start order. The last one is the run most recently started with this config, which is the
    one worth generating with unless told otherwise.

    Args:
        best_model_dir: The directory the best steps are written to.
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
    experiment_config: dict,
    step: int,
    selection_score: float,
    run_name: str,
    optimizer_state: dict,
    early_stopping_state: dict,
    rng_state: dict,
    path: str | Path,
) -> Path:
    """
    Save a checkpoint holding everything a run needs to be rebuilt or picked back up.

    The run's config travels with the weights, so the same model can be rebuilt later without
    being told a single hyperparameter, and the rest of the training state travels with them,
    so `pipelines/train.py -r` can carry on from here rather than from scratch.

    Args:
        model: The model whose weights to save.
        experiment_config: The run's whole `ExperimentConfig`, dumped to plain data, so that
            resuming needs nothing but this file. Its `model` section is written out beside it,
            since that is all `TransformerCVAE.from_checkpoint` reads.
        step: The optimizer step the weights are from. The filename does not say, so the file
            has to.
        selection_score: That step's generation score, the number the best is chosen on.
        run_name: The run these weights belong to, so a resumed run keeps writing to the same
            TensorBoard directory and the same files.
        optimizer_state: The optimizer's `state_dict`.
        early_stopping_state: The `EarlyStopper`'s `state_dict`, which carries the best score
            seen as well as the patience count.
        rng_state: The random streams, as captured by `training/train.py`'s `rng_state`.
        path: Where to write, parent directories included.
    Returns:
        The path written to.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Written aside and moved into place, so a run interrupted mid-save leaves the last good
    # checkpoint intact rather than a truncated file where one is expected.
    temp_path = path.with_name(f'{path.name}.tmp')
    torch.save(
        obj={
            'model_config': experiment_config['model'],
            'model_state_dict': model.state_dict(),
            'step': step,
            'selection_score': selection_score,
            'experiment_config': experiment_config,
            'run_name': run_name,
            'optimizer_state': optimizer_state,
            'early_stopping_state': early_stopping_state,
            'rng_state': rng_state,
        },
        f=temp_path,
    )
    temp_path.replace(target=path)
    return path


def load_checkpoint(model_path: str | Path) -> dict:
    """Read a checkpoint file written by `save_checkpoint`."""
    return torch.load(f=Path(model_path), map_location='cpu', weights_only=False)
