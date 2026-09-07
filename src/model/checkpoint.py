from collections.abc import Iterable
from dataclasses import asdict
from pathlib import Path

import torch
from torch import nn

from src.identity import RunIdentity

# What rebuilding the model a checkpoint holds reads, and so what `model_from_checkpoint`
# refuses to guess at.
MODEL_KEYS = ('model_config', 'model_state_dict')

# The whole of a checkpoint: the two keys `model_from_checkpoint` reads, the one
# `pipelines/generate.py` names its output after, and two that say which step of which run this is
# and how well it scored. A run is never carried on from, so there is no optimizer, early-stopping
# or random state here and no second, fatter kind of checkpoint to tell this one apart from: the
# file on disk is already what gets published and versioned.
CHECKPOINT_KEYS = (
    'model_config',
    'model_state_dict',
    'run',
    'experiment_config',
    'step',
    'selection_score',
)


def require_keys(
    checkpoint: dict, keys: Iterable[str], *, subject: str = 'checkpoint', purpose: str, remedy: str
) -> None:
    """Check that a checkpoint carries everything one use of it reads.

    Args:
        checkpoint: What `load_checkpoint` read.
        keys: The keys that use reads, named in the error in the order given.
        subject: What the error calls the file, e.g. the path it was read from.
        purpose: What it was about to be used for, e.g. `published`.
        remedy: What to do instead, one sentence.
    Raises:
        ValueError: If any key is missing, naming every one of them.
    """
    missing = [key for key in keys if key not in checkpoint]
    if missing:
        raise ValueError(
            f'{subject} is missing {", ".join(missing)}, so it cannot be {purpose}. {remedy}'
        )


def save_checkpoint(
    model: nn.Module,
    *,
    experiment_config: dict,
    step: int,
    selection_score: float,
    run: RunIdentity,
    path: str | Path,
) -> Path:
    """
    Save a checkpoint holding everything rebuilding this model needs.

    The run's config travels with the weights, so the same model can be rebuilt later without
    being told a single hyperparameter. Nothing beyond that is kept: a run that ends, for whatever
    reason, is over, and the file it leaves is read only to generate from or to publish.

    Args:
        model: The model whose weights to save.
        experiment_config: The run's whole `ExperimentConfig`, dumped to plain data, so that
            rebuilding needs nothing but this file. Its `model` section is written out beside it,
            since that is all `model_from_checkpoint` reads.
        step: The optimizer step the weights are from. The filename does not say, so the file
            has to.
        selection_score: That step's generation score, the number the best is chosen on.
        run: Which run these weights belong to, so the generations they produce are named after
            the run rather than after the file the weights were read from.
        path: Where to write, its directory already made, from `paths.BEST_CHECKPOINT.prepare`.
    Returns:
        The path written to.
    """
    path = Path(path)

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
            'run': asdict(run),
        },
        f=temp_path,
    )
    temp_path.replace(target=path)
    return path


def load_checkpoint(model_path: str | Path) -> dict:
    """Read a checkpoint file written by `save_checkpoint`."""
    return torch.load(f=Path(model_path), map_location='cpu', weights_only=False)
