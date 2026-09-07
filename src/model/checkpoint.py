"""Self-contained inference checkpoints, atomically replaced on improvement."""

from collections.abc import Iterable
from pathlib import Path

import torch
from torch import nn

MODEL_KEYS = ('config', 'model_state_dict')
CHECKPOINT_KEYS = (
    *MODEL_KEYS,
    'step',
    'selection_score',
    'selection_metric',
    'selection_direction',
)


def require_keys(
    checkpoint: dict, keys: Iterable[str], *, subject: str = 'checkpoint', purpose: str, remedy: str
) -> None:
    missing = [key for key in keys if key not in checkpoint]
    if missing:
        raise ValueError(
            f'{subject} is missing {", ".join(missing)}; cannot be {purpose}. {remedy}'
        )


def save_checkpoint(
    model: nn.Module,
    *,
    config: dict,
    step: int,
    selection_score: float,
    wandb_id: str | None,
    path: Path,
) -> Path:
    temp = path.with_suffix('.pt.tmp')
    torch.save(
        obj={
            'config': config,
            'model_state_dict': model.state_dict(),
            'step': step,
            'selection_score': selection_score,
            'selection_metric': 'energy_score',
            'selection_direction': 'min',
            'wandb_id': wandb_id,
        },
        f=temp,
    )
    temp.replace(path)
    return path


def load_checkpoint(model_path: str | Path) -> dict:
    checkpoint = torch.load(f=Path(model_path), map_location='cpu', weights_only=True)
    require_keys(checkpoint, CHECKPOINT_KEYS, purpose='loaded', remedy='Train a new checkpoint.')
    return checkpoint
