from dataclasses import dataclass

import pandas as pd

from src.registry import Registry


@dataclass(frozen=True)
class ModelStyle:
    """Label and visual style shared by a model's figures."""

    label: str
    color: str
    marker: str
    linestyle: str


# Style for observed log values.
LOG_STYLE = ModelStyle(label='Log', color='#737373', marker='o', linestyle='-.')

# Registered model labels and styles.
MODELS = Registry[ModelStyle](
    kind='model',
    where='MODELS in src/visualization/labels/models.py',
    entries={
        'cvae': ModelStyle(label='CVAE', color='#3B7EA1', marker='*', linestyle='-'),
        'transformer': ModelStyle(label='Transformer', color='#A05A4B', marker='D', linestyle=':'),
        'u-ed-lstm': ModelStyle(label='U-ED-LSTM', color='#7A7A7A', marker='s', linestyle='--'),
    },
)


def _reported(model: str) -> str:
    """Return the canonical name for a model style.

    Args:
        model: Registered model name.

    Returns:
        First registered name using the same style.
    """
    style = MODELS[model]
    return next(name for name, declared in MODELS.entries.items() if declared == style)


def reported_models(frame: pd.DataFrame) -> pd.DataFrame:
    """Map models to their reported names.

    Args:
        frame: Report rows containing model names.

    Returns:
        Copy with canonical model names.
    """
    return frame.assign(model=frame['model'].map(_reported))
