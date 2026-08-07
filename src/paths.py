from enum import StrEnum
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / 'data'
OUTPUTS_DIR = ROOT / 'outputs'


class Split(StrEnum):
    """The three splits preprocessing cuts a log into, named as their files on disk are.

    Iterating the members is what says a dataset has been preprocessed and what a pipeline
    reads, so the three names are written down here and nowhere else.
    """

    TRAIN = 'train'
    VAL = 'val'
    TEST = 'test'


def original_log(dataset: str) -> Path:
    """The raw log a dataset starts from, the one file preprocessing reads."""
    return DATA_DIR / dataset / 'original.csv'


def split_path(dataset: str, split: Split) -> Path:
    """One preprocessed split of a dataset.

    Args:
        dataset: The dataset the split belongs to.
        split: Which of the three to name.
    Returns:
        The path to that split's file.
    """
    return DATA_DIR / dataset / 'processed' / f'{split}.csv'


def codec_path(dataset: str) -> Path:
    """A dataset's fitted codec, next to the splits it was fit on."""
    return DATA_DIR / dataset / 'processed' / 'dataset.json'


def declare_model_path(dataset: str) -> Path:
    """A dataset's declarative model, discovered from its train split at preprocessing time."""
    return DATA_DIR / dataset / 'declare' / 'model.decl'


def tensorboard_dir(run_name: str) -> Path:
    """A run's TensorBoard events. One directory is one run, so the name has to be unique."""
    return OUTPUTS_DIR / 'tensorboard' / run_name


def checkpoint_path(run_name: str) -> Path:
    """A run's last validated step, overwritten every validation; what `--resume` reads."""
    return OUTPUTS_DIR / 'checkpoints' / f'{run_name}.pt'


def best_model_path(run_name: str) -> Path:
    """A run's best step, overwritten whenever the selection score improves."""
    return OUTPUTS_DIR / 'best-models' / f'{run_name}.pt'


def generations_path(run_name: str) -> Path:
    """The suffixes a run generated for the test split."""
    return OUTPUTS_DIR / 'generations' / f'{run_name}.parquet'


def evaluation_path(run_name: str) -> Path:
    """The report those generations scored."""
    return OUTPUTS_DIR / 'eval' / f'{run_name}.json'
