from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = ROOT / 'config'
DATA_DIR = ROOT / 'data'
OUTPUTS_DIR = ROOT / 'outputs'
PRETRAINED_DIR = ROOT / 'pretrained'
# Where the visualizations go
VISUAL_DIR = OUTPUTS_DIR / 'visual'
FIGURES_DIR = VISUAL_DIR / 'figures'
TABLES_DIR = VISUAL_DIR / 'tables'


def dataset_config(dataset: str) -> Path:
    """The config a dataset is preprocessed from, by the convention its files are named under.

    Named here because a missing preprocessing output has to say what to rerun; what a pipeline is
    actually given is a path typed on the command line, which is free not to be this one.

    Args:
        dataset: The dataset, as `data.name` in that config.
    Returns:
        The path that config is kept at.
    """
    return CONFIG_DIR / 'datasets' / f'{dataset}.yaml'
