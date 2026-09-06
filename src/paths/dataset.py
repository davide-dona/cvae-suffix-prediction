from dataclasses import dataclass
from pathlib import Path

from src.logs import Split
from src.paths.artifact import Artifact
from src.paths.locations import DATA_DIR, dataset_config


@dataclass(frozen=True)
class DatasetArtifact(Artifact):
    """A file belonging to one dataset, under `data/<dataset>/`."""

    relative: str  # where it sits under the dataset's own directory, e.g. `codec/dataset.json`

    def path(self, dataset: str) -> Path:
        """Where this dataset's file of this kind goes, whether or not anything is there."""
        return DATA_DIR / dataset / self.relative

    def require(self, dataset: str) -> Path:
        """That path, or a `FileNotFoundError` naming the dataset's config to rerun with."""
        return self._found(self.path(dataset), self.remedy.format(config=dataset_config(dataset)))

    def prepare(self, dataset: str) -> Path:
        """That path, with the directory it goes in created."""
        return self._made(self.path(dataset))


@dataclass(frozen=True)
class SplitArtifact(Artifact):
    """A file a dataset has one of per split, under a directory of its own.

    Keyed by split rather than by dataset alone: training selects checkpoints against the
    validation split and evaluation scores against the test split, and one standing in for the
    other would fold the held-out set into what gets kept.
    """

    subdirectory: str  # under the dataset's own directory, e.g. `processed`
    suffix: str  # what it is written with, e.g. `.csv`; the leaf is the split's own name

    def directory(self, dataset: str) -> Path:
        """Where every split's file of this kind sits, which is what a banner names."""
        return DATA_DIR / dataset / self.subdirectory

    def path(self, dataset: str, split: Split) -> Path:
        """Where one split's file of this kind goes, whether or not anything is there."""
        return self.directory(dataset) / f'{split}{self.suffix}'

    def require(self, dataset: str, split: Split) -> Path:
        """That path, or a `FileNotFoundError` naming the dataset's config to rerun with."""
        return self._found(
            self.path(dataset=dataset, split=split),
            self.remedy.format(config=dataset_config(dataset)),
        )

    def prepare(self, dataset: str, split: Split) -> Path:
        """That path, with the directory it goes in created."""
        return self._made(self.path(dataset=dataset, split=split))


# What preprocessing reads, and the four things it leaves behind.
ORIGINAL_LOG = DatasetArtifact(
    kind='original log',
    remedy='Put the raw log there first.',
    relative='original.csv',
)
PROCESSED_SPLIT = SplitArtifact(
    kind='split',
    remedy='Run `uv run python -m pipelines.preprocess -c {config}` first.',
    subdirectory='processed',
    suffix='.csv',
)
CODEC = DatasetArtifact(
    kind='dataset codec',
    remedy='Run `uv run python -m pipelines.preprocess -c {config}` first.',
    relative='codec/dataset.json',
)
CONTINUATIONS = SplitArtifact(
    kind='continuation index',
    remedy='Run `uv run python -m pipelines.preprocess -c {config}` first.',
    subdirectory='continuations',
    suffix='.parquet',
)
DECLARE_MODEL = DatasetArtifact(
    kind='declarative model',
    remedy='Run `uv run python -m pipelines.preprocess -c {config}` first.',
    relative='declare/model.decl',
)


def require_preprocessed(dataset: str) -> None:
    """Check that everything training and generation read from preprocessing is on disk, and say
    what is missing if it is not.

    A run reads all of it, so all of it is named at once rather than one rerun at a time.

    Args:
        dataset: The dataset to check.
    Raises:
        FileNotFoundError: If any preprocessing output is missing, naming every one of them.
    """
    outputs = [CODEC.path(dataset)]
    outputs += [PROCESSED_SPLIT.path(dataset=dataset, split=split) for split in Split]

    missing = [output for output in outputs if not output.exists()]
    if missing:
        raise FileNotFoundError(
            f'"{dataset}" has not been preprocessed: '
            f'{", ".join(str(output) for output in missing)} '
            f'{"are" if len(missing) > 1 else "is"} missing. '
            f'Run `uv run python -m pipelines.preprocess -c {dataset_config(dataset)}` first.'
        )
