from dataclasses import dataclass
from pathlib import Path

from src.paths.artifact import Artifact
from src.paths.locations import FIGURES_DIR, PRETRAINED_DIR, TABLES_DIR


@dataclass(frozen=True)
class PublishedArtifact(Artifact):
    """A checkpoint as the Hugging Face repo lays it out on disk, one per model per log.

    Not keyed by a run: a run's tag is what tells two attempts apart, and which attempt became the
    published one is a decision already taken by the time a file lands here.
    """

    directory: Path  # where `scripts/fetch.py` puts the repo
    suffix: str  # what a checkpoint is written with

    def path(self, dataset: str, model: str) -> Path:
        """Where one published checkpoint goes, whether or not anything is there.

        Args:
            dataset: The log the model was trained on.
            model: The model's name, as `model.name` in its config, e.g. `cvae`.
        Returns:
            The path that model's published checkpoint is fetched to.
        """
        return self.directory / dataset / f'{model}{self.suffix}'

    def require(self, dataset: str, model: str) -> Path:
        """That path, or a `FileNotFoundError` saying to fetch the published models."""
        return self._found(self.path(dataset=dataset, model=model), self.remedy)


@dataclass(frozen=True)
class NamedArtifact(Artifact):
    """A file named after what it holds rather than after a run, i.e. what `pipelines.visualize`
    writes: one figure covers every log and every model at once, so nothing but its name tells two
    of them apart.

    Nothing reads these back, so they are located and prepared but never required.
    """

    directory: Path  # which of the output directories holds this kind
    suffix: str  # what it is written with, e.g. `.pdf`

    def path(self, name: str) -> Path:
        """Where the file of this name goes, whether or not anything is there."""
        return self.directory / f'{name}{self.suffix}'

    def prepare(self, name: str) -> Path:
        """That path, with the directory it goes in created."""
        return self._made(self.path(name))


# What is published, and what the paper is drawn into.
PRETRAINED = PublishedArtifact(
    kind='published checkpoint',
    remedy='Run `uv run python -m scripts.fetch` first.',
    directory=PRETRAINED_DIR,
    suffix='.pt',
)
FIGURE = NamedArtifact(
    kind='figure',
    remedy='Run `uv run python -m pipelines.visualize` first.',
    directory=FIGURES_DIR,
    suffix='.pdf',
)
TABLE = NamedArtifact(
    kind='table',
    remedy='Run `uv run python -m pipelines.visualize` first.',
    directory=TABLES_DIR,
    suffix='.tex',
)
