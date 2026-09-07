from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from src.identity import RunIdentity
from src.paths.artifact import Artifact
from src.paths.locations import OUTPUTS_DIR


@dataclass(frozen=True)
class RunArtifact(Artifact):
    """A file one run writes, named after that run alone.

    The model sits above the tag, so one log's directory lists the models compared on it before
    any filename is read, and the leaf is the tag alone, since the directory it sits in already
    says which model wrote it.
    """

    directory: Path  # which of the output directories holds this kind
    suffix: str  # what it is written with, e.g. `.parquet`; empty where it is a directory

    def path(self, run: RunIdentity) -> Path:
        """Where this run's file of this kind goes, whether or not anything is there."""
        return self.directory / run.dataset / run.model / f'{run.tag}{self.suffix}'

    def require(self, run: RunIdentity) -> Path:
        """That path, or a `FileNotFoundError` naming the pipeline that writes it."""
        return self._found(self.path(run), self.remedy)

    def prepare(self, run: RunIdentity) -> Path:
        """That path, with the directory it goes in created."""
        return self._made(self.path(run))

    def beside(self, path: Path) -> Path:
        """This kind's file beside another of the same run, i.e. that path with this suffix.

        Two kinds sharing a directory are always a pair, so a reader that has just been handed one
        of them finds the other without being told where to look.

        Args:
            path: The other file, wherever it has been moved to. Its own directory is used rather
                than this artifact's, so a pair that has been moved together is still a pair.
        Returns:
            The path this kind's file sits at beside it.
        """
        return path.with_suffix(self.suffix)

    def sweep(self, directories: Sequence[Path]) -> list[Path]:
        """Find every file of this kind under one or more directories, e.g. `outputs/eval` and
        `pinned/eval` swept together to compare in-progress runs against pinned ones.

        Args:
            directories: The directories to sweep, at any depth.
        Returns:
            The files under them, in path order.
        Raises:
            FileNotFoundError: If they hold none at all.
        """
        found = sorted(
            path for directory in directories for path in directory.rglob(f'*{self.suffix}')
        )
        if not found:
            under = ', '.join(str(directory) for directory in directories)
            raise FileNotFoundError(
                f'no {self.kind}s under {under}. {self.remedy} Or sweep a directory holding some.'
            )
        return found


# What a run writes. The best step's weights, not the last: a run is never carried on from, so
# there is nothing a checkpoint of the final step would be read for. `scripts/publish.py` promotes
# one of these to the curated name `PRETRAINED` describes.
BEST_CHECKPOINT = RunArtifact(
    kind='best checkpoint',
    remedy='Run `uv run python -m pipelines.train` first.',
    directory=OUTPUTS_DIR / 'checkpoints' / 'best',
    suffix='.pt',
)
GENERATIONS = RunArtifact(
    kind='generations file',
    remedy='Run `uv run python -m pipelines.generate` first.',
    directory=OUTPUTS_DIR / 'generations',
    suffix='.parquet',
)
# The per-prefix scores are the report's own path with a different suffix, so a reader that has
# just read a report finds the scores it was averaged from beside it.
EVALUATION = RunArtifact(
    kind='evaluation report',
    remedy='Run `uv run python -m pipelines.evaluate` first.',
    directory=OUTPUTS_DIR / 'eval',
    suffix='.json',
)
PREFIX_SCORES = RunArtifact(
    kind='per-prefix scores file',
    remedy='Run `uv run python -m pipelines.evaluate` first.',
    directory=OUTPUTS_DIR / 'eval',
    suffix='.parquet',
)
# What sampler a trained run is read with, chosen on the validation split after training rather
# than written into the weights. Its own kind rather than a field of the checkpoint, so a
# checkpoint stays the record of what was trained and this stays the record of what was picked for
# it; one run can be searched again without the weights being touched.
TUNING = RunArtifact(
    kind='tuning report',
    remedy='Run `uv run python -m pipelines.tune` first.',
    directory=OUTPUTS_DIR / 'tuning',
    suffix='.json',
)
