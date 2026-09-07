from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Artifact:
    """One kind of file this project writes, described where it is laid out rather than where it
    is read: what to call it in a message, and what to run when it is not there.

    Subclasses say where one lands, since what names a file differs: a dataset, a split of one, a
    run, a figure. Everything else is shared, so no reader words a remedy of its own and none
    creates a directory before writing.
    """

    kind: str  # what one file is called in a message, e.g. `continuation index`
    # The command to run when one is missing, written out at every artifact rather than shared
    # through a constant, so what a message will say is read where the artifact is declared. The
    # ones a dataset names leave `{config}` for `DatasetArtifact` and `SplitArtifact` to fill in,
    # since which config to rerun with depends on the dataset the missing file belongs to.
    remedy: str

    def _found(self, path: Path, remedy: str) -> Path:
        """That path, having checked something is there.

        Args:
            path: Where a file of this kind goes, from the subclass.
            remedy: What to run to produce it, the subclass having filled in whatever depends on
                which file was asked for.
        Returns:
            That same path.
        Raises:
            FileNotFoundError: If nothing is there, naming what is missing and what writes it.
        """
        if not path.exists():
            raise FileNotFoundError(f'no {self.kind} at {path}. {remedy}')
        return path

    @staticmethod
    def _made(path: Path) -> Path:
        """That path, with the directory it goes in created."""
        path.parent.mkdir(parents=True, exist_ok=True)
        return path
