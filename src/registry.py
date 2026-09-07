from collections.abc import Iterable
from dataclasses import dataclass


@dataclass(frozen=True)
class Registry[T]:
    """One table of what is declared for a kind of thing, and the order they are drawn in."""

    kind: str  # what one entry is, e.g. `model`, as the error names it
    where: str  # where a missing one goes, e.g. `MODELS in src/visualization/labels/models.py`
    entries: dict[str, T]  # what is declared for each entry, in the order they are drawn

    def __getitem__(self, key: str) -> T:
        """What this table declares for one entry.

        Args:
            key: Its name, as a report or a generations file carries it.
        Returns:
            Its declared entry.
        Raises:
            ValueError: If nothing is declared for it, since anything derived from the key itself
                would read one way here and another in the paper.
        """
        if key not in self.entries:
            raise ValueError(
                f'nothing declared for the {self.kind} {key!r}. Add it to {self.where}. '
                f'The {self.kind}s declared are {", ".join(self.entries)}.'
            )
        return self.entries[key]

    def ordered(self, keys: Iterable[str]) -> list[str]:
        """Sort a set of entries into the order this table declares them in.

        Args:
            keys: The entries present, in any order and with repeats.
        Returns:
            Them alone, in the declared order, so a legend, a row of panels and a table's columns
            all read the same way however the files were named on the command line.
        Raises:
            ValueError: If one of them is not declared.
        """
        present = set(keys)
        # Asked of every one of them, so an undeclared entry stops the run here rather than being
        # quietly left out of what comes back.
        for key in present:
            self[key]
        return [key for key in self.entries if key in present]
