from __future__ import annotations
from omegaconf import DictConfig, OmegaConf

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Self

from pydantic import TypeAdapter

from src.identity import RunIdentity

# The pair searched over. A temperature scales every step alike; a nucleus reads how peaked each
# one is. Below 1.0 the temperature sharpens towards the mode, and those values are deliberately
# in the grid: the objective below is what has to rule them out, and a grid that cannot reach the
# collapsed corner cannot show that it was avoided.
TEMPERATURES = (0.8, 0.9, 1.0, 1.1, 1.25)
TOP_PS = (0.85, 0.9, 0.95, 0.98, 1.0)


def objective(precision: float, recall: float) -> float:
    """What a sampler is chosen on: the F1 of the two continuation scores.

    Not `emsc`, though that is what checkpoints are selected on. EMSC does not charge for
    collapse - a model answering every prefix with one central suffix scores well on it - so a
    grid holding temperatures below 1.0 would be free to sharpen its way to a deterministic
    decoder, which is the one thing the arm that draws per step exists not to be. Recall charges
    for that directly, precision charges for spraying the tail to buy it back, and `CLAUDE.md`
    already names the pair as what says whether the draws cover what the log continued with.

    Args:
        precision: `continuation_precision` of the pass.
        recall: `continuation_recall` of the pass.
    Returns:
        Their harmonic mean, or 0.0 where both are 0.0.
    """
    total = precision + recall
    return 2.0 * precision * recall / total if total > 0.0 else 0.0


@dataclass(frozen=True)
class TuningPoint:
    """One cell of the grid: the sampler tried, and what the validation pass scored it.

    `score` is the objective; the four below it are recorded but not optimized, so the operating
    point that was picked can be read against the ones that were not without the search being run
    again. `conformance_mean` is the reason `top_p` is in the grid at all, and
    `unique_sample_rate` is what a collapsing corner shows up in first.
    """

    sampling: DictConfig
    score: float
    continuation_precision: float
    continuation_recall: float
    emsc: float
    conformance_mean: float
    unique_sample_rate: float


@dataclass(frozen=True)
class SearchPass:
    """What the search was run over, so a report says how much to trust its own resolution."""

    pairs: int  # prefixes of the validation split drawn for, the same ones at every point
    samples: int  # suffixes drawn per prefix
    seed: int  # reset before every point, which is what makes the comparison paired


@dataclass(frozen=True)
class TuningReport:
    """The sampler chosen for one trained run, and the grid it was chosen from.

    Its own artifact rather than a field written back into the checkpoint: the weights are the
    record of what was trained and this is the record of what was picked to read them with, so a
    run can be searched again without the checkpoint being touched.
    """

    run: RunIdentity
    search: SearchPass
    chosen: DictConfig
    grid: tuple[TuningPoint, ...]

    @classmethod
    def of(cls, run: RunIdentity, *, search: SearchPass, grid: Sequence[TuningPoint]) -> Self:
        """Build a report around the best point of a finished grid.

        Args:
            run: The run whose checkpoint was searched.
            search: What the search was run over.
            grid: Every point tried, in the order they were tried.
        Returns:
            The report, its `chosen` the sampler of the highest-scoring point. Ties go to the one
            tried first, which is the lowest temperature and the tightest nucleus: of two samplers
            the validation split cannot tell apart, the more constrained is the one to report.
        Raises:
            ValueError: If the grid is empty, which is a search that never ran.
        """
        if not grid:
            raise ValueError('a tuning report needs at least one grid point')
        return cls(
            run=run,
            search=search,
            chosen=max(grid, key=lambda point: point.score).sampling,
            grid=tuple(grid),
        )

    @classmethod
    def read(cls, path: str | Path) -> Self:
        """Read a report back from the JSON `write` produced.

        Args:
            path: The report to read, e.g. from `paths.TUNING`.
        Returns:
            The report, validated against the current schema.
        """
        return _ADAPTER.validate_json(Path(path).read_bytes())

    def write(self, path: str | Path) -> Path:
        """Write the report as JSON.

        Args:
            path: Where to write, its directory already made, from `paths.TUNING.prepare`.
        Returns:
            The path written to.
        """
        path = Path(path)
        path.write_bytes(_ADAPTER.dump_json(self, indent=4))
        return path

    def sampling_for(self, run: RunIdentity) -> DictConfig:
        """The chosen sampler, having checked this report is that run's.

        A report is handed to `pipelines.generate` as a path, and a path says nothing about which
        checkpoint it was searched against. Reading the identity out of the file instead is what
        stops a run being generated under another run's operating point.

        Args:
            run: The run about to generate, from its checkpoint.
        Returns:
            The sampler this report chose.
        Raises:
            ValueError: If the report was written for a different run.
        """
        if self.run != run:
            raise ValueError(
                f'this tuning report was written for {self.run}, not {run}. Tune {run} first, or '
                'name its own report.'
            )
        return self.chosen


# Built once, since a `TypeAdapter` compiles the schema it validates against.
_ADAPTER = TypeAdapter(TuningReport)
