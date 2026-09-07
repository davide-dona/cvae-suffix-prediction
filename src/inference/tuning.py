"""Validation-only sampler selection and portable tuning reports."""

import json
import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Self

from omegaconf import DictConfig, OmegaConf
from pydantic import TypeAdapter


@dataclass(frozen=True)
class TuningPoint:
    sampling: dict[str, float]
    score: float
    continuation_precision: float
    continuation_recall: float
    emsc: float
    conformance_mean: float
    unique_sample_rate: float


@dataclass(frozen=True)
class SearchPass:
    pairs: int
    samples: int
    seed: int


@dataclass(frozen=True)
class TuningReport:
    checkpoint_sha256: str
    search: SearchPass
    chosen: dict[str, float]
    grid: tuple[TuningPoint, ...]
    selection_metric: str = 'energy_score'
    selection_direction: str = 'min'

    @classmethod
    def of(cls, checkpoint_sha256: str, *, search: SearchPass, grid: Sequence[TuningPoint]) -> Self:
        if not grid or any(not math.isfinite(point.score) for point in grid):
            raise ValueError('Tuning requires a nonempty grid of finite energy scores')
        return cls(
            checkpoint_sha256=checkpoint_sha256,
            search=search,
            chosen=min(grid, key=lambda point: point.score).sampling,
            grid=tuple(grid),
        )

    @classmethod
    def read(cls, path: str | Path) -> Self:
        return _ADAPTER.validate_json(Path(path).read_bytes())

    def write(self, path: Path) -> Path:
        path.write_text(json.dumps(asdict(self), indent=2))
        return path

    def sampling_for(self, checkpoint_sha256: str) -> DictConfig:
        if self.checkpoint_sha256 != checkpoint_sha256:
            raise ValueError('Tuning report belongs to a different checkpoint')
        return OmegaConf.create(self.chosen)


_ADAPTER = TypeAdapter(TuningReport)
