from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Self

import pyarrow.parquet as pq
from Declare4Py.ProcessModels.DeclareModel import DeclareModel
from tqdm import tqdm

from src.evaluation.accuracy import AccuracyScores, score_generation
from src.evaluation.conformance import ConformanceScores, score_conformance
from src.inference import read_generation_group
from src.logs.declare import load_declare_model


@dataclass(frozen=True)
class PrefixScores:
    """One prefix's scores, or their mean over a set of prefixes.

    Accuracy asks how close a generated suffix is to the one that actually happened;
    conformance asks whether it is a trace the process allows at all.
    """

    accuracy: AccuracyScores
    conformance: ConformanceScores

    @classmethod
    def mean(cls, values: Sequence[Self]) -> Self:
        """Average a set of prefix scores, each family averaged by its own rules.

        Args:
            values: The scores to average, one per prefix.
        Returns:
            The mean accuracy beside the mean conformance.
        """
        return cls(
            accuracy=AccuracyScores.mean([value.accuracy for value in values]),
            conformance=ConformanceScores.mean([value.conformance for value in values]),
        )


@dataclass(frozen=True)
class ScoredPrefix:
    """One prefix's identity beside its scores, the unit a worker hands back."""

    case_id: str
    prefix_len: int
    samples: int
    scores: PrefixScores


@dataclass(frozen=True)
class ByPrefixLengthMetrics:
    """The scores of the prefixes of one length, and how many pairs that length had."""

    length: int
    pairs_count: int
    scores: PrefixScores


@dataclass(frozen=True)
class EvaluationMetrics:
    """The scores of all prefixes, their breakdown by length, and the population
    they were taken over."""

    pairs: int
    cases: int
    samples_per_prefix: int

    scores: PrefixScores
    # In increasing order of prefix length
    by_prefix_length: list[ByPrefixLengthMetrics]


@dataclass(frozen=True)
class _Worker:
    """What one pool process holds for the whole of its life, rather than per row group."""

    parquet: pq.ParquetFile
    declare_model: DeclareModel
    consider_vacuity: bool


# Set by `_init_worker` in each pool process, and read by `_score_row_group` there. Left
# unassigned so that scoring outside a pool fails loudly rather than on a silent `None`.
_worker: _Worker


def _init_worker(generations_file: Path, dataset: str, consider_vacuity: bool) -> None:
    """Open the file and parse the declarative model once for this process.

    macOS spawns its workers, so neither an open file nor a parsed model can be inherited from the
    parent; doing this per row group instead would repeat it once per task.

    Args:
        generations_file: The generations every task of this process reads from.
        dataset: The dataset whose declarative model conformance is checked against.
        consider_vacuity: Whether a constraint a trace never activates counts as satisfied.
    """
    global _worker
    _worker = _Worker(
        parquet=pq.ParquetFile(generations_file),
        declare_model=load_declare_model(dataset),
        consider_vacuity=consider_vacuity,
    )


def _score_row_group(row_group: int) -> list[ScoredPrefix]:
    """Score every prefix of one row group, in a pool process.

    Args:
        row_group: Which group of the file `_init_worker` opened to score.
    Returns:
        One entry per prefix of the group, in the order it was written. Only the scores travel back
        to the parent, the generations themselves being dropped here.
    """
    return [
        ScoredPrefix(
            case_id=generation.case_id,
            prefix_len=generation.prefix_len,
            samples=len(generation.samples),
            scores=PrefixScores(
                accuracy=score_generation(generation),
                conformance=score_conformance(
                    generation,
                    model=_worker.declare_model,
                    consider_vacuity=_worker.consider_vacuity,
                ),
            ),
        )
        for generation in read_generation_group(parquet=_worker.parquet, row_group=row_group)
    ]


def evaluate_generations(
    generations_file: Path,
    *,
    dataset: str,
    consider_vacuity: bool,
    workers: int | None = None,
) -> EvaluationMetrics:
    """Score generated suffixes against the ground truth they were generated for, and against
    the declarative model the dataset was mined for.

    Prefixes are scored across a pool of processes, a row group at a time: a prefix's score depends
    on nothing but the prefix, and both the edit distances and the conformance checks are pure
    Python that holds the GIL.

    Args:
        generations_file: The generations to score, from `python -m pipelines.generate`. Passed as
            a path rather than as read prefixes, since each worker opens the file itself and only
            the scores it computes cross back.
        dataset: The dataset the prefixes were cut from, naming the declarative model to check
            conformance against.
        consider_vacuity: Whether a constraint a trace never activates counts as satisfied.
        workers: How many processes to score with, or `None` for one per available CPU.
    Returns:
        The averages over every prefix, the same averages broken down by cut point, and the
        population they were taken over. Every prefix weighs the same however many samples were
        drawn for it, so a prefix is the unit this describes and a sample is not.
    """
    # Each prefix is scored down to a handful of floats and its generation dropped in the worker,
    # so a split of hundreds of thousands of them never brings more than a row group's objects back
    # here. Bucketing by cut point as the scores arrive is what saves a second pass over them.
    buckets: dict[int, list[PrefixScores]] = {}
    cases: set[str] = set()
    samples_per_prefix = 0

    with pq.ParquetFile(generations_file) as parquet:
        row_groups, prefixes = parquet.num_row_groups, parquet.metadata.num_rows

    with (
        ProcessPoolExecutor(
            max_workers=workers,
            initializer=_init_worker,
            initargs=(generations_file, dataset, consider_vacuity),
        ) as executor,
        tqdm(total=prefixes, desc='Scoring', unit='prefix') as progress,
    ):
        # `map` yields the groups in the order they were submitted however they finished, so the
        # buckets fill in the order the file holds, and the means over them stay reproducible.
        for group in executor.map(_score_row_group, range(row_groups)):
            progress.update(len(group))
            for prefix in group:
                cases.add(prefix.case_id)
                # Every prefix is drawn for the same number of times, so the first answers for all.
                samples_per_prefix = samples_per_prefix or prefix.samples
                buckets.setdefault(prefix.prefix_len, []).append(prefix.scores)

    scored = [scores for bucket in buckets.values() for scores in bucket]
    return EvaluationMetrics(
        pairs=len(scored),
        cases=len(cases),
        samples_per_prefix=samples_per_prefix,
        scores=PrefixScores.mean(scored),
        by_prefix_length=[
            ByPrefixLengthMetrics(
                length=length,
                pairs_count=len(buckets[length]),
                scores=PrefixScores.mean(buckets[length]),
            )
            for length in sorted(buckets)
        ],
    )
