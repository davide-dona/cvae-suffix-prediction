import os
import time
from collections.abc import Iterator
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import hydra
from omegaconf import DictConfig
from tqdm import tqdm

from src import paths
from src.artifacts import sha256
from src.cli import banner, duration, step
from src.evaluation import EvaluationReport, EvaluationSummary, PrefixSummary, stream_prefix_scores
from src.evaluation.scores import MIN_REFERENCE_OCCURRENCES
from src.inference.generation_store import Generations
from src.logs import ContinuationIndex, Split
from src.logs.declare import ConformanceChecker, discovery_settings
from src.runtime import output_path, start_stage
from src.suffixes import ActivityCodes


@dataclass(frozen=True)
class _Worker:
    """What one pool process holds for the whole of its life, rather than per block."""

    generations: Generations
    checker: ConformanceChecker
    index: ContinuationIndex


# Set by `_init_worker` in each pool process, and read by `_score_block` there. Left
# unassigned so that scoring outside a pool fails loudly rather than on a silent `None`.
_worker: _Worker


def _init_worker(generations_file: Path, dataset: str) -> None:
    """Open the file, prepare the declarative model and read the continuation index once for this
    process.

    Args:
        generations_file: The generations every task of this process reads from.
        dataset: The dataset whose declarative model conformance is checked against, and whose
            observed continuations the generated ones are compared with.
    """
    global _worker
    generations = Generations(generations_file)
    index = ContinuationIndex.read(dataset=dataset, split=Split.TEST)
    vocabulary = generations.vocabulary
    # Check that the the vocabulary the generations were written under is the same
    # as the one of the dataset's continuation
    if vocabulary != index.vocabulary:
        raise ValueError(
            f'{generations_file} spells its activities on a different scale than the continuation '
            f'index of {dataset}: the two were built from different preprocessings. Rerun '
            'pipelines.preprocess, then pipelines.generate.'
        )

    _worker = _Worker(
        generations=generations,
        checker=ConformanceChecker(dataset, ActivityCodes.of(vocabulary)),
        index=index,
    )


def _score_block(block: int) -> list[PrefixSummary]:
    """Score every prefix of one block of the generations file, in the order they were written.

    Args:
        block: Which block of the file `_init_worker` opened to score.
    Returns:
        One entry per prefix of the block, in the order it was written.
    """
    return [
        PrefixSummary.of(generation, checker=_worker.checker, index=_worker.index)
        for generation in _worker.generations.block(block)
    ]


def _score_in_parallel(
    generations_file: Path,
    *,
    dataset: str,
    blocks: int,
    prefixes: int,
    workers: int | None,
) -> Iterator[PrefixSummary]:
    """Score a generations file across a pool of processes, a block at a time.

    A prefix's score depends on nothing but the prefix, and the conformance checks that dominate
    it are pure Python that holds the GIL. Each prefix is scored down to a handful of floats and
    its generation dropped in the worker, so a split of hundreds of thousands of them never brings
    more than a block's objects back here.

    Args:
        generations_file: The generations to score, from `python -m pipelines.generate`. Passed as
            a path rather than as read prefixes, since each worker opens the file itself and only
            the scores it computes cross back.
        dataset: The dataset the prefixes were cut from, naming the declarative model to check
            conformance against.
        blocks: How many blocks the file holds, one unit of work each.
        prefixes: How many prefixes it holds in total, for the progress bar.
        workers: How many processes to score with, or `None` for one per available CPU.
    Yields:
        Each prefix's scores, in the order the file holds them. The pool is shut down once they
        have all been drawn.
    """
    with (
        ProcessPoolExecutor(
            max_workers=workers,
            initializer=_init_worker,
            initargs=(generations_file, dataset),
        ) as executor,
        tqdm(total=prefixes, desc='Scoring', unit='prefix') as progress,
    ):
        # `map` yields the blocks in the order they were submitted however they finished, so the
        # scores arrive in the order the file holds, and the means over them stay reproducible.
        for block in executor.map(_score_block, range(blocks)):
            progress.update(len(block))
            yield from block


def run(generations_file: Path, workers: int | None) -> None:
    """Score a run's generated suffixes and write the result under `outputs/eval/`.

    Args:
        generations_file: The generations to score, from `python -m pipelines.generate`. It says
            which run and dataset wrote it, so the report is named after that run and the
            declarative model is looked up under that dataset.
        workers: How many processes to score with, or `None` for one per available CPU.
    """
    with Generations(generations_file) as generations:
        metadata = generations.metadata
        blocks, prefixes = generations.blocks, generations.prefixes
        # Which prefix each row answers, in the order the file holds them, which is the order the
        # pool scores them in. Two columns, so this is cheap even on a quarter of a million rows.
        keys = generations.prefix_keys()

    metadata = metadata | {'source_sha256': sha256(generations_file)}
    if workers is not None and (not isinstance(workers, int) or workers < 1):
        raise ValueError('workers must be a positive integer')
    if prefixes == 0:
        raise ValueError('Cannot evaluate an empty generations file')

    # Check that the dataset was preprocessed and that the test-split continuations were written
    dataset = metadata['dataset']
    paths.require_preprocessed(dataset)
    paths.CONTINUATIONS.require(dataset=dataset, split=Split.TEST)

    # What the pool will actually start, which is what the wait before the first block is spent on.
    processes = workers if workers is not None else os.cpu_count()

    # What the model being checked against was mined under, so a report is never read without
    # knowing which constraints it holds.
    model_path = paths.DECLARE_MODEL.require(dataset)
    mined = discovery_settings(model_path)
    mined_under = (
        f'min support {mined.min_support:.0%}, consider_vacuity={mined.consider_vacuity}'
        if mined is not None
        else 'settings not recorded, so this model predates the header'
    )

    banner(
        'Scoring generated suffixes',
        {
            'source': metadata,
            'dataset': dataset,
            'generations': f'{generations_file} ({prefixes:,} prefixes)',
            'declarative model': f'{model_path} (mined at {mined_under})',
            'continuations': paths.CONTINUATIONS.path(dataset=dataset, split=Split.TEST),
            'workers': f'{processes} processes, one block of ~{prefixes // max(blocks, 1):,} '
            'prefixes each',
            'report': output_path('evaluation.json'),
            'prefix scores': output_path('prefix_scores.parquet'),
        },
    )

    started = time.perf_counter()

    scores_path = output_path('prefix_scores.parquet')

    # Summarize the generation, folding each prefix's scores in as the pool hands them back and
    # writing them out on the way past. One stream, so the per-prefix file costs a write rather
    # than a second scoring pass.
    with step(
        f'Scoring {prefixes:,} prefixes across {processes} process(es), each loading the '
        'declarative model and the continuation index first'
    ):
        summary = EvaluationSummary.of(
            stream_prefix_scores(
                _score_in_parallel(
                    generations_file,
                    dataset=dataset,
                    blocks=blocks,
                    prefixes=prefixes,
                    workers=workers,
                ),
                keys,
                path=scores_path,
                metadata=metadata,
            )
        )

    # The report is named after the run the generations carry, so it sits under `outputs/eval/`
    # exactly where they sit under `outputs/generations/`.
    report = EvaluationReport(metadata=metadata, summary=summary)
    path = report.write(output_path('evaluation.json'))
    # The distributional scores are read over the prefixes the log ran often enough, so how many
    # of them there were is part of what the report says rather than something to go looking for.
    share = summary.compared / summary.prefixes if summary.prefixes else 0.0
    print(
        f'Scored {summary.prefixes:,} prefixes in {duration(time.perf_counter() - started)}, '
        f'{summary.compared:,} of them ({share:.0%}) run at least {MIN_REFERENCE_OCCURRENCES} '
        f'times by the log, which is what the distributional scores are read over. '
        f'Wrote evaluation report to {path} and its per-prefix scores to {scores_path}'
    )


@hydra.main(version_base='1.3', config_path='../config', config_name='evaluate')
def main(cfg: DictConfig) -> None:
    start_stage(cfg)
    run(Path(cfg.generations), cfg.workers)


if __name__ == '__main__':
    main()
