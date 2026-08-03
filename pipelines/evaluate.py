import argparse
from pathlib import Path

import pandas as pd

from src.evaluation import EvaluationReport, accuracy_metrics, conformance_metrics
from src.logs.discovery import read_process_model
from src.logs.io import read_log


def run(generations_file: Path) -> None:
    """Score a run's generated suffixes and write the result under the `eval` sibling of the
    generations directory.

    Nothing is generated here: the generations file is the input, so a metric can be added and
    every past run rescored without the model being run again. It is also the only input: the
    dataset it was generated from is `generations_file`'s parent directory name, per
    `src.inference.generate.generations_path`, and that dataset's splits and process model live
    at the fixed `data/<dataset>` convention every `config/*.yaml` follows.

    Args:
        generations_file: The generations to score, from `python -m pipelines.generate`.
    """
    if not generations_file.exists():
        raise FileNotFoundError(
            f'no generations at {generations_file}. Run `python -m pipelines.generate` first, '
            'or name the right generations file.'
        )

    dataset = generations_file.parent.name
    data_dir = Path('data') / dataset

    generations = pd.read_parquet(path=generations_file)
    # A truncated pair's ground-truth suffix stops short of the case's real ending, so it is
    # neither a fair target to score against nor a trace that can reach the net's final marking.
    # Dropped once, here, so every number below is measured on the same set of prefixes.
    scored = generations[~generations['truncated']]
    truncated_pairs = _pair_count(generations) - _pair_count(scored)

    run_name = f'{dataset}/{generations_file.stem}'
    print(
        f'Scoring {len(scored)} generated suffixes from {generations_file}'
        + (f', {truncated_pairs} truncated prefixes left out' if truncated_pairs else ''),
        flush=True,
    )

    net, initial_marking, final_marking = read_process_model(data_dir / 'model')
    processed_dir = data_dir / 'processed'

    report = EvaluationReport(
        run_name=run_name,
        pairs=_pair_count(scored),
        cases=int(scored['case_id'].nunique()),
        samples_per_prefix=int(scored['sample_index'].nunique()),
        truncated_pairs_excluded=truncated_pairs,
        accuracy=accuracy_metrics(scored),
        conformance=conformance_metrics(
            scored,
            test_log=read_log(processed_dir / 'test.csv'),
            train_log=read_log(processed_dir / 'train.csv'),
            net=net,
            initial_marking=initial_marking,
            final_marking=final_marking,
        ),
    )

    path = report.write(_eval_path(generations_file))
    print(
        f'Wrote {path}: activity DLS {report.accuracy.activity_dls_mean:.3f} mean / '
        f'{report.accuracy.activity_dls_best:.3f} best, diversity '
        f'{report.accuracy.sample_diversity:.3f}, replay fitness '
        f'{report.conformance.generated.fitness_mean:.3f} against '
        f'{report.conformance.reference.fitness_mean:.3f} for the ground truth, precision '
        f'{report.conformance.generated.precision:.3f} against '
        f'{report.conformance.reference.precision:.3f}'
    )


def _pair_count(generations: pd.DataFrame) -> int:
    """How many distinct (case, cut point) prefixes a set of generation rows covers."""
    return len(generations.drop_duplicates(subset=['case_id', 'prefix_len']))


def _eval_path(generations_file: Path) -> Path:
    """Where a run's evaluation report goes: the `eval` sibling of the generations directory,
    keeping the same `<dataset>/<run_name>.json` layout `generations_path` uses for `.parquet`.

    Args:
        generations_file: The generations file the report was scored from.
    Returns:
        The path to write the report to.
    """
    generations_dir = generations_file.parents[1]
    eval_dir = generations_dir.parent / 'eval'
    return eval_dir / generations_file.parent.name / f'{generations_file.stem}.json'


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Score a run's generated test-split suffixes against the log and the "
                    'process model mined from it.'
    )
    parser.add_argument('-g', '--generations', type=Path, required=True,
                        help='Path to the generations file to score, from `pipelines.generate`.')
    args = parser.parse_args()

    run(args.generations)


if __name__ == '__main__':
    main()
