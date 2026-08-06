import argparse
from pathlib import Path

import pandas as pd

from pipelines.preprocess import require_dataset
from src.configs import ExperimentConfig, load_config
from src.evaluation.metrics import evaluate_generations
from src.evaluation.report import EvaluationReport
from src.logs.declare import load_declare_model


def run(config: ExperimentConfig, generations_file: Path) -> None:
    """Score a run's generated suffixes and write the result under the `eval` sibling of the
    generations directory.

    Args:
        config: The validated experiment config of the run that wrote the generations, read for
            the dataset the suffixes belong to and the declarative model they are checked against.
        generations_file: The generations to score, from `python -m pipelines.generate`.
    """

    require_dataset(config.data)
    if not generations_file.exists():
        raise FileNotFoundError(
            f'no generations at {generations_file}. Run `python -m pipelines.generate` first, '
            'or name the right generations file.'
        )

    dataset = config.data.dir.name

    # Read the generations file into a DataFrame
    generations = pd.read_parquet(path=generations_file)
    if 'prefix_activities' not in generations.columns:
        raise ValueError(
            f'{generations_file} does not carry the prefix each suffix continues, which '
            'conformance is checked against. Re-run `python -m pipelines.generate` for this run.'
        )

    # Filter out truncated generations, which are not scored
    scored = generations[~generations['truncated']]
    truncated_pairs = _pair_count(generations) - _pair_count(scored)

    run_name = f'{dataset}/{generations_file.stem}'
    print(
        f'Scoring {len(scored)} generated suffixes from {generations_file}'
        + (f', {truncated_pairs} truncated prefixes left out' if truncated_pairs else '')
        + '. Checking each one against the declarative model is the slow part',
        flush=True,
    )

    metrics = evaluate_generations(
        scored,
        declare_model=load_declare_model(config.data),
        consider_vacuity=config.declare.consider_vacuity,
    )
    report = EvaluationReport(
        run_name=run_name,
        # Read off the breakdown rather than the frame, so the count is the population the
        # averages were actually taken over.
        pairs=sum(bucket.pairs_count for bucket in metrics.by_prefix_length),
        cases=int(scored['case_id'].nunique()),
        samples_per_prefix=int(scored['sample_index'].nunique()),
        truncated_pairs_excluded=truncated_pairs,
        metrics=metrics,
    )

    path = report.write(_eval_path(generations_file))
    print(f"Wrote evaluation report to {path}")


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
        description="Score a run's generated test-split suffixes against the ground truth."
    )
    parser.add_argument('-c', '--config', type=Path, required=True,
                        help="Path to the experiment config the generations were written under.")
    parser.add_argument('-g', '--generations', type=Path, required=True,
                        help='Path to the generations file to score, from `pipelines.generate`.')
    args = parser.parse_args()

    run(load_config(args.config), args.generations)


if __name__ == '__main__':
    main()
