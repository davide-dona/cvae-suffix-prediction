import argparse
from pathlib import Path

import pandas as pd

from src.evaluation import EvaluationReport, accuracy_metrics


def run(generations_file: Path) -> None:
    """Score a run's generated suffixes and write the result under the `eval` sibling of the
    generations directory.

    Args:
        generations_file: The generations to score, from `python -m pipelines.generate`.
    """
    if not generations_file.exists():
        raise FileNotFoundError(
            f'no generations at {generations_file}. Run `python -m pipelines.generate` first, '
            'or name the right generations file.'
        )

    # Retrieve the dataset name from the generations file's path
    dataset = generations_file.parent.name

    # Read the generations file into a DataFrame
    generations = pd.read_parquet(path=generations_file)
    
    # Filter out truncated generations, which are not scored
    scored = generations[~generations['truncated']]
    truncated_pairs = _pair_count(generations) - _pair_count(scored)

    run_name = f'{dataset}/{generations_file.stem}'
    print(
        f'Scoring {len(scored)} generated suffixes from {generations_file}'
        + (f', {truncated_pairs} truncated prefixes left out' if truncated_pairs else ''),
        flush=True,
    )

    report = EvaluationReport(
        run_name=run_name,
        pairs=_pair_count(scored),
        cases=int(scored['case_id'].nunique()),
        samples_per_prefix=int(scored['sample_index'].nunique()),
        truncated_pairs_excluded=truncated_pairs,
        accuracy=accuracy_metrics(scored)
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
    parser.add_argument('-g', '--generations', type=Path, required=True,
                        help='Path to the generations file to score, from `pipelines.generate`.')
    args = parser.parse_args()

    run(args.generations)


if __name__ == '__main__':
    main()
