import argparse
from pathlib import Path

from pipelines.preprocess import require_dataset
from src import paths
from src.configs import ExperimentConfig, load_config
from src.evaluation.metrics import evaluate_generations
from src.evaluation.report import EvaluationReport


def run(config: ExperimentConfig, generations_file: Path, workers: int | None) -> None:
    """Score a run's generated suffixes and write the result under `outputs/eval/`.

    Args:
        config: The validated experiment config of the run that wrote the generations, read for
            the dataset the suffixes belong to and the declarative model they are checked against.
        generations_file: The generations to score, from `python -m pipelines.generate`.
        workers: How many processes to score with, or `None` for one per available CPU.
    """

    dataset = config.data.name
    require_dataset(dataset)
    if not generations_file.exists():
        raise FileNotFoundError(
            f'no generations at {generations_file}. Run `python -m pipelines.generate` first, '
            'or name the right generations file.'
        )

    print(f'Scoring the suffixes generated for each prefix of {generations_file}...', flush=True)

    # Compute the metrics of the generation
    metrics = evaluate_generations(
        generations_file,
        dataset=dataset,
        consider_vacuity=config.declare.consider_vacuity,
        workers=workers,
    )

    # The report is named after the run the generations belong to, so it sits beside them under
    # the same `<dataset>/<run>` name.
    run_name = f'{dataset}/{generations_file.stem}'
    report = EvaluationReport(run_name=run_name, metrics=metrics)
    path = report.write(paths.evaluation_path(run_name))
    print(
        f'Scored {metrics.pairs} prefixes over {metrics.cases} cases. '
        f'Wrote evaluation report to {path}'
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Score a run's generated test-split suffixes against the ground truth."
    )
    parser.add_argument(
        '-c',
        '--config',
        type=Path,
        required=True,
        help='Path to the experiment config the generations were written under.',
    )
    parser.add_argument(
        '-g',
        '--generations',
        type=Path,
        required=True,
        help='Path to the generations file to score, from `pipelines.generate`.',
    )
    parser.add_argument(
        '-j',
        '--workers',
        type=int,
        default=None,
        help='How many processes to score with. Defaults to one per available CPU.',
    )
    args = parser.parse_args()

    run(load_config(args.config), args.generations, args.workers)


if __name__ == '__main__':
    main()
