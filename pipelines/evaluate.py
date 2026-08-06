import argparse
from pathlib import Path

from pipelines.preprocess import require_dataset
from src import paths
from src.configs import ExperimentConfig, load_config
from src.evaluation.metrics import evaluate_generations
from src.evaluation.report import EvaluationReport
from src.inference import read_generations
from src.logs.declare import load_declare_model


def run(config: ExperimentConfig, generations_file: Path) -> None:
    """Score a run's generated suffixes and write the result under `outputs/eval/`.

    Args:
        config: The validated experiment config of the run that wrote the generations, read for
            the dataset the suffixes belong to and the declarative model they are checked against.
        generations_file: The generations to score, from `python -m pipelines.generate`.
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
        read_generations(generations_file),
        declare_model=load_declare_model(dataset),
        consider_vacuity=config.declare.consider_vacuity,
    )

    # The report is named after the run the generations belong to, so it sits beside them under
    # the same `<dataset>/<run>` name.
    run_name = f'{dataset}/{generations_file.stem}'
    report = EvaluationReport(run_name=run_name, metrics=metrics)
    path = report.write(paths.evaluation_path(run_name))
    print(
        f'Scored {metrics.pairs} prefixes over {metrics.cases} cases'
        + (
            f', {metrics.truncated_pairs_excluded} truncated prefixes left out'
            if metrics.truncated_pairs_excluded
            else ''
        )
        + f'. Wrote evaluation report to {path}'
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
    args = parser.parse_args()

    run(load_config(args.config), args.generations)


if __name__ == '__main__':
    main()
