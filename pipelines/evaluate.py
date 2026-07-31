import argparse
from pathlib import Path

import pandas as pd

from pipelines.preprocess import ensure_dataset
from src.configs import ExperimentConfig, load_config
from src.evaluation import EvaluationReport, accuracy_metrics, conformance_metrics, evaluation_path
from src.inference import predictions_path
from src.logs.discovery import read_process_model
from src.logs.io import read_log
from src.model import latest_best_model_path


def run(config: ExperimentConfig, predictions_file: Path | None = None) -> None:
    """
    Score a run's generated suffixes and write the result next to them.

    Nothing is generated here: the predictions file is the input, so a metric can be added and
    every past run rescored without the model being run again.

    Args:
        config: The validated experiment config.
        predictions_file: The predictions to score. Defaults to those of the most recent run of
            this config, which is what a config alone can identify.
    """
    # Ensure the dataset is present, so the splits the metrics read can be opened
    ensure_dataset(config.data)

    dataset = config.data.dir.name
    if predictions_file is None:
        # Predictions are named after the checkpoint that produced them, so the newest run
        # resolves through the checkpoints rather than through a second convention.
        model_path = latest_best_model_path(
            config.training.best_model_dir, f'{dataset}/{config.experiment_name}'
        )
        predictions_file = predictions_path(
            config.inference.predictions_dir, f'{dataset}/{model_path.stem}'
        )
    if not predictions_file.exists():
        raise FileNotFoundError(
            f'no predictions at {predictions_file}. Run `python -m pipelines.predict` for this '
            'config first, or name a predictions file explicitly.'
        )

    predictions = pd.read_parquet(path=predictions_file)
    # A truncated pair's ground-truth suffix stops short of the case's real ending, so it is
    # neither a fair target to score against nor a trace that can reach the net's final marking.
    # Dropped once, here, so every number below is measured on the same set of prefixes.
    scored = predictions[~predictions['truncated']]
    truncated_pairs = _pair_count(predictions) - _pair_count(scored)

    run_name = f'{dataset}/{predictions_file.stem}'
    print(
        f'Scoring {len(scored)} predicted suffixes from {predictions_file}'
        + (f', {truncated_pairs} truncated prefixes left out' if truncated_pairs else ''),
        flush=True,
    )

    net, initial_marking, final_marking = read_process_model(config.data.dir / 'model')
    processed_dir = config.data.dir / 'processed'

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

    path = report.write(evaluation_path(config.evaluation.metrics_dir, run_name))
    print(
        f'Wrote {path}: activity DLS {report.accuracy.activity_dls_mean:.3f} mean / '
        f'{report.accuracy.activity_dls_best:.3f} best, replay fitness '
        f'{report.conformance.generated.fitness_mean:.3f} against '
        f'{report.conformance.reference.fitness_mean:.3f} for the ground truth'
    )


def _pair_count(predictions: pd.DataFrame) -> int:
    """How many distinct (case, cut point) prefixes a set of prediction rows covers."""
    return len(predictions.drop_duplicates(subset=['case_id', 'prefix_len']))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Score a run's generated test-split suffixes against the log and the "
                    'process model mined from it.'
    )
    parser.add_argument('-c', '--config', type=Path, required=True,
                        help="Path to this experiment's config YAML.")
    parser.add_argument('-p', '--predictions', type=Path,
                        help='Path to the predictions file to score. Defaults to those of '
                             "this config's most recent run.")
    args = parser.parse_args()

    run(load_config(args.config), args.predictions)


if __name__ == '__main__':
    main()
