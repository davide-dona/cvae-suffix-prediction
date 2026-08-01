import argparse
from pathlib import Path

import pandas as pd

from src.evaluation import EvaluationReport, accuracy_metrics, conformance_metrics
from src.logs.discovery import read_process_model
from src.logs.io import read_log


def run(predictions_file: Path) -> None:
    """
    Score a run's generated suffixes and write the result next to them.

    Nothing is generated here: the predictions file is the input, so a metric can be added and
    every past run rescored without the model being run again. It is also the only input: the
    dataset it was predicted from is `predictions_file`'s parent directory name, per
    `src.inference.predict.predictions_path`, and that dataset's splits and process model live
    at the fixed `data/<dataset>` convention every `config/*.yaml` follows.

    Args:
        predictions_file: The predictions to score, from `python -m pipelines.predict`.
    """
    if not predictions_file.exists():
        raise FileNotFoundError(
            f'no predictions at {predictions_file}. Run `python -m pipelines.predict` first, '
            'or name the right predictions file.'
        )

    dataset = predictions_file.parent.name
    data_dir = Path('data') / dataset

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

    path = report.write(predictions_file.with_suffix('.json'))
    print(
        f'Wrote {path}: activity DLS {report.accuracy.activity_dls_mean:.3f} mean / '
        f'{report.accuracy.activity_dls_best:.3f} best, diversity '
        f'{report.accuracy.sample_diversity:.3f}, replay fitness '
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
    parser.add_argument('-p', '--predictions', type=Path, required=True,
                        help='Path to the predictions file to score, from `pipelines.predict`.')
    args = parser.parse_args()

    run(args.predictions)


if __name__ == '__main__':
    main()
