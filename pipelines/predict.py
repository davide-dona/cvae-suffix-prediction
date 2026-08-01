import argparse
from pathlib import Path
import torch
from torch.utils.data import DataLoader

from pipelines.preprocess import ensure_dataset
from src.configs import DatasetInfo, ExperimentConfig, load_config
from src.datasets.codec import Codec
from src.datasets.dataset import SuffixDataset
from src.inference import generate_predictions, generation_batch_size, predictions_path
from src.model import TransformerCVAE, latest_best_model_path, load_checkpoint


def run(config: ExperimentConfig, model_path: Path | None = None) -> None:
    """
    Generate suffixes for every prefix of the test split and write them out.
    Args:
        config: The validated experiment config.
        model_path: The checkpoint to generate with. Defaults to the best model of the most
            recent run of this config, which is what a config alone can identify: the runs of
            one config differ only in the timestamp their name ends on.
    """
    # Ensure the dataset is present, so the test split can be read
    ensure_dataset(config.data)
    # Set the RNG seed for reproducibility
    torch.manual_seed(config.seed)

    # Build the dataset info and codec, needed to decode the predictions back into event sequences
    dataset_info = DatasetInfo.build(config.data)
    codec = Codec(dataset_info)
    test_dataset = SuffixDataset(dataset_info.test, dataset_info, codec)

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=generation_batch_size(inference=config.inference, upper_bound=config.data.batch_size),
        shuffle=False,
        num_workers=config.data.num_workers,
    )

    # Build the model from the checkpoint, defaulting to the best model of the most recent run of this config
    if model_path is None:
        model_path = latest_best_model_path(
            config.training.best_model_dir, f'{config.data.dir.name}/{config.experiment_name}'
        )
    model = TransformerCVAE.from_checkpoint(
        load_checkpoint(model_path), dataset_info, device=config.training.device
    )

    # Build the path to write the predictions to, named after the checkpoint's run, and ensure
    # its parent directory exists
    path = predictions_path(config.inference.predictions_dir, f'{config.data.dir.name}/{model_path.stem}')
    path.parent.mkdir(parents=True, exist_ok=True)

    print(
        f'Generating {config.inference.num_samples} suffixes for each of '
        f'{len(test_loader.dataset)} test prefixes, with {model_path}',
        flush=True,
    )

    # Generate the predictions for the test split
    predictions = generate_predictions(
        model,
        test_loader,
        codec,
        num_samples=config.inference.num_samples,
        device=torch.device(config.training.device),
    )
    
    # Write the predictions to disk as a parquet file
    predictions.to_parquet(path=path, index=False)
    print(f'Wrote {len(predictions)} predicted suffixes to {path}')


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Generate test-split suffix predictions from a trained model.'
    )
    parser.add_argument('-c', '--config', type=Path, required=True,
                        help="Path to this experiment's config YAML.")
    parser.add_argument('-m', '--model', type=Path,
                        help='Path to the checkpoint to generate with. Defaults to the best '
                             "model of this config's most recent run.")
    args = parser.parse_args()

    run(load_config(args.config), args.model)


if __name__ == '__main__':
    main()
