import argparse
from pathlib import Path
import torch
from torch.utils.data import DataLoader

from pipelines.preprocess import ensure_dataset
from src.configs import ExperimentConfig, load_config
from src.datasets.codec import Codec
from src.datasets.dataset import SuffixDataset
from src.datasets.info import DatasetInfo, read_split
from src.inference import generate_suffixes, generation_batch_size, generations_path
from src.model import TransformerCVAE, latest_best_model_path, load_checkpoint


def run(config: ExperimentConfig, model_path: Path | None = None) -> None:
    """Generate suffixes for every prefix of the test split and write them out.

    Args:
        config: The validated experiment config.
        model_path: The checkpoint to generate with. Defaults to the best model of the most
            recent run of this config, which is what a config alone can identify: the runs of
            one config differ only in the timestamp their name ends on.
    """
    ensure_dataset(config.data)
    torch.manual_seed(config.seed)

    # The codec decodes the generations back into event sequences.
    dataset_info = DatasetInfo.load(config.data)
    codec = Codec(dataset_info)
    test_dataset = SuffixDataset(
        read_split(config.data, split='test', info=dataset_info), dataset_info, codec
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=generation_batch_size(inference=config.inference, upper_bound=config.data.batch_size),
        shuffle=False,
        num_workers=config.data.num_workers,
    )

    if model_path is None:
        model_path = latest_best_model_path(
            config.training.best_model_dir, f'{config.data.dir.name}/{config.experiment_name}'
        )
    model = TransformerCVAE.from_checkpoint(
        load_checkpoint(model_path), dataset_info, device=config.training.device
    )

    # The output file is named after the checkpoint's run.
    path = generations_path(config.inference.generations_dir, f'{config.data.dir.name}/{model_path.stem}')
    path.parent.mkdir(parents=True, exist_ok=True)

    print(
        f'Generating {config.inference.num_samples} suffixes for each of '
        f'{len(test_loader.dataset)} test prefixes, with {model_path}',
        flush=True,
    )

    generations = generate_suffixes(
        model,
        test_loader,
        codec,
        num_samples=config.inference.num_samples,
        device=torch.device(config.training.device),
    )

    generations.to_parquet(path=path, index=False)
    print(f'Wrote {len(generations)} generated suffixes to {path}')


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Generate test-split suffixes from a trained model.'
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
