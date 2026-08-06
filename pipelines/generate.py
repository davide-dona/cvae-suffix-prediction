import argparse
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from pipelines.preprocess import require_dataset
from src.configs import ExperimentConfig, load_config
from src.datasets.dataset import TraceDataset
from src.datasets.description import DatasetDescription
from src.inference import (
    generate_batch,
    generation_batch_size,
    generations_path,
    open_generations,
    table_from_generations,
)
from src.model import TransformerCVAE, load_checkpoint


def run(config: ExperimentConfig, model_path: Path) -> None:
    """Generate suffixes for every prefix of the test split and write them out.

    Args:
        config: The validated experiment config.
        model_path: The checkpoint to generate with. Named rather than guessed at: a config
            matches every run ever started from it, and picking one of them is a decision the
            caller makes, not one to be inferred from a filename.
    """
    require_dataset(config.data)
    torch.manual_seed(config.seed)

    # Load the model from the checkpoint and put it on the right device
    model = TransformerCVAE.from_checkpoint(
        load_checkpoint(model_path), description, device=config.training.device
    )
    model.eval()

    # Build the DataLoader for the test split
    description = DatasetDescription.load(config.data)
    test_dataset = TraceDataset(description=description, split='test')
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=generation_batch_size(inference=config.inference, upper_bound=config.data.batch_size),
        sampler=test_dataset.length_sorted_indices(),       # sort the prefixes by length so the batches are more uniform and generation is faster
        num_workers=config.data.num_workers,                
    )

    # The output file is named after the checkpoint's run.
    path = generations_path(config.inference.generations_dir, f'{config.data.dir.name}/{model_path.stem}')
    device = torch.device(config.training.device)

    print(
        f'Generating {config.inference.num_samples} suffixes for each of '
        f'{len(test_dataset)} test prefixes, with {model_path}',
        flush=True,
    )

    # Write the generation while it is being produced, avoiding a huge in-memory DataFrame.
    with open_generations(path) as parquetWriter:
        for batch in tqdm(iterable=test_loader, desc='Generating', unit='batch'):
            generations = generate_batch(
                model=model,
                batch=batch.to(device),
                num_samples=config.inference.num_samples,
                description=description,
            )
            # Write the generations to the Parquet file in a single table, one row per prefix.
            parquetWriter.write_table(table=table_from_generations(generations))

    print(f'Wrote generated suffixes to {path}')


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Generate test-split suffixes from a trained model.'
    )
    parser.add_argument('-c', '--config', type=Path, required=True,
                        help="Path to this experiment's config YAML.")
    parser.add_argument('-m', '--model', type=Path, required=True,
                        help='Path to the checkpoint to generate with, from '
                             "`training.best_model_dir` or `training.checkpoint_dir`.")
    args = parser.parse_args()

    run(load_config(args.config), args.model)


if __name__ == '__main__':
    main()
