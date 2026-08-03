import argparse
from datetime import datetime
from pathlib import Path
import torch
from torch.utils.data import DataLoader

from pipelines.preprocess import ensure_dataset
from src.configs import ExperimentConfig, load_config
from src.datasets.codec import Codec
from src.datasets.dataset import SuffixDataset, fixed_subset
from src.datasets.info import DatasetInfo, read_split
from src.inference import generation_batch_size
from src.model import TransformerCVAE, best_model_path, save_checkpoint
from src.training.train import train


def run(config: ExperimentConfig) -> None:
    """
    Train the model an experiment config describes, on the dataset it names.
    The dataset is preprocessed first if its splits are not on disk yet, so a config file
    and a raw log are all a run needs.
    Args:
        config: The validated experiment config.
    """
    # Preprocess the dataset if it hasn't been done yet
    ensure_dataset(config.data)

    # Seeded before anything is built, so weight initialization and shuffling are both reproducible.
    torch.manual_seed(config.seed)
    generator = torch.Generator().manual_seed(config.seed)

    dataset_info = DatasetInfo.load(config.data)
    codec = Codec(dataset_info)

    # Build the datasets and loaders. The test split is not read: it is generated for by
    # `pipelines/generate.py` and never seen here.
    train_dataset = SuffixDataset(
        read_split(config.data, split='train', info=dataset_info), dataset_info, codec
    )
    validation_dataset = SuffixDataset(
        read_split(config.data, split='val', info=dataset_info), dataset_info, codec
    )
    
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=config.data.batch_size,
        shuffle=True,
        generator=generator,
        num_workers=config.data.num_workers,
    )
    # Validation and generation loaders are fixed subsets of the validation split, so every run of a config
    # reads the same pairs and their curves can be laid over each other.
    val_loader = DataLoader(
        dataset=fixed_subset(validation_dataset, size=config.training.validation_pairs, generator=generator),
        batch_size=config.data.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
    )
    generation_loader = DataLoader(
        dataset=fixed_subset(validation_dataset, size=config.training.generation_pairs, generator=generator),
        batch_size=generation_batch_size(inference=config.inference, upper_bound=config.data.batch_size),
        shuffle=False,
        num_workers=config.data.num_workers,
    )
    print(
        f'Training on {len(train_loader.dataset)} prefix/suffix pairs, scoring '
        f'{len(val_loader.dataset)} of the {len(validation_dataset)} validation pairs and '
        f'generating for {len(generation_loader.dataset)}'
    )

    model = TransformerCVAE(config.model, dataset_info).to(config.training.device)

    # The run name is used to name the best-model checkpoint file and the TensorBoard log directory
    run_name = f'{config.data.dir.name}/{config.experiment_name}-{datetime.now():%Y%m%d-%H%M%S}'

    def save_best(step: int, selection_score: float) -> None:
        """
        Decorator for train()'s on_best_step callback, which saves the model to disk.
        Defined here so it can access the config and model objects without passing them through train().
        Args:
            step: The optimizer step that just validated.
            selection_score: The generation score it reached, as `1 - activity_dls_mean`.
        """
        path = save_checkpoint(
            model,
            model_config=config.model.model_dump(),
            step=step,
            selection_score=selection_score,
            path=best_model_path(config.training.best_model_dir, run_name),
        )
        print(f'New best model (step {step}, score {selection_score:.4f}) saved at {path}')

    train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        generation_loader=generation_loader,
        run_name=run_name,
        on_best_step=save_best,
        generation_samples=config.inference.num_samples,
        loss_config=config.loss,
        optimizer_config=config.optimizer,
        training=config.training,
        early_stopping_config=config.early_stopping,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description='Train a suffix-prediction model from a config file.')
    parser.add_argument('-c', '--config', type=Path, required=True,
                        help="Path to this experiment's config YAML.")
    args = parser.parse_args()

    run(load_config(args.config))


if __name__ == '__main__':
    main()
