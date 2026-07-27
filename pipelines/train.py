import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from pipelines.preprocess import ensure_dataset
from src.configs import DatasetInfo, ExperimentConfig, load_config
from src.datasets.codec import Codec
from src.datasets.dataset import SuffixDataset
from src.models import AttentionCVAE, best_model_path_for_epoch, save_checkpoint
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

    dataset_info = DatasetInfo.build(config.data)
    # Fit once, on train, and shared by both splits: val must be encoded with the train
    # vocabulary, not one refit on itself.
    codec = Codec(dataset_info)

    train_loader = DataLoader(
        dataset=SuffixDataset(dataset_info.train, dataset_info, codec),
        batch_size=config.data.batch_size,
        shuffle=True,
        generator=generator,
        num_workers=config.data.num_workers,
    )
    val_loader = DataLoader(
        dataset=SuffixDataset(dataset_info.val, dataset_info, codec),
        batch_size=config.data.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
    )
    print(
        f'Training on {len(train_loader.dataset)} prefix/suffix pairs, '
        f'validating on {len(val_loader.dataset)}'
    )

    model = AttentionCVAE(config.model, dataset_info).to(config.training.device)

    def save_best(epoch: int, val_loss: float) -> None:
        """
        Decorator for train()'s on_best_epoch callback, which saves the model to disk.
        Defined here so it can access the config and model objects without passing them through train().
        Args:
            epoch: The epoch number that just finished.
            val_loss: The validation loss that just finished.
        """
        path = save_checkpoint(
            model,
            model_config=config.model.model_dump(),
            path=best_model_path_for_epoch(config.training.checkpoint_dir, epoch),
        )
        print(f'New best model (val loss {val_loss:.4f}) saved at {path}')

    train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        on_best_epoch=save_best,
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
