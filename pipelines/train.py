import argparse
from datetime import datetime
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from pipelines.preprocess import require_dataset
from src.configs import ExperimentConfig, load_config
from src.datasets.codec import DatasetCodec
from src.datasets.dataset import TraceDataset, fixed_subset
from src.inference import generation_batch_size
from src.model import TransformerCVAE, load_checkpoint
from src.paths import Split
from src.training.train import train


def resumed(resume_path: Path) -> tuple[ExperimentConfig, dict]:
    """
    Read a checkpoint together with the config of the run that wrote it.

    Args:
        resume_path: The checkpoint to carry on from.
    Returns:
        Its config, and the checkpoint itself.
    Raises:
        ValueError: If the checkpoint carries no training state, and so can only be generated with.
    """
    checkpoint = load_checkpoint(resume_path)
    missing = {
        'experiment_config',
        'run_name',
        'optimizer_state',
        'early_stopping_state',
        'rng_state',
    } - checkpoint.keys()
    if missing:
        raise ValueError(
            f'{resume_path} is missing {sorted(missing)}: it can be generated with, but not '
            'resumed from. Start a new run instead.'
        )
    return ExperimentConfig.model_validate(checkpoint['experiment_config']), checkpoint


def run(config: ExperimentConfig, checkpoint: dict | None = None) -> None:
    """
    Train the model an experiment config describes, on the dataset it names.
    The dataset must have been preprocessed already.
    Args:
        config: The validated experiment config.
        checkpoint: A checkpoint to carry on from, as read by `resumed`, or `None` to start a
            new run. The run keeps the name the checkpoint carries, so it writes to the
            TensorBoard directory and the files the interrupted run was writing to.
    """
    require_dataset(config.data.name)

    # Seeded before anything is built, so weight initialization and shuffling are both reproducible.
    torch.manual_seed(config.seed)
    generator = torch.Generator().manual_seed(config.seed)

    codec = DatasetCodec.load(config.data)

    model = TransformerCVAE(config.model, codec).to(config.training.device)

    # Build the datasets and loaders
    train_dataset = TraceDataset(codec=codec, split=Split.TRAIN)
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=config.data.batch_size,
        shuffle=True,
        generator=generator,
        num_workers=config.data.num_workers,
    )

    validation_dataset = TraceDataset(codec=codec, split=Split.VAL)
    # Validation and generation loaders are fixed subsets of the validation split, so every run
    # of a config reads the same traces and their curves can be laid over each other.
    val_loader = DataLoader(
        dataset=fixed_subset(
            validation_dataset, size=config.training.validation_pairs, generator=generator
        ),
        batch_size=config.data.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
    )
    generation_loader = DataLoader(
        dataset=fixed_subset(
            validation_dataset, size=config.training.generation_pairs, generator=generator
        ),
        batch_size=generation_batch_size(
            inference=config.inference, upper_bound=config.data.batch_size
        ),
        shuffle=False,
        num_workers=config.data.num_workers,
    )

    print(
        f'Training on {len(train_loader.dataset)} prefix/suffix pairs, scoring '
        f'{len(val_loader.dataset)} of the {len(validation_dataset)} validation pairs and '
        f'generating for {len(generation_loader.dataset)}'
    )

    # When resuming, keep the run name the checkpoint carries, so it writes to the same
    # TensorBoard directory and the same files.
    run_name = (
        f'{config.data.name}/{config.experiment_name}-{datetime.now():%Y%m%d-%H%M%S}'
        if checkpoint is None
        else checkpoint['run_name']
    )

    train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        generation_loader=generation_loader,
        run_name=run_name,
        experiment_config=config.model_dump(),
        generator=generator,
        resume=checkpoint,
        generation_samples=config.inference.num_samples,
        codec=codec,
        loss_config=config.loss,
        optimizer_config=config.optimizer,
        training=config.training,
        early_stopping_config=config.early_stopping,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description='Train a suffix-prediction model.')
    # A run is either started from a config or carried on from a checkpoint, which already
    # describes the run that wrote it. Two descriptions of one run could only disagree.
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        '-c',
        '--config',
        type=Path,
        help="Path to this experiment's config YAML, to start a new run.",
    )
    source.add_argument(
        '-r',
        '--resume',
        type=Path,
        help='Path to a checkpoint to carry on from, its config included. The '
        'run keeps its name, so it writes to the same TensorBoard directory '
        'and the same files.',
    )
    args = parser.parse_args()

    if args.resume is not None:
        run(*resumed(args.resume))
    else:
        run(load_config(args.config))


if __name__ == '__main__':
    main()
