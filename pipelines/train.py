import argparse
from datetime import datetime

import torch
from torch.utils.data import DataLoader

from src import paths
from src.cli import banner, step
from src.configs import ExperimentConfig, load_config
from src.datasets.codec import DatasetCodec
from src.datasets.dataset import TraceDataset, fixed_subset
from src.identity import RunIdentity
from src.inference.generate import generation_batch_size
from src.logs import Split
from src.model import build_model
from src.training import train


def run(config: ExperimentConfig) -> None:
    """
    Train the model an experiment config describes, on the dataset it names.
    The dataset must have been preprocessed already.
    Args:
        config: The validated experiment config.
    """
    paths.require_preprocessed(config.data.name)
    # Checkpoints are selected on EMSC against the validation split's continuations, so the index
    # is as much a precondition of training as the splits are.
    paths.CONTINUATIONS.require(dataset=config.data.name, split=Split.VAL)

    # Seeded before anything is built, so weight initialization and shuffling are both reproducible.
    torch.manual_seed(config.seed)
    generator = torch.Generator().manual_seed(config.seed)

    run = RunIdentity(
        dataset=config.data.name,
        model=config.model.name,
        tag=f'{datetime.now():%Y%m%d-%H%M%S}',
    )

    banner(
        'Training a suffix-prediction model',
        {
            'run': run,
            'dataset': config.data.name,
            'model': config.model.name,
            'device': config.training.device,
            'steps': f'at most {config.training.max_steps:,}, validating every '
            f'{config.training.val_every_n_steps:,}',
            'batch': f'{config.dataloader.batch_size} pairs, '
            f'{config.dataloader.num_workers} loader workers',
            'optimizer': f'Adam, lr {config.optimizer.lr} after '
            f'{config.optimizer.warmup_steps} warmup steps, '
            f'weight decay {config.optimizer.weight_decay}',
            'continuations': paths.CONTINUATIONS.path(dataset=config.data.name, split=Split.VAL),
            'checkpoints': paths.BEST_CHECKPOINT.path(run),
        },
    )

    with step('Loading the dataset codec'):
        codec = DatasetCodec.load(config.data)

    with step(f'Building the model and moving it onto {config.training.device}'):
        model = build_model(config.model, codec).to(config.training.device)
        parameters = sum(parameter.numel() for parameter in model.parameters())
        print(f'  {parameters:,} parameters', flush=True)

    # A loader without persistent workers forks its whole pool every time it is iterated, and the
    # two validation loaders are iterated once per validation check. Forking a process that torch
    # and CUDA have already put threads in is what Python warns about, so the pools are forked
    # once each and kept, at the cost of holding every worker for the length of the run.
    workers = config.dataloader.num_workers
    persistent_workers = workers > 0

    # Build the datasets and loaders
    with step('Reading and encoding the train split'):
        train_dataset = TraceDataset(codec=codec, split=Split.TRAIN)
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=config.dataloader.batch_size,
        shuffle=True,
        generator=generator,
        num_workers=workers,
        persistent_workers=persistent_workers,
    )

    with step('Reading and encoding the validation split'):
        validation_dataset = TraceDataset(codec=codec, split=Split.VAL)
    # Validation and generation loaders are fixed subsets of the validation split, so every run
    # of a config reads the same traces and their curves can be laid over each other.
    val_loader = DataLoader(
        dataset=fixed_subset(
            validation_dataset, size=config.training.validation_pairs, generator=generator
        ),
        batch_size=config.dataloader.batch_size,
        shuffle=False,
        num_workers=workers,
        persistent_workers=persistent_workers,
    )
    generation_loader = DataLoader(
        dataset=fixed_subset(
            validation_dataset, size=config.training.generation_pairs, generator=generator
        ),
        batch_size=generation_batch_size(
            inference=config.inference,
            num_samples=config.inference.validation_samples,
            prefixes_upper_bound=config.dataloader.batch_size,
        ),
        shuffle=False,
        num_workers=workers,
        persistent_workers=persistent_workers,
    )

    print(
        f'Training on {len(train_loader.dataset):,} prefix/suffix pairs, scoring '
        f'{len(val_loader.dataset):,} of the {len(validation_dataset):,} validation pairs and '
        f'generating for {len(generation_loader.dataset):,}'
    )

    train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        generation_loader=generation_loader,
        run=run,
        experiment_config=config.model_dump(),
        generation_samples=config.inference.validation_samples,
        codec=codec,
        dataset=config.data.name,
        optimizer_config=config.optimizer,
        training=config.training,
        early_stopping_config=config.early_stopping,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description='Train a suffix-prediction model.')
    parser.add_argument(
        '-c',
        '--config',
        type=paths.existing_file,
        metavar='CONFIG',
        required=True,
        help="Path to this experiment's dataset config, e.g. config/datasets/bpic17.yaml.",
    )
    parser.add_argument(
        '-m',
        '--model',
        type=paths.existing_file,
        metavar='MODEL',
        required=True,
        help='Path to the architecture to train, e.g. config/models/cvae.yaml. Its `model.kind` '
        'is what selects the class that gets built, and it also carries every setting that does '
        "not vary with the dataset: the training loop, the optimizer, and the model's own loss.",
    )
    args = parser.parse_args()

    run(load_config(args.model, args.config))


if __name__ == '__main__':
    main()
