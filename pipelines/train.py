import argparse
from datetime import datetime
from pathlib import Path
import torch
from torch.utils.data import DataLoader

from pipelines.preprocess import ensure_dataset
from src.configs import ExperimentConfig, load_config
from src.datasets.dataset import SuffixDataset, fixed_subset
from src.datasets.description import DatasetDescription
from src.inference import generation_batch_size
from src.model import TransformerCVAE, load_checkpoint
from src.training.train import train


def run(config: ExperimentConfig, resume_path: Path | None = None) -> None:
    """
    Train the model an experiment config describes, on the dataset it names.
    The dataset is preprocessed first if its splits are not on disk yet, so a config file
    and a raw log are all a run needs.
    Args:
        config: The validated experiment config.
        resume_path: A checkpoint to carry on from, or `None` to start a new run. The run keeps
            the name the checkpoint carries, so it writes to the TensorBoard directory and the
            files the interrupted run was writing to.
    Raises:
        ValueError: If the checkpoint holds no training state, or was trained with a different
            model config than the one given.
    """
    # Preprocess the dataset if it hasn't been done yet
    ensure_dataset(config.data)

    # Seeded before anything is built, so weight initialization and shuffling are both reproducible.
    torch.manual_seed(config.seed)
    generator = torch.Generator().manual_seed(config.seed)

    description = DatasetDescription.load(config.data)

    # Build the datasets and loaders. The test split is not read: it is generated for by
    # `pipelines/generate.py` and never seen here.
    train_dataset = SuffixDataset(config.data, split='train', description=description)
    validation_dataset = SuffixDataset(config.data, split='val', description=description)
    
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

    model = TransformerCVAE(config.model, description).to(config.training.device)

    checkpoint = load_checkpoint(resume_path) if resume_path is not None else None
    if checkpoint is None:
        # The run name names the checkpoint files and the TensorBoard log directory
        run_name = f'{config.data.dir.name}/{config.experiment_name}-{datetime.now():%Y%m%d-%H%M%S}'
    else:
        missing = {
            'run_name', 'optimizer_state', 'early_stopping_state', 'rng_state'
        } - checkpoint.keys()
        if missing:
            raise ValueError(
                f'{resume_path} is missing {sorted(missing)}: it can be generated with, but not '
                'resumed from. Start a new run instead.'
            )
        if checkpoint['model_config'] != config.model.model_dump():
            raise ValueError(
                f'{resume_path} was trained with a different model than this config describes. '
                'Resume with the config the run started from.'
            )
        # Resuming continues the interrupted run rather than starting one beside it.
        run_name = checkpoint['run_name']

    train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        generation_loader=generation_loader,
        run_name=run_name,
        model_config=config.model.model_dump(),
        generator=generator,
        resume=checkpoint,
        generation_samples=config.inference.num_samples,
        description=description,
        loss_config=config.loss,
        optimizer_config=config.optimizer,
        training=config.training,
        early_stopping_config=config.early_stopping,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description='Train a suffix-prediction model from a config file.')
    parser.add_argument('-c', '--config', type=Path, required=True,
                        help="Path to this experiment's config YAML.")
    parser.add_argument('-r', '--resume', type=Path,
                        help='Path to a checkpoint to carry on from. The run keeps its name, so '
                             'it writes to the same TensorBoard directory and the same files.')
    args = parser.parse_args()

    run(load_config(args.config), args.resume)


if __name__ == '__main__':
    main()
