import argparse
from datetime import datetime
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset, Subset

from pipelines.preprocess import ensure_dataset
from src.configs import DatasetInfo, ExperimentConfig, load_config
from src.datasets.codec import Codec
from src.datasets.dataset import SuffixDataset
from src.inference import generation_batch_size
from src.model import TransformerCVAE, best_model_path, save_checkpoint
from src.training.train import train

# How much of the validation split each validation pass reads. All three are fixed slices
# rather than fresh draws, so a curve moves when the model does and not when the sample does.
#
# The whole split is far more than any of them needs and validation is on the critical path: on
# bpic-2017 it is 167k pairs, and scoring all of them twice cost more wall clock than the 400
# training steps it was measuring. A fixed 20k puts the standard error of the mean well under
# the differences a run is read for.
VALIDATION_PAIRS = 20_000

# Generation is much the more expensive pass, a suffix costing one decoder pass per event
# rather than one per suffix, so it gets a slice smaller again. It draws `inference.num_samples`
# per prefix, the same number prediction and evaluation use, so `gen/activity_dls_best` and
# `gen/sample_diversity` mean on the training curve what they mean in the final report.
GENERATION_PAIRS = 256


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
    # One dataset, two fixed slices of it: the same generator draws both, so the smaller is not
    # a subset of the larger by construction and a run's two views of the split are independent.
    val_dataset = SuffixDataset(dataset_info.val, dataset_info, codec)
    val_loader = DataLoader(
        dataset=_fixed_subset(val_dataset, size=VALIDATION_PAIRS, generator=generator),
        batch_size=config.data.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
    )
    generation_loader = DataLoader(
        dataset=_fixed_subset(val_dataset, size=GENERATION_PAIRS, generator=generator),
        batch_size=generation_batch_size(config.inference.num_samples, config.data.batch_size),
        shuffle=False,
        num_workers=config.data.num_workers,
    )
    print(
        f'Training on {len(train_loader.dataset)} prefix/suffix pairs, scoring '
        f'{len(val_loader.dataset)} of the {len(val_dataset)} validation pairs and '
        f'generating for {len(generation_loader.dataset)}'
    )

    model = TransformerCVAE(config.model, dataset_info).to(config.training.device)

    # TensorBoard reads one directory as one run, so every run needs its own or their curves
    # are overlaid into a single unreadable one. The timestamp is what separates two runs of
    # the same config, and sorts them in the order they were started. The dataset goes in
    # front as a directory of its own: losses are only comparable within a dataset, and a
    # nested run shows up grouped, so TensorBoard's filter can pick out one dataset's runs.
    # Checkpoints are named by the same thing, for the same reason: without it a second run of
    # a config writes over the first one's epochs.
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


def _fixed_subset(dataset: Dataset, *, size: int, generator: torch.Generator) -> Dataset:
    """A random slice of `dataset`, or the whole of it if it is already no bigger.

    Drawn once, so every validation of a run reads the same pairs and two of its points differ
    because the model moved rather than because the sample did. Seeded through the run's own
    generator, so two runs of one config read the same slice and their curves can be laid over
    each other.

    Args:
        dataset: The split to take from.
        size: How many items to keep.
        generator: The run's seeded generator.
    Returns:
        The slice, as a `Subset` the loaders can be built on directly.
    """
    if len(dataset) <= size:
        return dataset
    indices = torch.randperm(n=len(dataset), generator=generator)[:size]
    return Subset(dataset=dataset, indices=indices.tolist())


def main() -> None:
    parser = argparse.ArgumentParser(description='Train a suffix-prediction model from a config file.')
    parser.add_argument('-c', '--config', type=Path, required=True,
                        help="Path to this experiment's config YAML.")
    args = parser.parse_args()

    run(load_config(args.config))


if __name__ == '__main__':
    main()
