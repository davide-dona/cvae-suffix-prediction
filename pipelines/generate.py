from __future__ import annotations

from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm

from src import paths
from src.artifacts import sha256
from src.cli import banner, step
from src.datasets.codec import DatasetCodec
from src.datasets.dataset import TraceDataset
from src.inference.generate import generate_batch, generation_batch_size
from src.inference.generation_store import GenerationWriter
from src.inference.tuning import TuningReport
from src.logs import Split
from src.model import load_checkpoint, model_from_checkpoint
from src.runtime import output_path, save_config, start_stage
from src.suffixes import ActivityCodes
from src.validation import validate_sampling, validate_training


def run(
    checkpoint_path: Path,
    *,
    device: str | None,
    num_samples: int | None,
    tuning: Path | None,
    sampling: DictConfig | None,
    num_workers: int | None = None,
) -> None:
    """Generate suffixes for every prefix of the test split and write them out.

    Args:
        checkpoint_path: The checkpoint to generate with. Named rather than guessed at: a config
            matches every run ever started from it, and picking one of them is a decision the
            caller makes, not one to be inferred from a filename. It carries the config of the
            run that wrote it, so nothing about the model or the dataset is passed alongside it.
        device: Overrides the run's own `training.device`, e.g. to generate on a different
            machine than the one it trained on. `None` keeps it.
        num_samples: How many suffixes to draw per prefix, or `None` for the run's own
            `inference.evaluation_samples`.
        tuning: A tuning report to read the sampler out of, or `None`. Named rather than looked
            up beside the checkpoint: which operating point a set of weights is read at is a
            decision the caller makes, and the report is checked against this run's identity
            before it is used.
        sampling: The sampler to draw with, or `None` for the one the run trained under.
            Mutually exclusive with `tuning`, which is enforced by the CLI.
    """
    # The run's own config. Read before the codec, since it is what says which dataset's codec
    # to read.
    with step(f'Reading the checkpoint at {checkpoint_path}'):
        checkpoint = load_checkpoint(checkpoint_path)
    metadata = {
        'dataset': checkpoint['config']['data']['name'],
        'model': checkpoint['config']['model']['name'],
        'checkpoint_sha256': sha256(checkpoint_path),
    }
    if tuning is not None and sampling is not None:
        raise ValueError('Choose either tuning or sampling overrides')
    if tuning is not None:
        sampling = TuningReport.read(tuning).sampling_for(metadata['checkpoint_sha256'])
    config = OmegaConf.create(checkpoint['config'])
    if device is not None:
        config.training.device = device
    if num_samples is not None:
        config.inference.evaluation_samples = num_samples
    if num_workers is not None:
        config.dataloader.num_workers = num_workers
    if sampling is not None:
        if config.model.kind != 'transformer':
            raise ValueError('Only transformer supports sampling overrides')
        validate_sampling(sampling)
        config.model.sampling = sampling
    validate_training(config)
    save_config(
        OmegaConf.create(
            {
                'checkpoint': str(checkpoint_path.resolve()),
                'checkpoint_sha256': metadata['checkpoint_sha256'],
                'effective': OmegaConf.to_container(config, resolve=True),
                'tuning': str(tuning) if tuning is not None else None,
            }
        )
    )

    paths.require_preprocessed(config.data.name)
    torch.manual_seed(config.seed)

    # The output file is named after the run the checkpoint carries, not after the file it was
    # read from, so the generations land under the run that produced them whatever it is called.
    path = output_path('generations.parquet')
    device = torch.device(config.training.device)
    batch_size = generation_batch_size(
        inference=config.inference,
        num_samples=config.inference.evaluation_samples,
        prefixes_upper_bound=config.dataloader.batch_size,
    )
    # A checkpoint that has been trimmed for publishing still carries both of these.
    trained_step, score = checkpoint.get('step'), checkpoint.get('selection_score')
    # None for an architecture that reads its heads at their mode, which is what the file records.
    drawn_with = config.model.get('sampling')

    banner(
        'Generating suffixes',
        {
            'checkpoint_sha256': metadata['checkpoint_sha256'],
            'dataset': config.data.name,
            'model': f'{config.model.name} (step {trained_step}, selection score {score:.4f})'
            if trained_step is not None and score is not None
            else config.model.name,
            'device': device,
            'samples': f'{config.inference.evaluation_samples} suffixes per prefix',
            'sampling': f'temperature {drawn_with.temperature}, top_p {drawn_with.top_p}'
            if drawn_with is not None
            else 'greedy heads; the draws vary in z alone',
            'batch': f'{batch_size} prefixes, {config.dataloader.num_workers} loader workers',
            'generations': path,
        },
    )

    with step('Loading the dataset codec'):
        codec = DatasetCodec.load(config.data)

    with step(f'Building the model and moving it onto {device}'):
        model = model_from_checkpoint(checkpoint, codec, device=config.training.device)
        if drawn_with is not None:
            model.decoder.read_with(drawn_with)
        model.eval()

    # Build the DataLoader for the test split
    with step('Reading and encoding the test split'):
        test_dataset = TraceDataset(codec=codec, split=Split.TEST)

    with step(f'Sorting {len(test_dataset):,} prefixes by suffix length'):
        sampler = test_dataset.length_sorted_indices()

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        # sort the prefixes by length so the batches are more uniform and generation is faster
        sampler=sampler,
        num_workers=config.dataloader.num_workers,
    )

    print(
        f'Generating {config.inference.evaluation_samples} suffixes for each of '
        f'{len(test_dataset):,} test prefixes, in {len(test_loader):,} batches',
        flush=True,
    )

    # The one codebook this run spells its suffixes on, seeded from every name the activity channel
    # can decode to so no name is ever coded on the fly. The continuation index is seeded from the
    # same list, which is what lets evaluation compare the two without translating either.
    codes = ActivityCodes.of(codec.activity.names)

    # Write the generation while it is being produced, avoiding a huge in-memory DataFrame.
    with GenerationWriter(
        path, metadata, vocabulary=codes.vocabulary, sampling=drawn_with
    ) as writer:
        for batch in tqdm(iterable=test_loader, desc='Generating', unit='batch'):
            generations = generate_batch(
                model=model,
                batch=batch.to(device),
                num_samples=config.inference.evaluation_samples,
                codec=codec,
                codes=codes,
            )
            # Write the generations to the Parquet file in a single block, one row per prefix.
            writer.write(generations)

    print(f'Wrote generated suffixes to {path}')


@hydra.main(version_base='1.3', config_path='../config', config_name='generate')
def main(cfg: DictConfig) -> None:
    start_stage(cfg)
    run(
        Path(cfg.checkpoint),
        device=cfg.device,
        num_samples=cfg.num_samples,
        tuning=Path(cfg.tuning) if cfg.tuning is not None else None,
        sampling=cfg.sampling,
        num_workers=cfg.num_workers,
    )


if __name__ == '__main__':
    main()
