import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src import paths
from src.cli import banner, step
from src.configs import load_generation_config
from src.configs.schema import SamplingConfig, sampling_of
from src.datasets.codec import DatasetCodec
from src.datasets.dataset import TraceDataset
from src.identity import RunIdentity
from src.inference.generate import generate_batch, generation_batch_size
from src.inference.generation_store import GenerationWriter
from src.inference.tuning import TuningReport
from src.logs import Split
from src.model import load_checkpoint, model_from_checkpoint
from src.suffixes import ActivityCodes


def run(
    checkpoint_path: Path,
    *,
    device: str | None,
    num_samples: int | None,
    tuning: Path | None,
    sampling: SamplingConfig | None,
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
    run = RunIdentity.from_dict(checkpoint['run'])
    if tuning is not None:
        sampling = TuningReport.read(tuning).sampling_for(run)
    config = load_generation_config(
        checkpoint['experiment_config'],
        device=device,
        num_samples=num_samples,
        sampling=sampling,
    )

    paths.require_preprocessed(config.data.name)
    torch.manual_seed(config.seed)

    # The output file is named after the run the checkpoint carries, not after the file it was
    # read from, so the generations land under the run that produced them whatever it is called.
    path = paths.GENERATIONS.prepare(run)
    device = torch.device(config.training.device)
    batch_size = generation_batch_size(
        inference=config.inference,
        num_samples=config.inference.evaluation_samples,
        prefixes_upper_bound=config.dataloader.batch_size,
    )
    # A checkpoint that has been trimmed for publishing still carries both of these.
    trained_step, score = checkpoint.get('step'), checkpoint.get('selection_score')
    # None for an architecture that reads its heads at their mode, which is what the file records.
    drawn_with = sampling_of(config.model)

    banner(
        'Generating suffixes',
        {
            'run': run,
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
    with GenerationWriter(path, run, vocabulary=codes.vocabulary, sampling=drawn_with) as writer:
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Generate test-split suffixes from a trained model.'
    )
    parser.add_argument(
        '-m',
        '--checkpoint',
        type=paths.existing_file,
        metavar='CHECKPOINT',
        required=True,
        help='Path to the checkpoint to generate with, from `pretrained/`, '
        '`outputs/checkpoints/best/` or `outputs/checkpoints/last/`. Its own config is what '
        'the model and the dataset are read from.',
    )
    parser.add_argument(
        '-d',
        '--device',
        type=str,
        default=None,
        metavar='DEVICE',
        help='Overrides the device to generate on, e.g. cpu or cuda:1. Defaults to the device '
        'the run was trained with.',
    )
    parser.add_argument(
        '-n',
        '--num-samples',
        type=int,
        default=None,
        metavar='N',
        help='How many suffixes to draw per prefix, overriding the config the checkpoint '
        "carries. Defaults to the run's own `inference.evaluation_samples`.",
    )
    parser.add_argument(
        '-t',
        '--tuning',
        type=paths.existing_file,
        default=None,
        metavar='REPORT',
        help='Path to the tuning report whose chosen sampler to draw with, from '
        '`outputs/tuning/`. Refused if it was written for a different run. This is the usual '
        'way a sampler reaches a generation: the report is what `pipelines.tune` picked on the '
        'validation split.',
    )
    parser.add_argument(
        '--temperature',
        type=float,
        default=None,
        metavar='T',
        help='Divides the activity logits before the draw, overriding the config the checkpoint '
        'carries. For a one-off; `--tuning` is how a searched value is handed over. Must be '
        'given together with `--top-p`.',
    )
    parser.add_argument(
        '--top-p',
        type=float,
        default=None,
        metavar='P',
        help='Keeps the smallest set of activities summing to this, overriding the config the '
        'checkpoint carries. Must be given together with `--temperature`.',
    )
    args = parser.parse_args()

    explicit = (args.temperature, args.top_p)
    if args.tuning is not None and any(value is not None for value in explicit):
        parser.error('name either --tuning or --temperature/--top-p, not both')
    if any(value is not None for value in explicit) and None in explicit:
        parser.error('--temperature and --top-p are given together: a sampler is the pair')
    sampling = (
        SamplingConfig(temperature=args.temperature, top_p=args.top_p)
        if args.temperature is not None
        else None
    )

    run(
        args.checkpoint,
        device=args.device,
        num_samples=args.num_samples,
        tuning=args.tuning,
        sampling=sampling,
    )


if __name__ == '__main__':
    main()
