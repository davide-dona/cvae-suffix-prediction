from __future__ import annotations

import itertools
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
from src.datasets.dataset import TraceDataset, fixed_subset
from src.evaluation.scores import AccuracyScores, ConformanceScores, DistributionScores
from src.inference.generate import generate_batch, generation_batch_size
from src.inference.tuning import (
    SearchPass,
    TuningPoint,
    TuningReport,
)
from src.logs import ContinuationIndex, Split
from src.logs.declare import ConformanceChecker
from src.model import Transformer, load_checkpoint, model_from_checkpoint
from src.runtime import output_path, save_config, start_stage
from src.suffixes import ActivityCodes
from src.validation import validate_sampling, validate_training


@torch.no_grad()
def _score(
    model: Transformer,
    loader: DataLoader,
    *,
    sampling: DictConfig,
    seed: int,
    num_samples: int,
    codec: DatasetCodec,
    codes: ActivityCodes,
    index: ContinuationIndex,
    checker: ConformanceChecker,
    device: torch.device,
) -> TuningPoint:
    """Draw the validation subset once under one sampler, and score what came out.

    Args:
        model: The trained model, in evaluation mode. Its decoder is re-read with `sampling`
            here, which changes nothing it learned.
        loader: The validation subset, the same prefixes in the same order at every point.
        sampling: The sampler this point is measuring.
        seed: Reset before the pass rather than left to run on from the previous point, so two
            points differ by the sampler and not by where the random stream happened to be. That
            pairing is what makes cells a few thousandths apart worth comparing at all.
        num_samples: Suffixes drawn per prefix.
        codec: The codec the split was encoded through, read in the decode direction.
        codes: The codebook the suffixes are spelled on, seeded from `codec.activity.names`.
        index: The validation split's continuations. Never the test split's: choosing an
            operating point against those would fold the held-out set into the choice, exactly as
            selecting a checkpoint on them would.
        checker: The declarative model, discovered from the train split and so the same object
            whichever split is being scored.
        device: The device to generate on.
    Returns:
        The point, its `score` the objective and the rest recorded beside it.
    """
    model.decoder.read_with(sampling)
    torch.manual_seed(seed)

    generations = [
        generation
        for batch in tqdm(
            iterable=loader, desc=f'T {sampling.temperature} p {sampling.top_p}', unit='batch'
        )
        for generation in generate_batch(
            model=model, batch=batch.to(device), num_samples=num_samples, codec=codec, codes=codes
        )
    ]
    # Over the prefixes a report reads these scores on, so the operating point is chosen on the
    # numbers that will be reported for it rather than on a wider population where a comparison
    # against a single observed continuation is an accuracy under another name.
    scored = [DistributionScores.of(one, index=index) for one in generations]
    distribution = DistributionScores.mean([one for one in scored if one.comparable])
    conformance = ConformanceScores.mean(
        [ConformanceScores.of(one, checker=checker) for one in generations]
    )
    return TuningPoint(
        sampling=OmegaConf.to_container(sampling, resolve=True),
        score=AccuracyScores.mean([AccuracyScores.of(one) for one in generations]).energy_score,
        continuation_precision=distribution.continuation_precision,
        continuation_recall=distribution.continuation_recall,
        emsc=distribution.emsc,
        conformance_mean=conformance.conformance_mean,
        unique_sample_rate=distribution.unique_sample_rate,
    )


def run(
    checkpoint_path: Path,
    *,
    device: str | None,
    pairs: int | None,
    samples: int | None,
    temperatures: list[float],
    top_ps: list[float],
    num_workers: int | None = None,
):
    """Search the sampler grid on the validation split and write what it picked.

    The stage between training and generation. It is separate from both because the sampler is
    neither trained nor a property of the test split: it is chosen after the weights are fixed,
    against the same split and the same continuations the checkpoint itself was selected on.

    Args:
        checkpoint_path: The checkpoint to search for. Named rather than guessed at, as
            `pipelines.generate` names it, and it carries the config of the run that wrote it.
        device: Overrides the run's own `training.device`. `None` keeps it.
        pairs: Validation prefixes to search over, or `None` for the run's own
            `training.generation_pairs`. The same subset at every grid point.
        samples: Suffixes drawn per prefix at each point, or `None` for the run's own
            `inference.validation_samples`.
    Raises:
        ValueError: If the checkpoint is of an architecture that reads its heads at their mode,
            which has no sampler to search.
    """
    with step(f'Reading the checkpoint at {checkpoint_path}'):
        checkpoint = load_checkpoint(checkpoint_path)
    checkpoint_hash = sha256(checkpoint_path)
    config = OmegaConf.create(checkpoint['config'])
    if device is not None:
        config.training.device = device
    if num_workers is not None:
        config.dataloader.num_workers = num_workers
    if pairs is not None:
        config.training.generation_pairs = pairs
    if samples is not None:
        config.inference.validation_samples = samples
    validate_training(config)
    save_config(
        OmegaConf.create(
            {
                'checkpoint': str(checkpoint_path.resolve()),
                'checkpoint_sha256': checkpoint_hash,
                'effective': OmegaConf.to_container(config, resolve=True),
                'temperatures': temperatures,
                'top_ps': top_ps,
            }
        )
    )

    paths.require_preprocessed(config.data.name)
    pairs = config.training.generation_pairs if pairs is None else pairs
    samples = config.inference.validation_samples if samples is None else samples

    report_path = output_path('tuning.json')
    torch_device = torch.device(config.training.device)
    grid = [
        OmegaConf.create({'temperature': temperature, 'top_p': top_p})
        for temperature, top_p in itertools.product(temperatures, top_ps)
    ]

    if not grid:
        raise ValueError('Sampler grid cannot be empty')
    for sampling in grid:
        validate_sampling(sampling)

    banner(
        'Tuning the sampler',
        {
            'checkpoint_sha256': checkpoint_hash,
            'dataset': config.data.name,
            'model': config.model.name,
            'device': torch_device,
            'split': f'{Split.VAL}, {pairs:,} prefixes, {samples} suffixes each',
            'grid': f'{len(grid)} points over temperature {temperatures} and top_p {top_ps}',
            'chosen on': 'minimum activity-sequence energy score',
            'report': report_path,
        },
    )

    with step('Loading the dataset codec'):
        codec = DatasetCodec.load(config.data)

    with step(f'Building the model and moving it onto {torch_device}'):
        model = model_from_checkpoint(checkpoint, codec, device=config.training.device)
        model.eval()
    if not isinstance(model, Transformer):
        raise ValueError(
            f'{checkpoint_hash} is a {config.model.kind}, which reads its heads at their mode and draws '
            'its variability from z. There is no sampler to search: giving it one would spread '
            'that variability over the decode steps, which is the arm it is measured against.'
        )

    with step(f'Reading and encoding the {Split.VAL} split'):
        validation_dataset = TraceDataset(codec=codec, split=Split.VAL)
    # Seeded from the config rather than carried over from training, so the subset is the same on
    # every rerun of this search and the same at every point within one.
    subset = fixed_subset(
        validation_dataset, size=pairs, generator=torch.Generator().manual_seed(config.seed)
    )
    loader = DataLoader(
        dataset=subset,
        batch_size=generation_batch_size(
            inference=config.inference,
            num_samples=samples,
            prefixes_upper_bound=config.dataloader.batch_size,
        ),
        shuffle=False,
        num_workers=config.dataloader.num_workers,
    )

    with step(f'Reading the {Split.VAL} continuations and the declarative model'):
        codes = ActivityCodes.of(codec.activity.names)
        index = ContinuationIndex.read(dataset=config.data.name, split=Split.VAL)
        checker = ConformanceChecker(config.data.name, codes)

    points = []
    for position, sampling in enumerate(grid, start=1):
        print(f'[{position}/{len(grid)}] {sampling.temperature} / {sampling.top_p}', flush=True)
        point = _score(
            model,
            loader,
            sampling=sampling,
            seed=config.seed,
            num_samples=samples,
            codec=codec,
            codes=codes,
            index=index,
            checker=checker,
            device=torch_device,
        )
        points.append(point)
        print(
            f'  energy {point.score:.4f}  precision {point.continuation_precision:.4f}  '
            f'recall {point.continuation_recall:.4f}  emsc {point.emsc:.4f}  '
            f'conformance {point.conformance_mean:.4f}  unique {point.unique_sample_rate:.4f}',
            flush=True,
        )

    report = TuningReport.of(
        checkpoint_hash,
        search=SearchPass(pairs=len(subset), samples=samples, seed=config.seed),
        grid=points,
    )
    report.write(report_path)
    print(
        f'Chose temperature {report.chosen["temperature"]}, top_p {report.chosen["top_p"]}. '
        f'Wrote the search to {report_path}'
    )


@hydra.main(version_base='1.3', config_path='../config', config_name='tune')
def main(cfg: DictConfig) -> None:
    start_stage(cfg)
    run(
        Path(cfg.checkpoint),
        device=cfg.device,
        pairs=cfg.pairs,
        samples=cfg.samples,
        temperatures=list(cfg.temperatures),
        top_ps=list(cfg.top_ps),
        num_workers=cfg.num_workers,
    )


if __name__ == '__main__':
    main()
