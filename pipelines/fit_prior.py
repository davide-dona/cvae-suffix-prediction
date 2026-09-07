"""Compare a frozen CVAE with its Gaussian prior fitted to cached training posteriors."""

import argparse
import hashlib
import json
import math
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import torch
import yaml
from torch.utils.data import DataLoader

from src import paths
from src.configs.loader import load_dataset_config
from src.configs.schema import ExperimentConfig
from src.datasets.codec import DatasetCodec
from src.datasets.dataset import TraceDataset
from src.evaluation.summary import EvaluationSummary, PrefixSummary, flatten_scores
from src.identity import RunIdentity, stamped
from src.inference.generate import generate_batch
from src.logs import ContinuationIndex, Split
from src.logs.declare import ConformanceChecker
from src.model import TransformerCVAE, load_checkpoint, model_from_checkpoint
from src.suffixes import ActivityCodes
from src.training.prior_fit import (
    PriorFitConfig,
    cache_latents,
    fit_prior,
    matching_loss,
    pair_identities,
    select_pairs,
)


def _finite(value):
    if isinstance(value, dict):
        return {key: _finite(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_finite(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(_finite(value), indent=2, allow_nan=False) + '\n')


def pair_scores(original: list[dict], fitted: list[dict]) -> list[dict]:
    """Join the two arms by their checked row identities.

    Raises:
        ValueError: Row counts or prefix identities differ between arms.
    """
    paired = []
    for left, right in zip(original, fitted, strict=True):
        identity = {key: left[key] for key in ('case_id', 'prefix_len', 'suffix_len', 'comparable')}
        if any(right[key] != value for key, value in identity.items()):
            raise ValueError('Original and fitted score rows are not paired.')
        paired.append(
            identity
            | {f'original/{key}': value for key, value in left.items() if key not in identity}
            | {f'fitted/{key}': value for key, value in right.items() if key not in identity}
        )
    return paired


@torch.inference_mode()
def score_prior(
    model: TransformerCVAE,
    loader: DataLoader,
    *,
    config: PriorFitConfig,
    codec: DatasetCodec,
    index: ContinuationIndex,
    checker: ConformanceChecker,
) -> tuple[dict, list[dict]]:
    """Score one arm with the same ordering and latent noise seed as its paired arm."""
    model.eval()
    torch.manual_seed(config.seed)
    codes = ActivityCodes.of(codec.activity.names)
    summaries, rows = [], []
    for batch in loader:
        generations = generate_batch(
            model=model,
            batch=batch.to(torch.device(config.device)),
            num_samples=config.generation_samples,
            codec=codec,
            codes=codes,
        )
        for generation in generations:
            summary = PrefixSummary.of(generation, checker=checker, index=index)
            summaries.append(summary)
            rows.append(
                {
                    'case_id': generation.case_id,
                    'prefix_len': summary.prefix_len,
                    'suffix_len': summary.suffix_len,
                    'comparable': summary.distribution.comparable,
                    **flatten_scores(summary),
                }
            )
    return asdict(EvaluationSummary.of(summaries)), rows


def run(checkpoint_path: Path, dataset_path: Path, config_path: Path, output: Path) -> Path:
    """Fit and compare two priors, writing a separate checkpoint and paired diagnostic artifacts.

    Raises:
        ValueError: The checkpoint is not a CVAE or names a different dataset.
        FileExistsError: The output directory already exists.
    """
    config = PriorFitConfig.model_validate(yaml.safe_load(config_path.read_text()))
    dataset_config = load_dataset_config(dataset_path)
    checkpoint = load_checkpoint(checkpoint_path)
    source_run = RunIdentity.from_dict(checkpoint['run'])
    if source_run.dataset != dataset_config.data.name:
        raise ValueError('Checkpoint and dataset config must name the same dataset.')
    if checkpoint['model_config']['kind'] != 'cvae':
        raise ValueError('Prior fitting requires a Gaussian CVAE checkpoint.')
    if output.exists():
        raise FileExistsError(f'Choose a fresh output directory: {output}')
    for split in (Split.TRAIN, Split.VAL):
        paths.PROCESSED_SPLIT.require(dataset=source_run.dataset, split=split)
    paths.CONTINUATIONS.require(dataset=source_run.dataset, split=Split.VAL)
    paths.DECLARE_MODEL.require(source_run.dataset)
    torch.set_num_threads(config.cpu_threads)
    torch.manual_seed(config.seed)
    codec = DatasetCodec.load(dataset_config.data)
    model = model_from_checkpoint(checkpoint=checkpoint, codec=codec, device=config.device)
    model.requires_grad_(False)
    run_identity = RunIdentity(
        dataset=source_run.dataset,
        model=config.name,
        tag=datetime.now(UTC).strftime('%Y%m%d-%H%M%S'),
    )
    resolved = ExperimentConfig.model_validate(
        {
            **checkpoint['experiment_config'],
            **dataset_config.model_dump(),
            'model': {**checkpoint['model_config'], 'name': config.name},
            'training': {**checkpoint['experiment_config']['training'], 'device': config.device},
        }
    )
    print('Selecting training and validation pairs', flush=True)
    validation = select_pairs(
        TraceDataset(codec=codec, split=Split.VAL),
        count=config.validation_pairs,
        seed=config.seed,
        excluded_cases=set(),
    )
    validation_ids = pair_identities(validation)
    training = select_pairs(
        TraceDataset(codec=codec, split=Split.TRAIN),
        count=config.training_pairs,
        seed=config.seed,
        excluded_cases={row['case_id'] for row in validation_ids},
    )
    training_ids = pair_identities(training)
    print('Caching frozen encoder and posterior outputs', flush=True)
    train_cache = cache_latents(
        model=model,
        loader=DataLoader(dataset=training, batch_size=config.batch_size, num_workers=0),
        device=config.device,
    )
    val_cache = cache_latents(
        model=model,
        loader=DataLoader(dataset=validation, batch_size=config.batch_size, num_workers=0),
        device=config.device,
    )
    loader = DataLoader(dataset=validation, batch_size=config.generation_batch_size, num_workers=0)
    index = ContinuationIndex.read(dataset=source_run.dataset, split=Split.VAL)
    checker = ConformanceChecker(
        dataset=source_run.dataset, codes=ActivityCodes.of(codec.activity.names)
    )
    with torch.no_grad():
        before = {
            'train': matching_loss(model=model, cache=train_cache).item(),
            'validation': matching_loss(model=model, cache=val_cache).item(),
        }
    print('Scoring original prior', flush=True)
    original, original_rows = score_prior(
        model=model, loader=loader, config=config, codec=codec, index=index, checker=checker
    )
    print(f'Fitting prior for {config.steps} updates', flush=True)
    history = fit_prior(model=model, cache=train_cache, config=config)
    for name, value in model.state_dict().items():
        if not name.startswith('prior.') and not torch.equal(
            value.cpu(), checkpoint['model_state_dict'][name]
        ):
            raise RuntimeError(f'Frozen state changed: {name}')
    with torch.no_grad():
        after = {
            'train': matching_loss(model=model, cache=train_cache).item(),
            'validation': matching_loss(model=model, cache=val_cache).item(),
        }
    print('Scoring fitted prior', flush=True)
    fitted, fitted_rows = score_prior(
        model=model, loader=loader, config=config, codec=codec, index=index, checker=checker
    )
    provenance = {
        'source_run': asdict(source_run),
        'source_checkpoint': str(checkpoint_path.resolve()),
        'source_sha256': hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
        'source_training_step': checkpoint['step'],
        'source_selection_score': checkpoint['selection_score'],
        'prior_fitting_steps': config.steps,
        'checkpoint_policy': 'Final fixed-budget prior; no validation selection',
        'fit_config': config.model_dump(),
        'dataset_config': dataset_config.model_dump(),
        'resolved_model_experiment_config': resolved.model_dump(),
        'subsets': {'train': training_ids, 'validation': validation_ids},
    }
    output.mkdir(parents=True, exist_ok=False)
    fitted_checkpoint = {
        'model_config': resolved.model.model_dump(),
        'model_state_dict': {name: value.cpu() for name, value in model.state_dict().items()},
        'experiment_config': resolved.model_dump(),
        'run': asdict(run_identity),
        'step': checkpoint['step'],
        # The final all-prefix EMSC distance is diagnostic; no checkpoint search is performed.
        'selection_score': 1.0 - sum(row['emsc'] for row in fitted_rows) / len(fitted_rows),
        'prior_fit': provenance,
    }
    torch.save(obj=fitted_checkpoint, f=output / 'fitted.pt')
    paired = pair_scores(original=original_rows, fitted=fitted_rows)
    table = pa.Table.from_pylist(paired)
    table = table.replace_schema_metadata(stamped(table.schema, run_identity).metadata)
    pq.write_table(table=table, where=output / 'paired_scores.parquet')
    _write_json(output / 'history.json', {'run': asdict(run_identity), 'updates': history})
    _write_json(
        output / 'comparison.json',
        {
            'run': asdict(run_identity),
            **provenance,
            'kl_nats': {'original': before, 'fitted': after},
            'scores': {'original': original, 'fitted': fitted},
            'distribution_population': (
                'Validation prefixes with at least five reference occurrences'
            ),
            'unavailable_metrics': (
                'Non-finite scores are null; consult compared population counts.'
            ),
        },
    )
    print(f'Comparison: {output / "comparison.json"}', flush=True)
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-m', '--checkpoint', required=True, type=Path)
    parser.add_argument('-c', '--dataset-config', required=True, type=Path)
    parser.add_argument('-e', '--experiment-config', required=True, type=Path)
    parser.add_argument('-o', '--output', required=True, type=Path)
    args = parser.parse_args()
    run(
        checkpoint_path=args.checkpoint,
        dataset_path=args.dataset_config,
        config_path=args.experiment_config,
        output=args.output,
    )
