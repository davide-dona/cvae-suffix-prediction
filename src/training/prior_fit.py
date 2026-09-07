"""Fit a conditional Gaussian prior against fixed posterior distributions."""

import math
from dataclasses import dataclass

import torch
from pydantic import Field, model_validator
from torch.utils.data import DataLoader, Subset

from src.configs.schema.base import NAME_PATTERN, StrictModel
from src.datasets.dataset import TraceDataset
from src.distributions import Gaussian
from src.model import TransformerCVAE
from src.training.kl import gaussian_kl


class PriorFitConfig(StrictModel):
    """The complete budget and random seed for a prior-only experiment."""

    name: str = Field(pattern=NAME_PATTERN)
    seed: int = Field(ge=0)
    device: str = Field(pattern=r'^(cpu|mps|cuda(:\d+)?)$')
    cpu_threads: int = Field(gt=0)
    training_pairs: int = Field(gt=0)
    matching_validation_pairs: int = Field(gt=0)
    generation_pairs: int = Field(gt=0)
    generation_reference: str = Field(min_length=1)
    batch_size: int = Field(gt=0)
    steps: int = Field(gt=0)
    learning_rate: float = Field(gt=0)
    weight_decay: float = Field(ge=0)
    grad_clip_norm: float = Field(gt=0)
    generation_batch_size: int = Field(gt=0)
    generation_samples: int = Field(gt=1)
    measurement_steps: tuple[int, ...]

    @model_validator(mode='after')
    def validate_measurement_steps(self) -> 'PriorFitConfig':
        """Require the full-cache measurements to cover the fixed-budget endpoints."""
        if not self.measurement_steps:
            raise ValueError('measurement_steps must not be empty.')
        if self.measurement_steps[0] != 0 or self.measurement_steps[-1] != self.steps:
            raise ValueError('measurement_steps must begin at 0 and end at steps.')
        if tuple(sorted(set(self.measurement_steps))) != self.measurement_steps:
            raise ValueError('measurement_steps must be unique and increasing.')
        return self


@dataclass(frozen=True)
class LatentCache:
    """Detached prefix summaries and Gaussian targets, held on the fitting device."""

    summaries: torch.Tensor  # [pairs, d_model]
    posterior: Gaussian  # [pairs, latent_dim] per field

    def select(self, indices: torch.Tensor) -> 'LatentCache':
        """Read a minibatch without running the encoder again."""
        return LatentCache(
            summaries=self.summaries[indices],
            posterior=Gaussian(
                mean=self.posterior.mean[indices], logvar=self.posterior.logvar[indices]
            ),
        )


def select_pairs(
    dataset: TraceDataset, *, count: int, seed: int, excluded_cases: set[str]
) -> Subset:
    """Sample eligible pairs without replacement, excluding held-out case identities.

    Raises:
        ValueError: Fewer eligible pairs exist than the requested fixed budget.
    """
    eligible = [
        i for i in range(len(dataset)) if dataset._get_cut(i)[0].case_id not in excluded_cases
    ]
    if len(eligible) < count:
        raise ValueError(f'Requested {count} pairs, but only {len(eligible)} are eligible.')
    generator = torch.Generator().manual_seed(seed)
    order = torch.randperm(n=len(eligible), generator=generator)[:count].tolist()
    return Subset(dataset=dataset, indices=[eligible[i] for i in order])


def pair_identities(subset: Subset) -> list[dict]:
    """Record dataset indices and semantic identities without materializing padded events."""
    result = []
    for index in subset.indices:
        trace, cut = subset.dataset._get_cut(index)
        result.append({'index': index, 'case_id': trace.case_id, 'prefix_len': cut})
    return result


def recorded_pairs(dataset: TraceDataset, identities: list[dict]) -> Subset:
    """Restore a recorded subset after validating its current case and prefix identities.

    Raises:
        ValueError: A recorded pair does not identify the current validation dataset.
    """
    indices = []
    for identity in identities:
        index = identity['index']
        trace, cut = dataset._get_cut(index)
        if trace.case_id != identity['case_id'] or cut != identity['prefix_len']:
            raise ValueError(
                f'Recorded pair at index {index} no longer matches the validation split.'
            )
        indices.append(index)
    if len(indices) != len(set(indices)):
        raise ValueError('Recorded generation pairs contain duplicate dataset indices.')
    return Subset(dataset=dataset, indices=indices)


@torch.no_grad()
def cache_latents(model: TransformerCVAE, loader: DataLoader, *, device: str) -> LatentCache:
    """Encode fixed targets in evaluation mode, without constructing an autograd graph."""
    model.eval()
    summaries, means, logvars = [], [], []
    for batch in loader:
        batch = batch.to(torch.device(device))
        prefix = model.encoder(events=batch.prefix, pad_mask=batch.prefix.pad_mask()).summary
        suffix = model.encoder(events=batch.suffix, pad_mask=batch.suffix.pad_mask()).summary
        posterior = model.posterior(prefix_summary=prefix, suffix_summary=suffix)
        summaries.append(prefix.detach())
        means.append(posterior.mean.detach())
        logvars.append(posterior.logvar.detach())
    return LatentCache(
        summaries=torch.cat(summaries, dim=0),
        posterior=Gaussian(mean=torch.cat(means, dim=0), logvar=torch.cat(logvars, dim=0)),
    )


def matching_loss(model: TransformerCVAE, cache: LatentCache) -> torch.Tensor:
    """Exact unfloored KL, summed across dimensions and averaged across pairs."""
    prior = model.prior(cache.summaries)
    return gaussian_kl(posterior=cache.posterior, prior=prior).sum(dim=-1).mean()


def fit_prior(
    model: TransformerCVAE,
    cache: LatentCache,
    *,
    config: PriorFitConfig,
    measurement_caches: dict[str, LatentCache],
) -> dict[str, list[dict]]:
    """Update only the prior for the fixed step budget, keeping dropout disabled.

    Raises:
        FloatingPointError: The objective or its gradients become non-finite.
    """
    model.requires_grad_(False)
    model.prior.requires_grad_(True)
    model.eval()
    optimizer = torch.optim.Adam(
        params=model.prior.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    generator = torch.Generator().manual_seed(config.seed)
    history = {'updates': [], 'measurements': []}

    def record_measurement(step: int) -> None:
        with torch.no_grad():
            values = {
                name: matching_loss(model=model, cache=measurement_cache).item()
                for name, measurement_cache in measurement_caches.items()
            }
            if not all(math.isfinite(value) for value in values.values()):
                raise FloatingPointError('Full-cache prior matching loss is not finite.')
        history['measurements'].append({'step': step, 'kl_nats': values})

    record_measurement(step=0)
    size = cache.summaries.size(dim=0)
    while len(history['updates']) < config.steps:
        order = torch.randperm(n=size, generator=generator).to(device=cache.summaries.device)
        for indices in order.split(config.batch_size):
            optimizer.zero_grad(set_to_none=True)
            loss = matching_loss(model=model, cache=cache.select(indices))
            if not torch.isfinite(loss):
                raise FloatingPointError('Prior matching loss is not finite.')
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(
                parameters=model.prior.parameters(),
                max_norm=config.grad_clip_norm,
                error_if_nonfinite=True,
            )
            optimizer.step()
            step = len(history['updates']) + 1
            history['updates'].append(
                {
                    'step': step,
                    'minibatch_kl_nats': loss.item(),
                    'grad_norm': norm.item(),
                }
            )
            if step in config.measurement_steps:
                record_measurement(step=step)
            if step == config.steps:
                break
    return history
