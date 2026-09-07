"""Semantic checks at dataset, model, and training boundaries."""

import math
import re

from omegaconf import DictConfig

from src.logs.keys import CYCLE_TIME_KEY


def _number(
    value: object, name: str, *, minimum: float = 0, inclusive: bool = False, integer: bool = False
) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, int if integer else (int, float))
        or not math.isfinite(value)
        or (value < minimum if inclusive else value <= minimum)
    ):
        relation = '>=' if inclusive else '>'
        raise ValueError(
            f'{name} must be a finite {"integer" if integer else "number"} '
            f'{relation} {minimum}, got {value!r}'
        )


def _name(value: str, name: str) -> None:
    if not isinstance(value, str) or not re.fullmatch(r'[a-z0-9][a-z0-9-]*', value):
        raise ValueError(f'{name} must contain only lowercase letters, digits, and hyphens')


def validate_data(data: DictConfig) -> None:
    _name(data.name, 'data.name')
    splits = [data.train_split, data.val_split, data.test_split]
    for value in splits:
        _number(value, 'split fraction')
    if abs(sum(splits) - 1) > 1e-6:
        raise ValueError('train/val/test splits must sum to 1')
    for key in ('max_seq_len_percentile', 'max_case_duration_percentile'):
        _number(data[key], f'data.{key}')
        if data[key] > 100:
            raise ValueError(f'data.{key} must not exceed 100')
    if not set(data.log_scaled_features) <= set(data.event_features):
        raise ValueError('log_scaled_features must be event_features')
    if CYCLE_TIME_KEY in data.event_features:
        raise ValueError('cycle_time must not also be an event_feature')
    if not isinstance(data.separator, str) or not data.separator:
        raise ValueError('data.separator must be a nonempty string')


def validate_declare(config: DictConfig) -> None:
    for key in ('min_support', 'itemsets_support'):
        _number(config[key], f'declare.{key}')
        if config[key] > 1:
            raise ValueError(f'declare.{key} must not exceed 1')
    _number(config.max_cardinality, 'declare.max_cardinality', integer=True)
    if not isinstance(config.consider_vacuity, bool):
        raise ValueError('declare.consider_vacuity must be boolean')


def validate_sampling(config: DictConfig) -> None:
    if set(config) != {'temperature', 'top_p'}:
        raise ValueError('sampling must contain exactly temperature and top_p')
    _number(config.temperature, 'sampling.temperature')
    _number(config.top_p, 'sampling.top_p')
    if config.top_p > 1:
        raise ValueError('sampling.top_p must not exceed 1')


def validate_model(model: DictConfig) -> None:
    _name(model.name, 'model.name')
    if model.kind not in ('cvae', 'transformer'):
        raise ValueError(f'Unknown model kind: {model.kind}')
    _number(model.d_model, 'model.d_model', integer=True)
    for key, value in model.embeddings.items():
        _number(value, f'model.embeddings.{key}', integer=True)
    for name in ('encoder', 'decoder'):
        section = model[name]
        for key in ('num_layers', 'num_heads', 'feedforward_dim'):
            _number(section[key], f'model.{name}.{key}', integer=True)
        if model.d_model % section.num_heads:
            raise ValueError(f'model.{name}.num_heads must divide model.d_model')
        for key in ('dropout', 'activity_dropout'):
            if key in section:
                _number(section[key], f'model.{name}.{key}', inclusive=True)
                if section[key] >= 1:
                    raise ValueError(f'model.{name}.{key} must be below 1')
    _number(model.decoder.head_hidden_dim, 'model.decoder.head_hidden_dim', integer=True)
    if model.kind == 'transformer':
        if any(key in model for key in ('prior', 'latent', 'loss')):
            raise ValueError('transformer does not accept prior, latent, or loss settings')
        validate_sampling(model.sampling)
    else:
        if 'sampling' in model:
            raise ValueError('cvae does not accept sampling settings')
        _number(model.latent.latent_dim, 'model.latent.latent_dim', integer=True)
        for width in model.prior.hidden_dims:
            _number(width, 'model.prior.hidden_dims', integer=True)
        _number(model.prior.dropout, 'model.prior.dropout', inclusive=True)
        if model.prior.dropout >= 1:
            raise ValueError('model.prior.dropout must be below 1')
        _number(model.loss.kl_annealing_ramp_steps, 'kl_annealing_ramp_steps', integer=True)
        for key in ('kl_annealing_start_weight', 'kl_annealing_full_weight', 'free_bits'):
            _number(model.loss[key], f'model.loss.{key}', inclusive=True)


def validate_training(config: DictConfig) -> None:
    validate_data(config.data)
    validate_declare(config.declare)
    validate_model(config.model)
    _number(config.seed, 'seed', inclusive=True, integer=True)
    for key in ('max_steps', 'val_every_n_steps', 'validation_pairs', 'generation_pairs'):
        _number(config.training[key], f'training.{key}', integer=True)
    if config.training.grad_clip_norm is not None:
        _number(config.training.grad_clip_norm, 'training.grad_clip_norm')
    if not re.fullmatch(r'(cpu|mps|cuda(:\d+)?)', config.training.device):
        raise ValueError('training.device must be cpu, mps, cuda, or cuda:<index>')
    _number(config.dataloader.batch_size, 'dataloader.batch_size', integer=True)
    _number(config.dataloader.num_workers, 'dataloader.num_workers', inclusive=True, integer=True)
    _number(config.optimizer.lr, 'optimizer.lr')
    _number(config.optimizer.weight_decay, 'optimizer.weight_decay', inclusive=True)
    _number(config.optimizer.warmup_steps, 'optimizer.warmup_steps', inclusive=True, integer=True)
    _number(config.early_stopping.patience, 'early_stopping.patience', integer=True)
    _number(config.early_stopping.min_delta_perc, 'early_stopping.min_delta_perc', inclusive=True)
    for key in ('validation_samples', 'evaluation_samples'):
        _number(config.inference[key], f'inference.{key}', minimum=10, inclusive=True, integer=True)
    _number(
        config.inference.generation_rows_upper_bound, 'generation_rows_upper_bound', integer=True
    )
    if config.inference.generation_rows_upper_bound < max(
        config.inference.validation_samples, config.inference.evaluation_samples
    ):
        raise ValueError('generation_rows_upper_bound must fit at least one prefix of samples')
    if config.wandb.mode not in ('online', 'offline', 'disabled'):
        raise ValueError('wandb.mode must be online, offline, or disabled')
