import math
from dataclasses import dataclass
from typing import Optional

import torch
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from src.configs import DatasetInfo
from src.configs.model_spec import ModelSpec
from src.configs.schema import (
    DataConfig,
    EarlyStoppingConfig,
    LossConfig,
    ModelConfig,
    OptimizerConfig,
    TrainingConfig,
)
from src.datasets.dataset import GenericDataset
from src.datasets.encoding import Encoding
from src.datasets.utils import move_to_device
from src.models import best_model_path_for_epoch, save_checkpoint
from src.models.architectures.SuffixCVAE.SuffixCVAE import SuffixCVAE
from src.training.annealing import cyclical_linear_weights
from src.training.early_stopping import EarlyStopper
from src.training.loss import reconstruction_loss_fn, kl_divergence_loss_fn


@dataclass
class EpochMetrics:
    """The metrics of a single pass over a dataset, already averaged over its traces."""
    loss: float
    reconstruction_loss: float
    kl_loss: float
    # The reconstruction loss split per component:
    # [cat_attributes, num_attributes, cf, ts, resource].
    reconstruction_loss_components: torch.Tensor
    # Fraction of reconstructed timestamps that are monotonically increasing.
    ts_perc_conformant: float


def _build_optimizer(model: torch.nn.Module, config: OptimizerConfig) -> optim.Optimizer:
    """Instantiate the optimizer configured by `config`."""
    return optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)


def _run_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    *,
    is_training: bool,
    w_kl: float,
    encoding: Encoding,
    dataset_info: DatasetInfo,
    device: torch.device,
    optimizer: Optional[optim.Optimizer] = None,
    grad_clip_norm: Optional[float] = None,
) -> EpochMetrics:
    """
    Run one pass over `loader`, either learning from it or just evaluating on it.

    Args:
        model: The model to train or evaluate.
        loader: The dataloader to iterate over.
        is_training: Whether to backpropagate. When False the model is put in evaluation
            mode and the gradients are not tracked.
        w_kl: The weight the KL term is given in this epoch.
        encoding: The vocabularies the batches are encoded with.
        dataset_info: The dataset information.
        device: The device to run the computations on.
        optimizer: The optimizer to step. Required when `is_training`.
        grad_clip_norm: Max gradient norm to clip to. Only applied when `is_training`.
    Returns:
        The metrics of the epoch.
    """
    if is_training and optimizer is None:
        raise ValueError('an optimizer is required to train')

    # Set the model to training or evaluation mode
    model.train(is_training)

    # Initialize accumulators for the metrics
    total_loss = 0.0
    total_reconstruction_loss = 0.0
    total_kl_loss = 0.0
    total_reconstruction_loss_components = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0])
    ts_num_conformant = 0

    # Run the epoch
    with torch.set_grad_enabled(is_training):
        for x, y in loader:
            # Move the batch to the device
            x, y = move_to_device(x, device), y.to(device)

            # Forward pass
            x_rec, mean, std = model(x, y)
            x_rec, mean, std = move_to_device(x_rec, device), move_to_device(mean, device), move_to_device(std, device)

            # Compute the reconstruction loss
            reconstruction_loss, reconstruction_loss_components = reconstruction_loss_fn(
                x=x,
                x_predicted=x_rec,
                encoding=encoding,
                dataset_info=dataset_info,
                device=device,
            )
            # Compute the KL divergence loss
            kl_loss = kl_divergence_loss_fn(mean, std)

            loss = reconstruction_loss + w_kl * kl_loss

            # Accumulate the metrics
            total_loss += loss.item()
            total_reconstruction_loss += reconstruction_loss.item()
            total_kl_loss += kl_loss.item()
            total_reconstruction_loss_components += reconstruction_loss_components

            # If training, backpropagate and update the model parameters
            if is_training:
                optimizer.zero_grad()
                loss.backward()
                if grad_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                optimizer.step()

            # compute number of traces with monotonically increasing ts
            _, _, ts_rec, _ = x_rec
            ts_num_conformant += torch.sum(ts_rec >= -0.000001).item()

    # Compute the average metrics over the dataset
    num_traces = len(loader.dataset)
    return EpochMetrics(
        loss=total_loss / num_traces,
        reconstruction_loss=total_reconstruction_loss / num_traces,
        kl_loss=total_kl_loss / num_traces,
        reconstruction_loss_components=total_reconstruction_loss_components / num_traces,
        ts_perc_conformant=ts_num_conformant / (num_traces * dataset_info.max_trace_length),
    )


def train(
    *,
    dataset_info: DatasetInfo,
    data: DataConfig,
    model_config: ModelConfig,
    loss_config: LossConfig,
    optimizer_config: OptimizerConfig,
    training: TrainingConfig,
    early_stopping_config: EarlyStoppingConfig,
    seed: int,
) -> list[dict]:
    """
    Train the model on the given dataset.

    Each argument is the sub-section of `ExperimentConfig` that this function
    actually needs, e.g. call it as
    `train(dataset_info=..., data=cfg.data, model_config=cfg.model, loss_config=cfg.loss,
    optimizer_config=cfg.optimizer, training=cfg.training,
    early_stopping_config=cfg.early_stopping, seed=cfg.seed)`.

    Returns:
        The per-epoch metrics, in the order they were recorded.
    """
    device = torch.device(training.device)

    # Datasets and dataloaders
    generator = torch.Generator().manual_seed(seed)
    encoding = Encoding(dataset_info)
    train_dataset = GenericDataset(dataset_info.train, dataset_info, encoding)
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=data.batch_size,
        shuffle=True,
        generator=generator,
        num_workers=data.num_workers,
    )
    val_dataset = GenericDataset(dataset_info.val, dataset_info, encoding)
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=data.batch_size,
        shuffle=True,
        generator=generator,
        num_workers=data.num_workers,
    )

    # Instantiate model and move it to the device
    spec = ModelSpec.build(dataset_info, model_config)
    model = SuffixCVAE(spec).to(device)

    optimizer = _build_optimizer(model, optimizer_config)

    early_stopper = EarlyStopper(
        early_stopping_config,
        full_kl_weight=loss_config.kl_annealing_full_weight,
        output_dir=training.checkpoint_dir,
    )

    current_best_val_loss = float('inf')

    # Compute the KL annealing weights for each epoch
    w_kl = cyclical_linear_weights(
        training.max_num_epochs,
        n_cycles=loss_config.kl_annealing_cycles,
        ratio=loss_config.kl_annealing_ratio,
        start=loss_config.kl_annealing_start_weight,
        stop=loss_config.kl_annealing_full_weight,
    )

    history: list[dict] = []
    last_epoch = 0

    # Training loop
    writer = SummaryWriter(log_dir=training.log_dir)
    try:
        for epoch in range(training.max_num_epochs):
            last_epoch = epoch
            train_metrics = _run_epoch(
                model,
                train_loader,
                is_training=True,
                w_kl=w_kl[epoch],
                encoding=encoding,
                dataset_info=dataset_info,
                device=device,
                optimizer=optimizer,
                grad_clip_norm=training.grad_clip_norm,
            )

            epoch_record = {
                'epoch': epoch + 1,
                'loss': train_metrics.loss,
                'reconstruction_loss': train_metrics.reconstruction_loss,
                'kl_loss': train_metrics.kl_loss,
                'w_kl': w_kl[epoch],
                # reconstruction_loss components
                'rec_cat_attributes_loss': train_metrics.reconstruction_loss_components[0].item(),
                'rec_num_attributes_loss': train_metrics.reconstruction_loss_components[1].item(),
                'rec_cf_loss': train_metrics.reconstruction_loss_components[2].item(),
                'rec_ts_loss': train_metrics.reconstruction_loss_components[3].item(),
                'rec_resource_loss': train_metrics.reconstruction_loss_components[4].item(),
                'rec_ts_perc_conformant': train_metrics.ts_perc_conformant,
            }

            should_stop = False
            if epoch % training.val_every_n_epochs == 0:
                val_metrics = _run_epoch(
                    model,
                    val_loader,
                    is_training=False,
                    w_kl=w_kl[epoch],
                    encoding=encoding,
                    dataset_info=dataset_info,
                    device=device,
                )
                epoch_record['val_loss'] = val_metrics.loss

                # only epochs evaluating the full loss function are comparable with each other
                is_new_best = (
                    math.isclose(w_kl[epoch], loss_config.kl_annealing_full_weight)
                    and val_metrics.loss < current_best_val_loss
                )
                if is_new_best:
                    current_best_val_loss = val_metrics.loss
                    best_model_path = save_checkpoint(
                        model,
                        model_spec=spec.model_dump(),
                        path=best_model_path_for_epoch(training.checkpoint_dir, epoch + 1),
                    )
                    print(f'New best model saved at {best_model_path}')

                should_stop = early_stopper.update(epoch=epoch + 1, val_loss=val_metrics.loss, w_kl=w_kl[epoch])

            for metric_name, value in epoch_record.items():
                if metric_name != 'epoch':
                    writer.add_scalar(metric_name, value, epoch + 1)

            if (epoch + 1) % training.log_every_n_steps == 0:
                print(f"[epoch {epoch + 1}/{training.max_num_epochs}] {epoch_record}")

            history.append(epoch_record)

            if should_stop:
                break
    finally:
        writer.close()

    print(f'Finished training after {last_epoch + 1} epochs')
    return history
