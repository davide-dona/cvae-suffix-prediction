from typing import Callable, Optional
import numpy as np
import torch
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from src.configs.schema import (
    EarlyStoppingConfig,
    LossConfig,
    OptimizerConfig,
    TrainingConfig,
)
from src.datasets.utils import move_to_device
from src.model import AttentionCVAE
from src.training.annealing import cyclical_linear_weights
from src.training.early_stopping import EarlyStopper
from src.training.loss import compute_loss
from src.training.metrics import Metrics


def run_epoch(
    model: AttentionCVAE,
    loader: DataLoader,
    *,
    kl_weight: float,
    is_training: bool,
    device: torch.device,
    optimizer: Optional[optim.Optimizer] = None,
    grad_clip_norm: Optional[float] = None,
) -> Metrics:
    """
    Run one pass over `loader`, either learning from it or just evaluating on it.

    Args:
        model: The model to train or evaluate.
        loader: The dataloader to iterate over. Its batches are dicts of tensors.
        kl_weight: The weight this epoch's KL term is given, passed straight through to
            `compute_loss`.
        is_training: Whether to backpropagate. When False the model is put in evaluation
            mode and the gradients are not tracked.
        device: The device to run the computations on.
        optimizer: The optimizer to step. Required when `is_training`.
        grad_clip_norm: Max gradient norm to clip to. Only applied when `is_training`.
    Returns:
        The metrics of the pass, averaged over the traces of the split.
    """
    if is_training and optimizer is None:
        raise ValueError('an optimizer is required to train')

    model.train(is_training)

    totals = Metrics()
    with torch.set_grad_enabled(is_training):
        for batch in loader:
            batch = move_to_device(batch, device)
            loss, metrics = compute_loss(model, batch, kl_weight)
            totals += metrics

            # Backpropagate and step the optimizer if training
            if is_training:
                optimizer.zero_grad()
                loss.backward()
                if grad_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                optimizer.step()

    return totals / len(loader.dataset)


def train(
    *,
    model: AttentionCVAE,
    train_loader: DataLoader,
    val_loader: DataLoader,
    on_best_epoch: Callable[[int, float], None],
    loss_config: LossConfig,
    optimizer_config: OptimizerConfig,
    training: TrainingConfig,
    early_stopping_config: EarlyStoppingConfig,
) -> None:
    """
    Train a model on a dataset, logging to TensorBoard and saving checkpoints.

    Every number a run produces goes to TensorBoard, which is the record of it; nothing is
    returned or printed except the events TensorBoard cannot express.

    Args:
        model: The model to train, already on `training.device`.
        train_loader: Batches to learn from.
        val_loader: Batches to evaluate on, every `training.val_every_n_epochs` epochs.
        on_best_epoch: Called with the epoch number and its validation loss whenever an
            epoch improves on the best so far, so that saving a checkpoint stays the
            caller's business.
        loss_config: The KL annealing schedule.
        optimizer_config: The optimizer hyperparameters.
        training: Epoch count, gradient clipping, device and the TensorBoard destination.
        early_stopping_config: When to give up.
    """
    device = torch.device(training.device)
    optimizer = optim.Adam(
        model.parameters(), lr=optimizer_config.lr, weight_decay=optimizer_config.weight_decay
    )
    early_stopper = EarlyStopper(early_stopping_config)

    kl_weights = cyclical_linear_weights(
        training.max_num_epochs,
        n_cycles=loss_config.kl_annealing_cycles,
        ratio=loss_config.kl_annealing_ratio,
        start=loss_config.kl_annealing_start_weight,
        stop=loss_config.kl_annealing_full_weight,
    )
    # While the weight is still ramping, a lower one makes an epoch look better for free, so
    # only epochs that reached the full weight may be compared for early stopping and
    # best-model selection.
    is_comparable = np.isclose(kl_weights, loss_config.kl_annealing_full_weight)

    current_best_val_loss = float('inf')
    last_epoch = 0

    writer = SummaryWriter(log_dir=training.log_dir)
    try:
        for epoch in range(training.max_num_epochs):
            last_epoch = epoch
            epoch_number = epoch + 1
            epoch_kl_weight = float(kl_weights[epoch])
            epoch_is_comparable = bool(is_comparable[epoch])

            train_metrics = run_epoch(
                model,
                train_loader,
                kl_weight=epoch_kl_weight,
                is_training=True,
                device=device,
                optimizer=optimizer,
                grad_clip_norm=training.grad_clip_norm,
            )
            # Log the metrics to TensorBoard
            train_metrics.log(writer, epoch_number, prefix='train')
            writer.add_scalar('kl_weight', epoch_kl_weight, epoch_number)

            should_stop = False
            if epoch % training.val_every_n_epochs == 0:
                val_metrics = run_epoch(
                    model,
                    val_loader,
                    kl_weight=epoch_kl_weight,
                    is_training=False,
                    device=device,
                )
                val_metrics.log(writer, epoch_number, prefix='val')

                # Only epochs evaluating the full loss function are comparable with each other,
                # since a lower weight makes an epoch look better for free.
                val_loss = val_metrics.loss
                is_new_best = epoch_is_comparable and val_loss < current_best_val_loss
                if is_new_best:
                    current_best_val_loss = val_loss
                    on_best_epoch(epoch_number, val_loss)

                should_stop = early_stopper.update(val_loss, epoch_is_comparable)

            if should_stop:
                break
    finally:
        writer.close()

    reason = (
        f'no validation improvement for {early_stopping_config.patience} epochs'
        if should_stop
        else 'reached max_num_epochs'
    )
    print(f'Finished training after {last_epoch + 1} epochs ({reason})')
