from typing import Callable
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
from src.model import TransformerCVAE
from src.training.early_stopping import EarlyStopper
from src.training.kl import cyclical_linear_weight
from src.training.loss import Loss, compute_loss
from src.training.validation import GenerationMetrics, validate, validate_generation


def train(
    *,
    model: TransformerCVAE,
    train_loader: DataLoader,
    val_loader: DataLoader,
    generation_loader: DataLoader,
    generation_samples: int,
    run_name: str,
    on_best_step: Callable[[int, float], None],
    loss_config: LossConfig,
    optimizer_config: OptimizerConfig,
    training: TrainingConfig,
    early_stopping_config: EarlyStoppingConfig,
) -> None:
    """
    Train a model on a dataset, logging to TensorBoard and saving checkpoints.
    Args:
        model: The model to train, already on `training.device`.
        train_loader: Batches to learn from.
        val_loader: Batches to score teacher-forced, every `training.val_every_n_steps` steps.
        generation_loader: Prefixes to generate suffixes for on the same cadence. A far smaller
            slice than `val_loader`, since a suffix costs one decoder pass per event.
        generation_samples: Suffixes to draw per prefix on that pass, normally
            `inference.num_samples`, so that a training curve and the final report describe the
            same number of draws.
        run_name: Subdirectory of `training.log_dir` this run writes its events to. One
            directory is one TensorBoard run, so a name reused across runs overlays their
            curves instead of listing them side by side; what makes it unique is the
            caller's business. A `/` in it nests the run, which is how TensorBoard groups
            runs under a common prefix.
        on_best_step: Called with the step number and its selection score whenever a validation
            improves on the best so far, so that saving a checkpoint stays the caller's
            business.
        loss_config: The KL annealing schedule.
        optimizer_config: The optimizer hyperparameters.
        training: Step budget, validation cadence, gradient clipping, device and the
            TensorBoard destination.
        early_stopping_config: When to give up.
    """
    device = torch.device(training.device)
    
    optimizer = optim.Adam(
        model.parameters(), lr=optimizer_config.lr, weight_decay=optimizer_config.weight_decay
    )
    early_stopper = EarlyStopper(early_stopping_config)

    step = 0
    should_stop = False
    interval_totals = Loss()
    seen = 0

    writer = SummaryWriter(log_dir=training.log_dir / run_name)
    print(f'Logging to {writer.log_dir}')
    try:
        while step < training.max_steps and not should_stop:
            for batch in train_loader:
                model.train()
                batch = batch.to(device)
                # Get the current KL weight for this step
                kl_weight = cyclical_linear_weight(
                    step,
                    period_steps=loss_config.kl_annealing_period_steps,
                    ratio=loss_config.kl_annealing_ratio,
                    start=loss_config.kl_annealing_start_weight,
                    stop=loss_config.kl_annealing_full_weight,
                )
                # Run a forward pass
                output = model(batch)
                
                # Compute the loss and propagate gradients
                loss, metrics = compute_loss(
                    output, batch,
                    pad_activity_index=model.pad_activity_index,
                    kl_weight=kl_weight,
                    free_bits=loss_config.free_bits,
                )
                optimizer.zero_grad()
                loss.backward()
                if training.grad_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), training.grad_clip_norm)
                optimizer.step()

                # Update the running totals and log to TensorBoard
                batch_size = batch.suffix.activities.size(0)
                interval_totals += metrics
                seen += batch_size
                step += 1
                (metrics / batch_size).log(writer, step, prefix='train')

                if step % training.val_every_n_steps == 0 or step >= training.max_steps:
                    train_metrics = interval_totals / seen
                    interval_totals, seen = Loss(), 0

                    # Score the model on the validation set and the generation set, and log the results.
                    val_metrics = validate(
                        model, val_loader, kl_weight=kl_weight, free_bits=loss_config.free_bits, device=device
                    )
                    val_metrics.log(writer, step, prefix='val')
                    
                    gen_metrics = validate_generation(
                        model, generation_loader,
                        num_samples=generation_samples, device=device,
                    )
                    gen_metrics.log(writer, step, prefix='gen')

                    writer.add_scalar('kl_weight', kl_weight, step)

                    # The one line of live feedback: enough to see a run is alive and heading down
                    print(
                        f'Step {step:>{len(str(training.max_steps))}}/{training.max_steps}  '
                        f'kl {kl_weight:.2f}  train {train_metrics.loss:.4f}  '
                        f'val {val_metrics.loss:.4f}  '
                        f'gen_dls {gen_metrics.activity_dls_mean:.4f}  '
                        f'energy {gen_metrics.activity_energy_score:.4f}',
                        flush=True,
                    )
                    selection_score = gen_metrics.activity_energy_score

                    # `early_stopper` already tracks the best score seen for its own patience
                    # count, so reading it here rather than keeping a second one is what a
                    # validation improving on it means.
                    is_best = selection_score < early_stopper.min_validation_score
                    should_stop = early_stopper.update(selection_score)
                    if is_best:
                        on_best_step(step, selection_score)

                if should_stop or step >= training.max_steps:
                    break
    finally:
        writer.close()

    reason = (
        f'no validation improvement for {early_stopping_config.patience} validations'
        if should_stop
        else 'reached max_steps'
    )
    print(f'Finished training after {step} steps ({reason})')
