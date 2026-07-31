from typing import Callable, Literal
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
from src.training.annealing import cyclical_linear_weight
from src.training.early_stopping import EarlyStopper
from src.training.loss import compute_loss
from src.training.metrics import Metrics

# The three namespaces a run logs under, in the order TensorBoard should overlay them.
LOG_PREFIXES = ('train', 'val', 'val_prior')


@torch.no_grad()
def evaluate(
    model: TransformerCVAE,
    loader: DataLoader,
    *,
    kl_weight: float,
    free_bits: float,
    sample_from: Literal['posterior', 'prior'],
    device: torch.device,
) -> Metrics:
    """
    Run one pass over `loader` without learning from it.

    Args:
        model: The model to evaluate. Put in evaluation mode here, and left in it.
        loader: The dataloader to iterate over. Its batches are `SuffixItem`s.
        kl_weight: The weight this step's KL term is given. Ignored on the prior path, which
            has no KL term at all.
        free_bits: Passed straight through to `compute_loss`.
        sample_from: Which distribution z is drawn from. `'posterior'` reproduces the
            training objective on held-out data; `'prior'` scores the path `generate` runs on
            and is the one a run is judged by.
        device: The device to run the computations on.
    Returns:
        The metrics of the pass, averaged over the traces of the split.
    """
    model.eval()

    totals = Metrics()
    for batch in loader:
        batch = batch.to(device)
        _, metrics = compute_loss(
            model, batch, kl_weight=kl_weight, free_bits=free_bits, sample_from=sample_from
        )
        totals += metrics

    return totals / len(loader.dataset)


def train(
    *,
    model: TransformerCVAE,
    train_loader: DataLoader,
    val_loader: DataLoader,
    run_name: str,
    on_best_step: Callable[[int, float], None],
    loss_config: LossConfig,
    optimizer_config: OptimizerConfig,
    training: TrainingConfig,
    early_stopping_config: EarlyStoppingConfig,
) -> None:
    """
    Train a model on a dataset, logging to TensorBoard and saving checkpoints.

    The unit throughout is the optimizer step, not the epoch: one pass over the training split
    is 31 steps on sepsis and 863 on traffic_fines, so a budget, an annealing schedule or a
    patience denominated in epochs means a different thing on every log. Passes over the
    loader are made silently, one after another, until the step budget runs out.

    Every validation runs the split twice. The posterior pass is the training objective
    measured on held-out data, and its KL curve is where posterior collapse shows up. The
    prior pass draws z from `p(z | prefix)`, which is what `generate` does, and so is the only
    number that reflects how the model will actually be used; it also carries no KL term,
    which is what makes it comparable across a run whatever the annealing weight is doing.
    Early stopping and best-model selection therefore read that one.

    Every number a run produces goes to TensorBoard, which is the record of it; nothing is
    returned. `train/*` is logged every step, raw and unaveraged, so instability is visible at
    the step it happens rather than smoothed into whatever interval validation runs on;
    `val/*` and `val_prior/*` are logged only at validation. The console gets one line per
    validation, so a run can be watched without opening TensorBoard, plus the events
    TensorBoard cannot express; its train figure is the interval average; TensorBoard has the
    per-step one.

    Args:
        model: The model to train, already on `training.device`.
        train_loader: Batches to learn from.
        val_loader: Batches to evaluate on, every `training.val_every_n_steps` steps.
        run_name: Subdirectory of `training.log_dir` this run writes its events to. One
            directory is one TensorBoard run, so a name reused across runs overlays their
            curves instead of listing them side by side; what makes it unique is the
            caller's business. A `/` in it nests the run, which is how TensorBoard groups
            runs under a common prefix.
        on_best_step: Called with the step number and its prior-path validation loss whenever
            a validation improves on the best so far, so that saving a checkpoint stays the
            caller's business.
        loss_config: The KL annealing schedule and the free-bits floor.
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

    current_best_val_loss = float('inf')
    step = 0
    should_stop = False
    step_width = len(str(training.max_steps))

    # Accumulated between validations rather than over a pass, since a pass is no longer the
    # unit anything is reported at. `seen` is what they are averaged by: the interval is a
    # count of batches, not a whole split, and batches are not equal-sized.
    interval_totals = Metrics()
    seen = 0

    writer = SummaryWriter(log_dir=training.log_dir / run_name)
    Metrics.log_layout(writer, prefixes=LOG_PREFIXES)
    print(f'Logging to {writer.log_dir}')
    try:
        while step < training.max_steps and not should_stop:
            # Another pass over the training split. The loader reshuffles on each one, and
            # nothing is reported at its boundaries.
            for batch in train_loader:
                model.train()
                batch = batch.to(device)
                kl_weight = cyclical_linear_weight(
                    step,
                    period_steps=loss_config.kl_annealing_period_steps,
                    ratio=loss_config.kl_annealing_ratio,
                    start=loss_config.kl_annealing_start_weight,
                    stop=loss_config.kl_annealing_full_weight,
                )

                loss, metrics = compute_loss(
                    model, batch, kl_weight=kl_weight, free_bits=loss_config.free_bits
                )
                optimizer.zero_grad()
                loss.backward()
                if training.grad_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), training.grad_clip_norm)
                optimizer.step()

                batch_size = batch.suffix.activities.size(0)
                interval_totals += metrics
                seen += batch_size
                step += 1

                # Logged every step and unaveraged beyond the batch itself, so instability
                # shows up immediately instead of being smoothed into the interval average
                # below, which stays for the console line only.
                (metrics / batch_size).log(writer, step, prefix='train')

                if step % training.val_every_n_steps == 0 or step >= training.max_steps:
                    train_metrics = interval_totals / seen
                    interval_totals, seen = Metrics(), 0

                    # The objective, on held-out data: comparable to `train` above, and the
                    # place the KL is visible at all.
                    val_metrics = evaluate(
                        model, val_loader, kl_weight=kl_weight,
                        free_bits=loss_config.free_bits, sample_from='posterior', device=device,
                    )
                    # The path inference runs on, and the one a run is judged by.
                    val_prior_metrics = evaluate(
                        model, val_loader, kl_weight=kl_weight,
                        free_bits=loss_config.free_bits, sample_from='prior', device=device,
                    )

                    val_metrics.log(writer, step, prefix='val')
                    val_prior_metrics.log(writer, step, prefix='val_prior')
                    writer.add_scalar('kl_weight', kl_weight, step)

                    # The one line of live feedback: enough to see a run is alive and heading
                    # down, while TensorBoard stays the place the numbers are actually read.
                    # Printed before the best-model check so that its message lands under the
                    # step it belongs to.
                    print(
                        f'Step {step:>{step_width}}/{training.max_steps}  '
                        f'kl {kl_weight:.2f}  train {train_metrics.loss:.4f}  '
                        f'val {val_metrics.loss:.4f}  '
                        f'val_prior {val_prior_metrics.loss:.4f}',
                        flush=True,
                    )

                    if val_prior_metrics.loss < current_best_val_loss:
                        current_best_val_loss = val_prior_metrics.loss
                        on_best_step(step, val_prior_metrics.loss)

                    should_stop = early_stopper.update(val_prior_metrics.loss)

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
