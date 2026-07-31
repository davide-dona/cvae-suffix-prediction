from typing import Callable, Literal, Sequence
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
from src.evaluation.sequences import sequence_similarity
from src.model import TransformerCVAE
from src.training.annealing import cyclical_linear_weight
from src.training.early_stopping import EarlyStopper
from src.training.loss import compute_loss
from src.training.metrics import GenerationMetrics, Metrics

# The three namespaces a run logs the teacher-forced metrics under, in the order TensorBoard
# should overlay them. `gen/` is not among them: it holds different quantities, not another
# pass over the same ones.
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


@torch.no_grad()
def evaluate_generation(
    model: TransformerCVAE,
    loader: DataLoader,
    *,
    num_samples: int,
    device: torch.device,
) -> GenerationMetrics:
    """
    Make the model write suffixes on its own, and score them against the ground truth.

    This is `evaluate`'s counterpart on the free-running path: no ground-truth event is handed
    back, so an early mistake is carried through the rest of the suffix, exactly as it is when
    `pipelines/predict.py` runs. It is much the more expensive of the two, a suffix costing one
    decoder pass per event rather than one per suffix, which is why the loader it is given is a
    far smaller slice of the split than `evaluate`'s.

    A generated suffix ends before its EOT and the ground truth is stored with one, so the
    truth is cut back to its content before the two are compared. A suffix cut from a truncated
    case carries no EOT to cut, and the comparison below finds none.

    Args:
        model: The model to evaluate. Put in evaluation mode here, and left in it.
        loader: The prefixes to generate for, from a `SuffixDataset`.
        num_samples: Suffixes to draw per prefix. The spread across them is what
            `sample_diversity` measures, and `generate` puts `len(batch) * num_samples` rows
            through the decoder at once, so it is also what the caller sizes its batches by.
        device: The device to run the computations on.
    Returns:
        The metrics of the pass, averaged over prefixes.
    """
    model.eval()

    dls_means: list[float] = []
    dls_bests: list[float] = []
    diversities: list[float] = []
    for batch in loader:
        batch = batch.to(device)
        generated = model.generate(batch, num_samples=num_samples)

        # The EOT closing a complete suffix is a marker rather than an event, and a generated
        # suffix stops short of its own, so it comes off the truth before the two are compared.
        last_position = (batch.suffix_len - 1).unsqueeze(dim=1)  # [batch_size, 1]
        ends_with_eot = (
            batch.suffix.activities.gather(dim=1, index=last_position).squeeze(dim=1)
            == model.eot_activity_index
        )  # [batch_size]
        true_lengths = (batch.suffix_len - ends_with_eot.long()).tolist()

        true_activities = batch.suffix.activities.cpu().tolist()
        sample_activities = generated.activities.cpu().tolist()  # [batch, num_samples, steps]
        sample_lengths = generated.lengths.cpu().tolist()

        for position, true_length in enumerate(true_lengths):
            truth = true_activities[position][:true_length]
            samples = [
                sample_activities[position][sample][: sample_lengths[position][sample]]
                for sample in range(num_samples)
            ]
            similarities = [sequence_similarity(sample, truth) for sample in samples]
            dls_means.append(sum(similarities) / len(similarities))
            dls_bests.append(max(similarities))
            diversities.append(_diversity(samples))

    return GenerationMetrics(
        activity_dls_mean=_mean(dls_means),
        activity_dls_best=_mean(dls_bests),
        sample_diversity=_mean(diversities),
    )


def _diversity(samples: Sequence[Sequence[int]]) -> float:
    """How far the suffixes generated for one prefix are from each other, in `[0, 1]`.

    The mean distance over every pair of them, which is 0.0 when a prefix's samples are all the
    same suffix. Two of them differ only in the z they were drawn with, so this is measuring
    what `p(z | prefix)` claims the prefix leaves open.
    """
    pairs = [
        1.0 - sequence_similarity(samples[first], samples[second])
        for first in range(len(samples))
        for second in range(first + 1, len(samples))
    ]
    return _mean(pairs)


def _mean(values: Sequence[float]) -> float:
    """The mean of `values`, or 0.0 if there are none."""
    return sum(values) / len(values) if values else 0.0


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

    The unit throughout is the optimizer step, not the epoch: one pass over the training split
    is 31 steps on sepsis and 863 on traffic_fines, so a budget, an annealing schedule or a
    patience denominated in epochs means a different thing on every log. Passes over the
    loader are made silently, one after another, until the step budget runs out.

    Every validation runs three passes. Two are teacher-forced: the posterior pass is the
    training objective measured on held-out data, and its KL curve is where posterior collapse
    shows up; the prior pass draws z from `p(z | prefix)` as `generate` does, and the gap
    between the two reconstruction losses (`val/prior_gap`) is how much of the suffix the
    posterior is smuggling through the latent. The third pass generates freely and scores the
    result, which is the only measurement of what the model is actually for, and so is what
    early stopping and best-model selection read. Selecting on it also makes the annealing
    weight a non-issue: a model halfway up a KL ramp generates worse and simply does not win.

    Every number a run produces goes to TensorBoard, which is the record of it; nothing is
    returned. `train/*` is logged every step, raw and unaveraged, so instability is visible at
    the step it happens rather than smoothed into whatever interval validation runs on;
    `val/*`, `val_prior/*` and `gen/*` are logged only at validation. The console gets one line
    per validation, so a run can be watched without opening TensorBoard, plus the events
    TensorBoard cannot express; its train figure is the interval average; TensorBoard has the
    per-step one.

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

    current_best_score = float('inf')
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
                    # The same objective with z drawn the way inference draws it, so the two
                    # differ only in where the latent came from.
                    val_prior_metrics = evaluate(
                        model, val_loader, kl_weight=kl_weight,
                        free_bits=loss_config.free_bits, sample_from='prior', device=device,
                    )
                    # What the model is for, and what it is judged by.
                    gen_metrics = evaluate_generation(
                        model, generation_loader,
                        num_samples=generation_samples, device=device,
                    )
                    # Higher similarity is a better model, and everything downstream of here
                    # takes a loss, so the score is turned round once, here.
                    selection_score = 1.0 - gen_metrics.activity_dls_mean

                    val_metrics.log(writer, step, prefix='val')
                    val_prior_metrics.log(writer, step, prefix='val_prior')
                    gen_metrics.log(writer, step)
                    # How much of the suffix the posterior is carrying that the prior cannot
                    # reproduce: the reconstruction terms alone, since the totals also differ
                    # by a KL term only one of them has.
                    writer.add_scalar(
                        'val/prior_gap',
                        val_prior_metrics.reconstruction_loss - val_metrics.reconstruction_loss,
                        step,
                    )
                    writer.add_scalar('kl_weight', kl_weight, step)

                    # The one line of live feedback: enough to see a run is alive and heading
                    # down, while TensorBoard stays the place the numbers are actually read.
                    # Printed before the best-model check so that its message lands under the
                    # step it belongs to.
                    print(
                        f'Step {step:>{step_width}}/{training.max_steps}  '
                        f'kl {kl_weight:.2f}  train {train_metrics.loss:.4f}  '
                        f'val {val_metrics.loss:.4f}  '
                        f'val_prior {val_prior_metrics.loss:.4f}  '
                        f'gen_dls {gen_metrics.activity_dls_mean:.4f}  '
                        f'diversity {gen_metrics.sample_diversity:.4f}',
                        flush=True,
                    )

                    if selection_score < current_best_score:
                        current_best_score = selection_score
                        on_best_step(step, selection_score)

                    should_stop = early_stopper.update(selection_score)

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
