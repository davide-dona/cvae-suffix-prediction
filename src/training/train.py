from __future__ import annotations
from omegaconf import DictConfig, OmegaConf

from typing import TYPE_CHECKING

import torch
import wandb
from torch import optim
from torch.utils.data import DataLoader

from src import paths
from src.datasets.codec import DatasetCodec
from src.identity import WANDB_PROJECT, RunIdentity, experiment, wandb_artifact, wandb_id
from src.logs import ContinuationIndex, Split
from src.logs.declare import ConformanceChecker
from src.suffixes import ActivityCodes
from src.training.early_stopping import EarlyStopper
from src.training.loss import Loss
from src.training.validation import validate, validate_generation

if TYPE_CHECKING:
    from src.model import SuffixModel


def _lr_factor(step: int, *, warmup_steps: int) -> float:
    """What `optimizer.lr` is multiplied by at one step: a linear warmup, then nothing.

    Args:
        step: The step about to be taken, counted from 0 as `LambdaLR` counts it.
        warmup_steps: Steps the ramp spans. 0 leaves the rate at `lr` from the first step.
    Returns:
        The multiplier, in `(0, 1]`.
    """
    if step >= warmup_steps:
        return 1.0
    # From one step's worth of the rate rather than from 0, so the first step still moves.
    return (step + 1) / warmup_steps


def train(
    *,
    model: SuffixModel,
    train_loader: DataLoader,
    val_loader: DataLoader,
    generation_loader: DataLoader,
    generation_samples: int,
    codec: DatasetCodec,
    dataset: str,
    run: RunIdentity,
    experiment_config: dict,
    optimizer_config: DictConfig,
    training: DictConfig,
    early_stopping_config: DictConfig,
) -> None:
    """
    Train a model on a dataset, logging to W&B and saving checkpoints.

    A validation that improves on the best selection score so far overwrites
    `paths.BEST_CHECKPOINT`; no other step is kept. A run that ends, however it ends, is over:
    there is no carrying one on, so nothing here writes the optimizer, early-stopping or random
    state a resume would have read.

    Args:
        model: The model to train, already on `training.device`.
        train_loader: Batches to learn from.
        val_loader: Batches to score teacher-forced, every `training.val_every_n_steps` steps.
        generation_loader: Prefixes to generate suffixes for on the same cadence. A far smaller
            slice than `val_loader`, since a suffix costs one decoder pass per event.
        generation_samples: Suffixes to draw per prefix on that pass, normally
            `inference.validation_samples`, which matches the `inference.evaluation_samples` a
            report is built from: the selection score and the reported one are read at one budget.
        codec: The codec the splits were encoded through, passed on to the
            generation pass so its remaining times are scored in minutes.
        dataset: The log being trained on, naming the validation split's continuation index the
            selection score is read against.
        run: What every file this run writes is named after (see `src/paths.py`), what its W&B
            run is identified by (`src.identity.wandb_id`) and, minus its tag, which group and
            Artifact lineage it belongs to. One W&B run is one identity, so an identity reused
            across runs overlays their curves instead of listing them side by side; what makes
            its tag unique is the caller's business.
        experiment_config: The whole `DictConfig`, dumped to plain data, written into the
            checkpoint so the model can be rebuilt from the file alone.
        optimizer_config: The optimizer hyperparameters, its learning rate's warmup included.
            The warmup is stepped per optimizer step, so it means the same on every dataset.
        training: Step budget, validation cadence, gradient clipping and device.
        early_stopping_config: When to give up.
    """
    from src.model import save_checkpoint

    device = torch.device(training.device)

    # The validation split's continuations, which the selection score is measured against. Read
    # once here rather than per validation, and never the test split's: selecting against those
    # would fold the held-out set into which checkpoint is kept.
    continuations = ContinuationIndex.read(dataset=dataset, split=Split.VAL)

    # The declarative model generated suffixes are checked against, built once and reused: it
    # caches a trace's rate across the run rather than rebuilding the constraints per validation.
    checker = ConformanceChecker(dataset, ActivityCodes.of(codec.activity.names))

    optimizer = optim.Adam(
        model.parameters(), lr=optimizer_config.lr, weight_decay=optimizer_config.weight_decay
    )
    scheduler = optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: _lr_factor(step, warmup_steps=optimizer_config.warmup_steps),
    )
    early_stopper = EarlyStopper(early_stopping_config)

    step = 0
    should_stop = False
    interval_totals = Loss()
    seen = 0

    # The step the best checkpoint on disk came from, kept so the Artifact this run leaves can
    # say which step it is without anyone downloading it.
    best_step = 0

    # `group` is the experiment, `dataset/model`, so runs of one model on one log sit together and
    # a run is one attempt at it; `job_type` says which stage of the pipeline this is, leaving room
    # for a later generate or evaluate stage on the same run. The tags repeat the two halves of the
    # group so either can be filtered on alone, which one group string cannot do. The commit is
    # W&B's own to record: it reads it off the working tree at `init`.
    wandb.init(
        project=WANDB_PROJECT,
        id=wandb_id(run),
        name=str(run),
        group=experiment(run),
        job_type='train',
        tags=[run.dataset, run.model],
        config=experiment_config,
    )
    print(f'Logging to {wandb.run.url}')

    try:
        while step < training.max_steps and not should_stop:
            for batch in train_loader:
                model.train()
                batch = batch.to(device)
                # Run a forward pass
                output = model(batch)

                # Compute the loss and propagate gradients. Whatever the architecture anneals or
                # charges a KL term for is its own business, read off `step`.
                loss, metrics, latent = model.compute_loss(output, batch, step=step)
                optimizer.zero_grad()
                loss.backward()
                if training.grad_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), training.grad_clip_norm)
                # Read before the scheduler advances, so the rate logged is the one this step
                # was actually taken at rather than the one the next will be.
                learning_rate = scheduler.get_last_lr()[0]
                optimizer.step()
                scheduler.step()

                # Update the running totals and log to W&B
                batch_size = batch.suffix.activities.size(0)
                interval_totals += metrics
                seen += batch_size
                step += 1
                (metrics / batch_size).log(step, prefix='train')
                # So a loss curve can be read against where in the warmup it sits.
                wandb.log({'train/lr': learning_rate}, step=step)
                # Only a model with a latent has one to watch, and only it is charged a KL term.
                if latent is not None:
                    (latent / batch_size).log(step, prefix='train')

                if step % training.val_every_n_steps == 0 or step >= training.max_steps:
                    train_metrics = interval_totals / seen
                    interval_totals, seen = Loss(), 0

                    # Score the model on the validation set and the generation set, and log
                    # the results.
                    val_metrics, val_latent = validate(model, val_loader, step=step, device=device)
                    val_metrics.log(step, prefix='val')
                    if val_latent is not None:
                        val_latent.log(step, prefix='val')

                    gen_metrics = validate_generation(
                        model,
                        generation_loader,
                        num_samples=generation_samples,
                        codec=codec,
                        index=continuations,
                        checker=checker,
                        device=device,
                    )
                    gen_metrics.log(step)

                    # `val_latent.log` above already wrote `kl_weight` beside the rest of the
                    # latent metrics; only a model with a latent has one to show here.
                    kl_info = f'kl {val_latent.kl_weight:.2f}  ' if val_latent is not None else ''

                    # The one line of live feedback: enough to see a run is alive and heading down
                    print(
                        f'Step {step:>{len(str(training.max_steps))}}/{training.max_steps}  '
                        f'{kl_info}train {train_metrics.loss:.4f}  '
                        f'val {val_metrics.loss:.4f}  '
                        f'gen_dls {gen_metrics.accuracy.dls_mean:.4f} mean / '
                        f'{gen_metrics.accuracy.dls_point:.4f} point  '
                        f'emsc {gen_metrics.distribution.emsc:.4f} all / '
                        f'{gen_metrics.comparable.emsc:.4f} compared',
                        flush=True,
                    )
                    # The early stopper minimizes, and EMSC is a similarity, so it is the distance
                    # that is tracked. Over every prefix rather than over the ones a report reads
                    # it on, which is what every checkpoint under `outputs/` was selected on; the
                    # comparable score is logged beside it rather than selected on.
                    selection_score = 1.0 - gen_metrics.distribution.emsc

                    # Read before `update` folds this score into it, since afterwards it can
                    # no longer tell an improvement from a step that just matched the best.
                    is_best = selection_score < early_stopper.min_validation_score
                    should_stop = early_stopper.update(selection_score)

                    # Only an improvement is worth a file: the last step is never read back.
                    # The Artifact waits for the end of the run, so one run leaves one version
                    # rather than one per improvement.
                    if is_best:
                        best_step = step
                        path = save_checkpoint(
                            model,
                            experiment_config=experiment_config,
                            step=step,
                            selection_score=selection_score,
                            run=run,
                            path=paths.BEST_CHECKPOINT.prepare(run),
                        )
                        print(
                            f'New best model (step {step}, score {selection_score:.4f}) '
                            f'saved at {path}'
                        )

                if should_stop or step >= training.max_steps:
                    break

        # Everything below is only reached on a normal finish, not a crash, and while the W&B run
        # is still open. A run that dies leaves its best checkpoint on the machine that ran it and
        # nothing on W&B, which is the same thing a run that dies leaves behind anywhere else: it
        # cannot be carried on from either way.
        reason = (
            f'no validation improvement for {early_stopping_config.patience} validations'
            if should_stop
            else 'reached max_steps'
        )
        print(f'Finished training after {step} steps ({reason})')

        # One version per run, in the lineage its experiment shares, aliased with the run's own
        # tag. The file is uploaded as it sits: a checkpoint holds nothing a downloader would want
        # trimmed off it.
        artifact = wandb.Artifact(
            name=wandb_artifact(run),
            type='model',
            metadata={
                'run': str(run),
                'step': best_step,
                'selection_score': early_stopper.min_validation_score,
            },
        )
        artifact.add_file(str(paths.BEST_CHECKPOINT.path(run)), name='model.pt')
        wandb.log_artifact(artifact, aliases=[run.tag])

        # The alert is the one nobody has to be watching a terminal to get.
        wandb.alert(title=f'Training finished: {run}', text=f'{step} steps, {reason}.')
    finally:
        wandb.finish()
