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
from src import paths
from src.datasets.description import DatasetDescription
from src.model import TransformerCVAE, save_checkpoint
from src.training.early_stopping import EarlyStopper
from src.training.kl import cyclical_linear_weight
from src.training.loss import Loss, compute_loss
from src.training.validation import validate, validate_generation


# The device's own stream, which dropout and the latent sampling draw from, lives behind a
# namespace of its own; `cpu` has none beside the global one.
DEVICE_RNG = {
    'cuda': (torch.cuda.get_rng_state, torch.cuda.set_rng_state),
    'mps': (torch.mps.get_rng_state, torch.mps.set_rng_state),
}


def rng_state(generator: torch.Generator, device: torch.device) -> dict:
    """
    Every random stream a run draws from, so a resumed one carries on where it left off.

    Args:
        generator: The generator the training loader shuffles with.
        device: The device the model runs on.
    Returns:
        The states, keyed for `restore_rng_state`.
    """
    capture = DEVICE_RNG.get(device.type)
    return {
        'cpu': torch.get_rng_state(),
        'dataloader': generator.get_state(),
        'device': capture[0]() if capture is not None else None,
    }


def restore_rng_state(state: dict, *, generator: torch.Generator, device: torch.device) -> None:
    """
    Put back what `rng_state` captured, on the device the run trained on.

    Args:
        state: What `rng_state` returned.
        generator: The generator the training loader shuffles with.
        device: The device the model runs on.
    """
    torch.set_rng_state(state['cpu'])
    generator.set_state(state['dataloader'])

    restore = DEVICE_RNG.get(device.type)
    if restore is not None:
        restore[1](state['device'])


def train(
    *,
    model: TransformerCVAE,
    train_loader: DataLoader,
    val_loader: DataLoader,
    generation_loader: DataLoader,
    generation_samples: int,
    description: DatasetDescription,
    run_name: str,
    experiment_config: dict,
    generator: torch.Generator,
    resume: dict | None,
    loss_config: LossConfig,
    optimizer_config: OptimizerConfig,
    training: TrainingConfig,
    early_stopping_config: EarlyStoppingConfig,
) -> None:
    """
    Train a model on a dataset, logging to TensorBoard and saving checkpoints.

    Every validation writes a checkpoint to `paths.checkpoint_path`, whether or not the step is
    the best one, and a validation that improves on the best writes a second copy to
    `paths.best_model_path`. Either file can be handed back as `resume`.

    Args:
        model: The model to train, already on `training.device`.
        train_loader: Batches to learn from.
        val_loader: Batches to score teacher-forced, every `training.val_every_n_steps` steps.
        generation_loader: Prefixes to generate suffixes for on the same cadence. A far smaller
            slice than `val_loader`, since a suffix costs one decoder pass per event.
        generation_samples: Suffixes to draw per prefix on that pass, normally
            `inference.num_samples`, so that a training curve and the final report describe the
            same number of draws.
        description: The description the splits were encoded through, passed on to the
            generation pass so its remaining times are scored in minutes.
        run_name: The name every file this run writes is derived from (see `src/paths.py`). One
            TensorBoard directory is one run, so a name reused across runs overlays their
            curves instead of listing them side by side; what makes it unique is the
            caller's business. A `/` in it nests the run, which is how TensorBoard groups
            runs under a common prefix.
        experiment_config: The whole `ExperimentConfig`, dumped to plain data, written into
            every checkpoint so the run can be rebuilt and carried on from the file alone.
        generator: The generator the training loader shuffles with, whose state is saved and
            restored along with the rest.
        resume: A checkpoint to carry on from, read by `load_checkpoint`, or `None` to start
            from step zero. Its step, weights, optimizer state, early-stopping state and random
            streams are all restored.
        loss_config: The KL annealing schedule.
        optimizer_config: The optimizer hyperparameters.
        training: Step budget, validation cadence, gradient clipping and device.
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

    # If resuming, restore the model, optimizer, early stopper and random streams to the state
    # they were in when the checkpoint was written.
    if resume is not None:
        model.load_state_dict(state_dict=resume['model_state_dict'])
        optimizer.load_state_dict(state_dict=resume['optimizer_state'])
        early_stopper.load_state_dict(resume['early_stopping_state'])
        # Restored after the loaders were built, so that the fixed validation and generation
        # subsets are still drawn from the seeded stream exactly as an unresumed run draws them.
        restore_rng_state(resume['rng_state'], generator=generator, device=device)
        step = resume['step']
        print(f'Resuming {run_name} from step {step}, best score {resume["selection_score"]:.4f}')

    # Overwrite the TensorBoard logs after the step we resumed from, so that a resumed run's curves are continuous with the original.
    writer = SummaryWriter(
        log_dir=paths.tensorboard_dir(run_name), purge_step=step if resume is not None else None
    )
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
                        num_samples=generation_samples, description=description, device=device,
                    )
                    gen_metrics.log(writer, step, prefix='gen')

                    writer.add_scalar('kl_weight', kl_weight, step)

                    # The one line of live feedback: enough to see a run is alive and heading down
                    print(
                        f'Step {step:>{len(str(training.max_steps))}}/{training.max_steps}  '
                        f'kl {kl_weight:.2f}  train {train_metrics.loss:.4f}  '
                        f'val {val_metrics.loss:.4f}  '
                        f'gen_dls {gen_metrics.dls_mean:.4f} mean / '
                        f'{gen_metrics.dls_point:.4f} point  '
                        f'energy {gen_metrics.energy_score:.4f}',
                        flush=True,
                    )
                    selection_score = gen_metrics.energy_score

                    # Read before `update` folds this score into it, since afterwards it can
                    # no longer tell an improvement from a step that just matched the best.
                    is_best = selection_score < early_stopper.min_validation_score
                    should_stop = early_stopper.update(selection_score)

                    # Save a checkpoint with the model, optimizer, early stopper and random streams in their current state.
                    checkpoint = dict(
                        experiment_config=experiment_config,
                        step=step,
                        selection_score=selection_score,
                        run_name=run_name,
                        optimizer_state=optimizer.state_dict(),
                        early_stopping_state=early_stopper.state_dict(),
                        rng_state=rng_state(generator, device),
                    )
                    save_checkpoint(model, **checkpoint, path=paths.checkpoint_path(run_name))
                    # If this validation improved on the best, save a second copy to the best-model directory.
                    if is_best:
                        path = save_checkpoint(
                            model, **checkpoint, path=paths.best_model_path(run_name)
                        )
                        print(f'New best model (step {step}, score {selection_score:.4f}) saved at {path}')

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
