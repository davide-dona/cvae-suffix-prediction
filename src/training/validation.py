from dataclasses import dataclass
import numpy as np
import torch
from torch.utils.data import DataLoader

from src.datasets.description import DatasetDescription
from src.evaluation.sequences import score_samples, sequence_similarity
from src.inference import generate_batch
from src.model import TransformerCVAE
from src.training.loss import Loss, compute_loss
from src.training.metrics import ScalarMetrics


@dataclass(frozen=True)
class GenerationMetrics(ScalarMetrics):
    """What the model produced when it was made to write suffixes on its own.
    Attributes:
        activity_dls_mean: The mean similarity of a prefix's samples to the ground truth.
        activity_dls_point: The similarity of the suffix it writes from the mean of
            `p(z | prefix)`, its single answer.
        activity_dls_best: The similarity of its closest sample.
        activity_energy_score: Its samples read as a predictive distribution, lower being better.
            The score a checkpoint is selected on.
        sample_diversity: How far apart its samples are from each other.
        unique_sample_rate: The fraction of its samples that are distinct sequences.
        remaining_time_ae_mean_minutes: The mean absolute error of its samples' remaining cycle
            time, in minutes, and `_point` the error of the one it answers with.
        length_ae_mean: The same for the suffix length, in events.

    The errors are the free-running ones: `Loss.remaining_time_loss` scores the same head
    teacher-forced, which is a different question and a far easier one.
    """
    activity_dls_mean: float = 0.0
    activity_dls_point: float = 0.0
    activity_dls_best: float = 0.0
    activity_energy_score: float = 0.0
    sample_diversity: float = 0.0
    unique_sample_rate: float = 0.0
    remaining_time_ae_mean_minutes: float = 0.0
    remaining_time_ae_point_minutes: float = 0.0
    length_ae_mean: float = 0.0
    length_ae_point: float = 0.0


@torch.no_grad()
def validate(
    model: TransformerCVAE,
    loader: DataLoader,
    *,
    kl_weight: float,
    free_bits: float,
    device: torch.device,
) -> Loss:
    """
    Run one pass over `loader` without learning from it.
    Args:
        model: The model to evaluate. Put in evaluation mode here, and left in it.
        loader: The dataloader to iterate over. Its batches are `SuffixItem`s.
        kl_weight: The weight this step's KL term is given.
        free_bits: Nats per latent dimension the KL is not penalized below.
        device: The device to run the computations on.
    Returns:
        The metrics of the pass, averaged over the traces of the split.
    """
    model.eval()

    totals = Loss()
    for batch in loader:
        batch = batch.to(device)
        output = model(batch)
        _, metrics = compute_loss(
            output, batch,
            pad_activity_index=model.pad_activity_index, kl_weight=kl_weight, free_bits=free_bits,
        )
        totals += metrics

    return totals / len(loader.dataset)


@torch.no_grad()
def validate_generation(
    model: TransformerCVAE,
    loader: DataLoader,
    *,
    num_samples: int,
    description: DatasetDescription,
    device: torch.device,
) -> GenerationMetrics:
    """
    Generate suffixes from the prefixes in `loader` and compare them to the ground truth.
    Args:
        model: The model to evaluate. Put in evaluation mode here, and left in it.
        loader: The prefixes to generate for, from a `SuffixDataset`.
        num_samples: Suffixes to draw per prefix. The spread across them is what
            `sample_diversity` measures, and `generate` puts `len(batch) * num_samples` rows
            through the decoder at once, so it is also what the caller sizes its batches by.
        description: The description the split was encoded through, read here to put the
            remaining times back into minutes. Passed rather than read off `loader.dataset`,
            which is a `Subset` wherever the split is bigger than the slice validated on.
        device: The device to run the computations on.
    Returns:
        The metrics of the pass, averaged over prefixes.
    """
    model.eval()
    totals = GenerationMetrics()
    # The log1p range the remaining times are normalized through is not linear, so an error in
    # that space is not an error in minutes rescaled; they are read back before being measured.
    remaining_time = description.remaining_time

    for batch in loader:
        batch = batch.to(device)
        generation = generate_batch(model, batch, num_samples=num_samples)

        positions = range(len(generation.true_lengths))
        scores = [
            score_samples(generation.samples(position), generation.truth(position))
            for position in positions
        ]

        true_minutes = remaining_time.denormalize(generation.true_remaining_time)  # [batch_size]
        # [batch_size, num_samples] against the truth broadcast over the samples, then [batch_size]
        time_ae = np.abs(
            remaining_time.denormalize(generation.remaining_time) - true_minutes[:, None]
        )
        point_time_ae = np.abs(
            remaining_time.denormalize(generation.point_remaining_time) - true_minutes
        )
        length_ae = np.abs(generation.lengths - generation.true_lengths[:, None])
        point_length_ae = np.abs(generation.point_lengths - generation.true_lengths)

        totals += GenerationMetrics(
            activity_dls_mean=sum(score.dls_mean for score in scores),
            activity_dls_point=sum(
                sequence_similarity(generation.point(position), generation.truth(position))
                for position in positions
            ),
            activity_dls_best=sum(score.dls_best for score in scores),
            activity_energy_score=sum(score.energy_score for score in scores),
            sample_diversity=sum(score.sample_diversity for score in scores),
            unique_sample_rate=sum(score.unique_sample_rate for score in scores),
            # Averaged over a prefix's samples first, so a prefix weighs the same here as it
            # does in `accuracy_metrics`.
            remaining_time_ae_mean_minutes=float(time_ae.mean(axis=1).sum()),
            remaining_time_ae_point_minutes=float(point_time_ae.sum()),
            length_ae_mean=float(length_ae.mean(axis=1).sum()),
            length_ae_point=float(point_length_ae.sum()),
        )

    return totals / len(loader.dataset)
