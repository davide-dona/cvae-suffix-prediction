from dataclasses import dataclass
import torch
from torch.utils.data import DataLoader

from src.evaluation.sequences import score_samples
from src.model import TransformerCVAE
from src.training.loss import Loss, compute_loss
from src.training.metrics import ScalarMetrics


@dataclass(frozen=True)
class GenerationMetrics(ScalarMetrics):
    """What the model produced when it was made to write suffixes on its own.
    Attributes:
        activity_dls_mean: The mean similarity of a prefix's samples to the ground truth.
        activity_dls_best: The similarity of its closest sample.
        sample_diversity: How far apart its samples are from each other.
    """
    activity_dls_mean: float = 0.0
    activity_dls_best: float = 0.0
    sample_diversity: float = 0.0


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
        device: The device to run the computations on.
    Returns:
        The metrics of the pass, averaged over prefixes.
    """
    model.eval()
    totals = GenerationMetrics()

    for batch in loader:
        batch = batch.to(device)
        # Generate `num_samples` suffixes for each prefix in the batch
        generated = model.generate(batch, num_samples=num_samples)

        # Get the true lengths of the suffixes, accounting for whether they end with EOT or not
        last_position = (batch.suffix_len - 1).unsqueeze(dim=1)  # [batch_size, 1]
        ends_with_eot = (
            batch.suffix.activities.gather(dim=1, index=last_position).squeeze(dim=1)
            == model.eot_activity_index
        )  # [batch_size]
        true_lengths = (batch.suffix_len - ends_with_eot.long()).tolist()
        sample_lengths = generated.lengths.cpu().tolist()
        
        # Get the true and generated activities as lists of lists, for easier comparison
        true_activities = batch.suffix.activities.cpu().tolist()
        sample_activities = generated.activities.cpu().tolist()  # [batch, num_samples, steps]
        

        scores = []
        for position, true_length in enumerate(true_lengths):
            # Get the ground truth suffix and the generated samples for this prefix
            truth = true_activities[position][:true_length]
            samples = [
                sample_activities[position][sample][: sample_lengths[position][sample]]
                for sample in range(num_samples)
            ]
            scores.append(score_samples(samples, truth))

        totals += GenerationMetrics(
            activity_dls_mean=sum(score.dls_mean for score in scores),
            activity_dls_best=sum(score.dls_best for score in scores),
            sample_diversity=sum(score.sample_diversity for score in scores),
        )

    return totals / len(loader.dataset)
