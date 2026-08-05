import torch
from torch.utils.data import DataLoader

from src.datasets.description import DatasetDescription
from src.inference import generate_batch
from src.model import TransformerCVAE
from src.scoring import SuffixScores, score_prefix
from src.training.loss import Loss, compute_loss


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
) -> SuffixScores:
    """
    Generate suffixes from the prefixes in `loader` and compare them to the ground truth.

    Scored through `score_prefix`, the same function the final report is built from. The one
    remaining difference is the population: truncated pairs are kept here and dropped by
    `pipelines/evaluate.py`, whose ground-truth suffixes stop short of the real ending.

    Args:
        model: The model to evaluate. Put in evaluation mode here, and left in it.
        loader: The prefixes to generate for, from a `SuffixDataset`.
        num_samples: Suffixes to draw per prefix. The spread across them is what
            `sample_diversity` measures, and `generate` puts `len(batch) * num_samples` rows
            through the decoder at once, so it is also what the caller sizes its batches by.
        description: The description the split was encoded through, read here to put the
            generations back into the log's own units. Passed rather than read off
            `loader.dataset`, which is a `Subset` wherever the split is bigger than the slice
            validated on.
        device: The device to run the computations on.
    Returns:
        The metrics of the pass, averaged over prefixes.
    """
    model.eval()

    scores = [
        score_prefix(generation)
        for batch in loader
        for generation in generate_batch(
            model=model,
            batch=batch.to(device),
            num_samples=num_samples,
            description=description,
        )
    ]
    return SuffixScores.mean(scores)
