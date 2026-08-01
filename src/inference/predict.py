from __future__ import annotations
from pathlib import Path
from typing import Union
import pandas as pd
import torch
from torch.utils.data import DataLoader
from dataclasses import dataclass

from src.configs.schema import InferenceConfig
from src.datasets.codec import Codec, EncodedSequence
from src.datasets.dataset import SuffixDataset, SuffixItem
from src.model import TransformerCVAE
from src.model.components.decoder import GeneratedSuffix

@dataclass(frozen=True)
class PredictionRow:
    """One row of a predictions file. Each row corresponds to one (prefix, sample) pair.
    The predicted suffix is decoded, denormalized and aligned with the ground-truth suffix, so the two can be compared directly.
    The `sample_index` column enumerates the generated suffixes of one prefix, from 0 to `num_samples - 1`;
    it identifies a row rather than describing it, so nothing should be read into its value.

    The model writes activities and a remaining time only, so there is no `predicted_resources`
    to put beside `true_resources`. The ground-truth resources stay: they cost nothing to carry
    and they are what a later run that does predict them would be scored against.

    The two remaining times are the time from the end of the prefix to the end of the case, in
    minutes. They are scalars rather than the per-event gaps an earlier version wrote: a gap
    between consecutive events is not well defined on a log that registers a batch of them
    under one timestamp, and their sum over a suffix is."""
    case_id: str
    prefix_len: int
    truncated: bool         # whether the ground-truth suffix was cut short of its real ending
    sample_index: int       # which of the `num_samples` generated suffixes of this prefix this is
    predicted_activities: list[str]
    predicted_remaining_time_minutes: float
    true_activities: list[str]
    true_resources: list[str]
    true_remaining_time_minutes: float

def generation_batch_size(inference: InferenceConfig, upper_bound: int) -> int:
    """Compute how many prefixes to hand the decoder at once, to protect its memory.

    Each prefix expands into `num_samples` rows, so batch_size is derived
    as `generation_rows // num_samples` — keeping total rows in check
    regardless of num_samples, which is configured separately. Also capped
    at `upper_bound` (typically `data.batch_size`) and floored at 1.

    Args:
        inference: Provides `generation_rows` (row budget) and `num_samples`.
        upper_bound: Hard ceiling on the batch size, e.g. to stop a
            single-sample-per-prefix config from exceeding the training batch size.

    Returns:
        The batch size, at least 1.
    """
    return max(1, min(upper_bound, inference.generation_rows // inference.num_samples))


def predictions_path(predictions_dir: Union[str, Path], run_name: str) -> Path:
    """
    Where the predictions of one run are kept: `<predictions_dir>/<run_name>.parquet`.

    One file per run, named after it exactly as `best_model_path` names a run's checkpoint,
    so a run's predictions are found without being told anything but its name.
    """
    return Path(predictions_dir) / f'{run_name}.parquet'


def generate_predictions(
    model: TransformerCVAE,
    loader: DataLoader,
    codec: Codec,
    *,
    num_samples: int,
    device: torch.device,
) -> pd.DataFrame:
    """Generate `num_samples` suffixes for every prefix in `loader`; return them as a DataFrame of `PredictionRow`s.
    The model predictions are denormalized and decoded back into the original activity values and minutes, so
    what comes out is comparable against the log itself.
    
    Args:
        model: The trained model, already on `device`.
        loader: Batches of the split to predict for, from a `SuffixDataset`.
        codec: The codec the split was encoded with, used here in its decode direction.
        num_samples: How many suffixes to generate per prefix.
        device: Where to run the generation.
    Returns:
        One row per (prefix, sample), as a `PredictionRow`.
    """
    dataset: SuffixDataset = loader.dataset

    # The model is in eval mode, so dropout and other training-only behavior is off
    model.eval()

    rows: list[PredictionRow] = []
    # Retrieve one batch at a time, generate `num_samples` suffixes for each prefix in the batch, and turn them into rows
    for batch in loader:
        batch = batch.to(device)
        generated = model.generate(batch, num_samples=num_samples)
        rows += _batch_rows(batch=batch, generated=generated, dataset=dataset, codec=codec)

    return pd.DataFrame(rows)


def _batch_rows(
    batch: SuffixItem,
    generated: GeneratedSuffix,
    dataset: SuffixDataset,
    codec: Codec,
) -> list[PredictionRow]:
    """Turn one batch of generated suffixes, and the ground truths they were generated against, into rows."""
    pair_indices = batch.pair_index.tolist()
    suffix_lengths = batch.suffix_len.cpu().numpy()
    lengths = generated.lengths.cpu().numpy()  # [batch_size, num_samples]

    # The whole batch moved off the device once, in the shape `decode_suffix` reads: indexing
    # one of these is what picks a single suffix out of it. Neither carries a time channel,
    # which `decode_suffix` no longer reads: what the model predicts about time is the scalar
    # below, not a value per event.
    predicted_suffixes = EncodedSequence(
        activities=generated.activities.cpu().numpy(),  # [batch_size, num_samples, steps]
    )
    true_suffixes = EncodedSequence(
        activities=batch.suffix.activities.cpu().numpy(),  # [batch_size, seq_len]
        resources=batch.suffix.resources.cpu().numpy(),
    )

    # Both sides of the time comparison, back in minutes: one predicted per (prefix, sample),
    # one ground truth per prefix.
    predicted_remaining = codec.denormalize_remaining_time(
        generated.remaining_time.cpu().numpy()  # [batch_size, num_samples]
    )
    true_remaining = codec.denormalize_remaining_time(batch.remaining_time.cpu().numpy())

    rows = []
    for position, pair_index in enumerate(pair_indices):
        info = dataset.pair_info(pair_index)
        # `suffix_len` counts the EOT token that closes a complete suffix, which is a marker
        # and not an event of the log. A truncated case has none to drop.
        content_len = suffix_lengths[position] - (0 if info.truncated else 1)
        ground_truth = codec.decode_suffix(true_suffixes[position], length=content_len)

        for sample_index in range(lengths.shape[1]):
            predicted = codec.decode_suffix(
                predicted_suffixes[position, sample_index],
                length=lengths[position, sample_index],
            )
            rows.append(
                PredictionRow(
                    case_id=info.case_id,
                    prefix_len=info.prefix_len,
                    truncated=info.truncated,
                    sample_index=sample_index,
                    predicted_activities=predicted.activities,
                    predicted_remaining_time_minutes=float(
                        predicted_remaining[position, sample_index]
                    ),
                    true_activities=ground_truth.activities,
                    true_resources=ground_truth.resources,
                    true_remaining_time_minutes=float(true_remaining[position]),
                )
            )
    return rows
