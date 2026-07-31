from __future__ import annotations
from pathlib import Path
from typing import Union
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.datasets.codec import Codec, EncodedSequence
from src.datasets.dataset import SuffixDataset, SuffixItem
from src.inference.prediction import PredictionRow
from src.model import TransformerCVAE
from src.model.components.decoder import GeneratedSuffix


# The most rows `generate` should put through the decoder in one call. A row carries a cached
# key and value per position, per layer, for both its self-attention and the encoded prefix, so
# what a call holds grows with this and with `max_seq_len` together: at bpic-2017's 95 positions
# and d_model 128 a row is about 400 KB of cache, which puts 512 rows at a couple of hundred
# megabytes and 2560 into the region where this machine starts paging.
GENERATION_ROWS = 512


def generation_batch_size(num_samples: int, upper_bound: int) -> int:
    """How many prefixes to generate for at once.

    `generate` repeats every prefix once per sample, so a batch of prefixes reaches the decoder
    as `batch_size * num_samples` rows. Sizing a generation loader by `data.batch_size`, which
    was chosen for training where a row is one sequence, therefore asks for `num_samples` times
    the work and the memory it looks like it is asking for.

    Args:
        num_samples: Suffixes drawn per prefix.
        upper_bound: Batch size not to exceed anyway, normally `data.batch_size`. Keeps a log
            generating a single sample per prefix from being handed a larger batch than it
            trains on.
    Returns:
        A batch size of at least 1.
    """
    return max(1, min(upper_bound, GENERATION_ROWS // num_samples))


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
    """Generate `num_samples` suffixes for every prefix in `loader`, and return them as a DataFrame of `PredictionRow`s.
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
    # one of these is what picks a single suffix out of it.
    predicted_suffixes = EncodedSequence(
        activities=generated.activities.cpu().numpy(),  # [batch_size, num_samples, steps]
        time_deltas=generated.time_deltas.cpu().numpy(),  # normalized, [0, 1]
    )
    true_suffixes = EncodedSequence(
        activities=batch.suffix.activities.cpu().numpy(),  # [batch_size, seq_len]
        time_deltas=batch.suffix.time_deltas.cpu().numpy(),  # normalized, [0, 1]
        resources=batch.suffix.resources.cpu().numpy(),
    )

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
                    predicted_time_deltas_minutes=predicted.time_deltas_minutes,
                    true_activities=ground_truth.activities,
                    true_resources=ground_truth.resources,
                    true_time_deltas_minutes=ground_truth.time_deltas_minutes,
                )
            )
    return rows
