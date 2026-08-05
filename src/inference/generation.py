from dataclasses import dataclass

from src.configs.schema import InferenceConfig
from src.datasets.codec import DecodedSequence, decode_sequence
from src.datasets.dataset import SuffixItem
from src.datasets.description import DatasetDescription
from src.model import TransformerCVAE


@dataclass(frozen=True)
class Generation:
    """The model's answer for one prefix: the suffixes it drew, the one it answers with, and the
    ground truth beside them.

    The one shape everything downstream reads, and exactly what `score_prefix` takes. A training
    curve, the generations file and the final report are all scored off this, so they measure the
    same thing rather than agreeing by coincidence. `generate_batch` builds it from the model;
    `generation_from_rows` builds it back out of a generations file, which is only this written down.

    Which prefix this answers is not part of it: the caller that needs identity has the batch, and
    the file names a prefix by its case and cut point.
    """
    samples: list[DecodedSequence]  # one per draw of z
    point: DecodedSequence          # the suffix written from the mean of `p(z | prefix)`
    truth: DecodedSequence


def generation_batch_size(inference: InferenceConfig, upper_bound: int) -> int:
    """How many prefixes to hand the decoder at once, to protect its memory.

    Each prefix expands into `num_samples` rows, so the batch size is
    `generation_rows // num_samples`: the row count stays bounded however `num_samples` is
    configured. Capped at `upper_bound` and floored at 1.

    Args:
        inference: Provides `generation_rows` (the row budget) and `num_samples`.
        upper_bound: Hard ceiling on the batch size, typically `data.batch_size`.
    Returns:
        The batch size, at least 1.
    """
    return max(1, min(upper_bound, inference.generation_rows // inference.num_samples))


def generate_batch(
    model: TransformerCVAE,
    batch: SuffixItem,
    *,
    num_samples: int,
    description: DatasetDescription,
) -> list[Generation]:
    """Generate `num_samples` suffixes per prefix of one batch, and the point prediction beside them.

    The one call into the raw `model.generate`, and the unit every caller works in: a caller walks
    its own loader and calls this per batch. The second pass costs one prefix's worth of decoding
    against `num_samples`, and it is what lets the report say what the model answers as well as
    what it draws.

    Args:
        model: The model to generate with, already in eval mode.
        batch: A batch from `SuffixDataset`, already on the model's device.
        num_samples: How many suffixes to draw per prefix.
        description: The description the split was encoded through, read here in the decode
            direction. Passed rather than read off the dataset, which is a `Subset` wherever only
            a slice of the split is generated for.
    Returns:
        One generation per prefix of the batch, in the batch's own order, so a caller that needs
        identity reads it off `batch.pair_index`. Everything is decoded into the log's own units and
        cut at its length, so what comes back holds events and nothing else, the EOT a generation
        ended on and the padding behind it both dropped.
    """
    generated = model.generate(item=batch, num_samples=num_samples)
    point = model.generate(item=batch, num_samples=1, sample_latent=False)

    # `suffix_len` counts the EOT closing a complete suffix, which is a marker and not an
    # event; a truncated suffix has none to drop.
    last_position = (batch.suffix_len - 1).unsqueeze(dim=1)  # [batch_size, 1]
    ends_with_eot = (
        batch.suffix.activities.gather(dim=1, index=last_position).squeeze(dim=1)
        == model.eot_activity_index
    )  # [batch_size]
    true_lengths = (batch.suffix_len - ends_with_eot.long()).cpu().numpy()  # [batch_size]

    activities = generated.activities.cpu().numpy()          # [batch_size, num_samples, steps]
    lengths = generated.lengths.cpu().numpy()                # [batch_size, num_samples]
    remaining_time = generated.remaining_time.cpu().numpy()  # [batch_size, num_samples]
    point_activities = point.activities.squeeze(dim=1).cpu().numpy()          # [batch_size, steps]
    point_lengths = point.lengths.squeeze(dim=1).cpu().numpy()                # [batch_size]
    point_remaining_time = point.remaining_time.squeeze(dim=1).cpu().numpy()  # [batch_size]
    true_activities = batch.suffix.activities.cpu().numpy()  # [batch_size, seq_len]
    true_remaining_time = batch.remaining_time.cpu().numpy()  # [batch_size]

    return [
        Generation(
            samples=[
                decode_sequence(
                    description,
                    activities=activities[position, sample],
                    length=lengths[position, sample],
                    remaining_time=remaining_time[position, sample],
                )
                for sample in range(num_samples)
            ],
            point=decode_sequence(
                description,
                activities=point_activities[position],
                length=point_lengths[position],
                remaining_time=point_remaining_time[position],
            ),
            truth=decode_sequence(
                description,
                activities=true_activities[position],
                length=true_lengths[position],
                remaining_time=true_remaining_time[position],
            ),
        )
        for position in range(len(true_lengths))
    ]
