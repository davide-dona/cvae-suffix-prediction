from dataclasses import dataclass

import numpy as np

from src.configs.schema import InferenceConfig
from src.datasets.codec import DatasetCodec
from src.datasets.dataset import SplitTrace
from src.model import TransformerCVAE


@dataclass(frozen=True)
class DecodedEvents:
    """Set of predicted events, in the log's own units. Any new addition
    to the model's output must be added here"""

    activities: list[str]
    remaining_time_minutes: float

    def __len__(self) -> int:
        return len(self.activities)


@dataclass(frozen=True)
class Generation:
    """One prefix's generated suffixes, the point prediction beside them, and the truth they were
    generated for.
    - case_id is used to identify the case in the log the prefix was cut from;
    - prefix_activities are added for convenience in the conformance report.
    """

    case_id: str  # which case of the log the prefix was cut from
    prefix_activities: list[str]  # the events before the cut, in order
    samples: list[DecodedEvents]  # one per draw of z
    point: DecodedEvents  # the suffix written from the mean of `p(z | prefix)`
    truth: DecodedEvents

    @property
    def prefix_len(self) -> int:
        """The cut point this answers: how many events ran before it."""
        return len(self.prefix_activities)


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
    batch: SplitTrace,
    *,
    num_samples: int,
    codec: DatasetCodec,
) -> list[Generation]:
    """Generate `num_samples` suffixes per prefix of one batch, and the point prediction beside
    them.

    Args:
        model: The model to generate with, already in eval mode.
        batch: A batch from `TraceDataset`, already on the model's device.
        num_samples: How many suffixes to draw per prefix.
        codec: The codec the split was encoded through, read here in the decode
            direction. Passed rather than read off the dataset, which is a `Subset` wherever only
            a slice of the split is generated for.
    Returns:
        One generation per prefix of the batch, in the batch's own order, each naming the case it
        was cut from. Everything is decoded into the log's own units and cut at its length, so what
        comes back holds events and nothing else, the EOT a generation ended on and the padding
        behind it both dropped.
    """
    generated = model.generate(item=batch, num_samples=num_samples)
    point = model.generate(item=batch, num_samples=1, sample_latent=False)

    # Every suffix closes on an EOT, so true lengths are one less tha batch.suffix.length.
    true_lengths = (batch.suffix.length - 1).cpu().numpy()  # [batch_size]

    activities = generated.activities.cpu().numpy()  # [batch_size, num_samples, steps]
    lengths = generated.lengths.cpu().numpy()  # [batch_size, num_samples]
    remaining_time = generated.remaining_time.cpu().numpy()  # [batch_size, num_samples]
    point_activities = point.activities.squeeze(dim=1).cpu().numpy()  # [batch_size, steps]
    point_lengths = point.lengths.squeeze(dim=1).cpu().numpy()  # [batch_size]
    point_remaining_time = point.remaining_time.squeeze(dim=1).cpu().numpy()  # [batch_size]
    true_activities = batch.suffix.activities.cpu().numpy()  # [batch_size, seq_len]
    true_remaining_time = batch.remaining_time.cpu().numpy()  # [batch_size]
    prefix_activities = batch.prefix.activities.cpu().numpy()  # [batch_size, seq_len]
    prefix_lengths = batch.prefix.length.cpu().numpy()  # [batch_size]

    return [
        Generation(
            case_id=batch.case_id[position],
            prefix_activities=codec.activity.decode(
                prefix_activities[position], length=prefix_lengths[position]
            ),
            samples=[
                _decode(
                    codec,
                    activities=activities[position, sample],
                    length=lengths[position, sample],
                    remaining_time=remaining_time[position, sample],
                )
                for sample in range(num_samples)
            ],
            point=_decode(
                codec,
                activities=point_activities[position],
                length=point_lengths[position],
                remaining_time=point_remaining_time[position],
            ),
            truth=_decode(
                codec,
                activities=true_activities[position],
                length=true_lengths[position],
                remaining_time=true_remaining_time[position],
            ),
        )
        for position in range(len(true_lengths))
    ]


def _decode(
    codec: DatasetCodec, *, activities: np.ndarray, length: int, remaining_time: float
) -> DecodedEvents:
    """One run of events, back in the log's own units.

    Args:
        codec: The codec the split was encoded through, read here in the decode direction.
        activities: The run's activity indices, `[steps]`.
        length: How many of them are events, the rest being the EOT and the padding behind it.
        remaining_time: The run's standardized remaining time.
    Returns:
        The run as the report and the generations file hold it.
    """
    return DecodedEvents(
        activities=codec.activity.decode(activities, length=length),
        remaining_time_minutes=float(codec.remaining_time.denormalize(remaining_time)),
    )
