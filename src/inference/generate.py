from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from src.configs.schema import InferenceConfig
from src.datasets.codec import DatasetCodec
from src.datasets.dataset import SplitTrace
from src.inference.generation import DecodedEvents, Draws, Generation
from src.suffixes import ActivityCodes

if TYPE_CHECKING:
    from src.model import SuffixModel


def generation_batch_size(
    inference: InferenceConfig, num_samples: int, prefixes_upper_bound: int
) -> int:
    """How many prefixes to hand the decoder at once, to protect its memory.

    Two bounds apply, in two different units: `prefixes_upper_bound` counts prefixes, and
    `generation_rows_upper_bound` counts the rows they expand into, one prefix being
    `num_samples` rows. They are weighed in rows, since rows are what decoder memory is spent on,
    and the smaller of the two is divided back into prefixes.

    Args:
        inference: Provides `generation_rows_upper_bound`, the row budget both passes share.
        num_samples: How many suffixes the pass being sized draws per prefix, which is what one
            prefix costs in rows: `inference.validation_samples` or `inference.evaluation_samples`.
        prefixes_upper_bound: Ceiling on prefixes per call, typically `dataloader.batch_size`.
    Returns:
        The batch size in prefixes, at least 1.
    """
    rows_per_call = min(inference.generation_rows_upper_bound, prefixes_upper_bound * num_samples)
    return max(1, rows_per_call // num_samples)


def generate_batch(
    model: SuffixModel,
    batch: SplitTrace,
    *,
    num_samples: int,
    codec: DatasetCodec,
    codes: ActivityCodes,
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
        codes: The dataset's codebook, seeded from `codec.activity.names`, which every suffix is
            spelled on. Passed in rather than built here so one codebook serves the whole run and
            is the one written into the file's metadata.
    Returns:
        One generation per prefix of the batch, in the batch's own order, each naming the case it
        was cut from. Everything is decoded into the log's own units and cut at its length, so what
        comes back holds events and nothing else, the EOT a generation ended on and the padding
        behind it both dropped.
    """
    generated = model.generate(item=batch, num_samples=num_samples)
    point = model.generate(item=batch, num_samples=1, sample=False)

    # Every suffix closes on an EOT, so true lengths are one less tha batch.suffix.length.
    true_lengths = (batch.suffix.length - 1).cpu().numpy()  # [batch_size]

    activities = generated.activities.cpu().numpy()  # [batch_size, num_samples, steps]
    lengths = generated.lengths.cpu().numpy()  # [batch_size, num_samples]
    # [batch_size, num_samples, steps]
    cycle_times = generated.cycle_times.cpu().numpy()
    remaining_time = generated.remaining_time.cpu().numpy()  # [batch_size, num_samples]
    point_activities = point.activities.squeeze(dim=1).cpu().numpy()  # [batch_size, steps]
    point_lengths = point.lengths.squeeze(dim=1).cpu().numpy()  # [batch_size]
    # [batch_size, steps]
    point_cycle_times = point.cycle_times.squeeze(dim=1).cpu().numpy()
    point_remaining_time = point.remaining_time.squeeze(dim=1).cpu().numpy()  # [batch_size]
    true_activities = batch.suffix.activities.cpu().numpy()  # [batch_size, seq_len]
    true_cycle_times = batch.cycle_times.cpu().numpy()  # [batch_size, seq_len]
    # Position 0 answers for the last prefix event, which is what a remaining time is measured
    # from.
    true_remaining_time = batch.remaining_times[:, 0].cpu().numpy()  # [batch_size]
    prefix_activities = batch.prefix.activities.cpu().numpy()  # [batch_size, seq_len]
    prefix_lengths = batch.prefix.length.cpu().numpy()  # [batch_size]

    return [
        Generation(
            case_id=batch.case_id[position],
            prefix_activities=codes.encode(
                codec.activity.decode(prefix_activities[position], length=prefix_lengths[position])
            ),
            samples=Draws.of(
                [
                    _decode(
                        codec,
                        codes,
                        activities=activities[position, sample],
                        cycle_times=cycle_times[position, sample],
                        length=lengths[position, sample],
                        remaining_time=remaining_time[position, sample],
                        clamp=True,
                    )
                    for sample in range(num_samples)
                ]
            ),
            point=_decode(
                codec,
                codes,
                activities=point_activities[position],
                cycle_times=point_cycle_times[position],
                length=point_lengths[position],
                remaining_time=point_remaining_time[position],
                clamp=True,
            ),
            truth=_decode(
                codec,
                codes,
                activities=true_activities[position],
                cycle_times=true_cycle_times[position],
                length=true_lengths[position],
                remaining_time=true_remaining_time[position],
            ),
        )
        for position in range(len(true_lengths))
    ]


def _decode(
    codec: DatasetCodec,
    codes: ActivityCodes,
    *,
    activities: np.ndarray,
    cycle_times: np.ndarray,
    length: int,
    remaining_time: float,
    clamp: bool = False,
) -> DecodedEvents:
    """One run of events, back in the log's own units.

    Args:
        codec: The codec the split was encoded through, read here in the decode direction.
        codes: The dataset's codebook, which the decoded names are spelled onto.
        activities: The run's activity indices, `[steps]`.
        cycle_times: The run's standardized cycle time before each of `activities`, `[steps]`.
        length: How many of them are events, the rest being the EOT and the padding behind it.
        remaining_time: The run's standardized remaining time.
        clamp: Whether to floor the denormalized times at 0, matching the baselines' behaviour on
            a model prediction. Left off for ground truth, which is never negative to begin with.
    Returns:
        The run as the report and the generations file hold it.
    """
    cycle_time_minutes = codec.cycle_time.denormalize(cycle_times[:length])
    remaining_time_minutes = float(codec.remaining_time.denormalize(remaining_time))
    if clamp:
        cycle_time_minutes = np.maximum(cycle_time_minutes, 0.0)
        remaining_time_minutes = max(remaining_time_minutes, 0.0)
    return DecodedEvents(
        activities=codes.encode(codec.activity.decode(activities, length=length)),
        cycle_time_minutes=cycle_time_minutes.tolist(),
        remaining_time_minutes=remaining_time_minutes,
    )
