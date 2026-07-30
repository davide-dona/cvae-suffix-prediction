from dataclasses import dataclass


@dataclass(frozen=True)
class PredictionRow:
    """One row of a predictions file. Each row corresponds to one (prefix, sample) pair.
    The predicted suffix is decoded, denormalized and aligned with the ground-truth suffix, so the two can be compared directly.
    The `sample_index` column enumerates the generated suffixes of one prefix, from 0 to `num_samples - 1`;
    it identifies a row rather than describing it, so nothing should be read into its value."""
    case_id: str
    prefix_len: int
    truncated: bool         # whether the ground-truth suffix was cut short of its real ending
    sample_index: int       # which of the `num_samples` generated suffixes of this prefix this is
    predicted_activities: list[str]
    predicted_resources: list[str]
    predicted_time_deltas_minutes: list[float]
    true_activities: list[str]
    true_resources: list[str]
    true_time_deltas_minutes: list[float]
