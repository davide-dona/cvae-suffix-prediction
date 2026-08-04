from dataclasses import dataclass

from src.inference.prediction import PrefixPrediction
from src.metrics import ScalarMetrics
from src.scoring.sequences import diversity, energy_score, mean, sequence_similarity

MINUTES_PER_DAY = 1440.0


@dataclass(frozen=True)
class SuffixScores(ScalarMetrics):
    """The scores of one prefix's generated samples, or their mean over a set of prefixes.

    These are the free-running numbers: the model wrote every suffix scored here on its own, one
    event at a time. `Loss.remaining_time_loss` scores the same head teacher-forced, which is a
    different question and a far easier one.

    No defaults: a field added here must be set in `score_prefix`, and a `TypeError` is how that
    is enforced.
    """
    # The Damerau-Levenshtein Similarity (DLS) is the edit distance normalized to [0, 1] and inverted.
    activity_dls_mean: float        # The mean similarity of a prefix's samples to the ground truth
    activity_dls_point: float       # z = mean(p(z | prefix), a single greedy answer, scored against the ground truth
    activity_dls_best: float        # The similarity of the closest sample to the ground truth

    # The samples read as a predictive distribution, lower being better. The score a checkpoint
    # is selected on.
    activity_energy_score: float

    # How far apart the samples of a prefix are, and how many of them are distinct sequences.
    # Both say how much of the prefix's uncertainty z carries, not how good the model is.
    sample_diversity: float
    unique_sample_rate: float

    # Absolute error (AE) between the predicted and true remaining cycle time, in days.
    # No best since the closest of ten draws of a scalar measures how widely the head scatters
    # rather than how well it predicts.
    remaining_time_ae_mean_days: float
    remaining_time_ae_point_days: float

    # Absolute error (AE) between the predicted and true suffix length, in events.
    length_ae_mean: float
    length_ae_point: float

    # Events left after the cut point: the scale every error above is read against. A property of
    # the prefixes scored rather than of the model, so it is flat across a training run.
    suffix_length: float


def score_prefix(prediction: PrefixPrediction) -> SuffixScores:
    """
    Score the suffixes generated for one prefix against the ground truth they continue.

    The one place the reported numbers are defined: `validate_generation` averages them over a
    validation slice while training, `accuracy_metrics` over the test split afterwards, so a
    training curve and a final report measure the same thing rather than agreeing by coincidence.

    Args:
        prediction: The model's answer for one prefix, decoded into the log's own units.
    Returns:
        The prefix's scores. A prefix with no samples scores 0.0 on everything and 1.0 on
        `activity_energy_score`, the worst it can be, rather than looking like a perfect
        prediction.
    """
    samples, point, truth = prediction.samples, prediction.point, prediction.truth

    similarities = [sequence_similarity(sample.activities, truth.activities) for sample in samples]
    dls_mean = mean(similarities)
    sample_diversity = diversity([sample.activities for sample in samples])

    return SuffixScores(
        activity_dls_mean=dls_mean,
        activity_dls_point=sequence_similarity(point.activities, truth.activities),
        activity_dls_best=max(similarities) if similarities else 0.0,
        activity_energy_score=energy_score(dls_mean, sample_diversity) if samples else 1.0,
        sample_diversity=sample_diversity,
        unique_sample_rate=(
            len({tuple(sample.activities) for sample in samples}) / len(samples)
            if samples
            else 0.0
        ),
        remaining_time_ae_mean_days=mean([
            abs(sample.remaining_time_minutes - truth.remaining_time_minutes)
            for sample in samples
        ]) / MINUTES_PER_DAY,
        remaining_time_ae_point_days=abs(
            point.remaining_time_minutes - truth.remaining_time_minutes
        ) / MINUTES_PER_DAY,
        length_ae_mean=mean([float(abs(len(sample) - len(truth))) for sample in samples]),
        length_ae_point=float(abs(len(point) - len(truth))),
        suffix_length=float(len(truth)),
    )
