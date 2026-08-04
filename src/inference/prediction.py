from dataclasses import dataclass

from src.datasets.codec import DecodedSequence


@dataclass(frozen=True)
class PrefixPrediction:
    """The model's answer for one prefix: the suffixes it drew, the one it answers with, and the
    ground truth beside them.

    The one shape everything downstream reads. A training curve, the generations file and the final
    report are all scored off this, so they measure the same thing rather than agreeing by
    coincidence. `generate_predictions` builds it from the model; `src/evaluation` builds it back
    out of a generations file, which is only this written down.

    Identity-free by design: which case and cut point this is stays with `PairInfo`, since the two
    callers that need it already hold it and the training pass does not need it at all.
    """
    samples: list[DecodedSequence]  # one per draw of z
    point: DecodedSequence          # the suffix written from the mean of `p(z | prefix)`
    truth: DecodedSequence
