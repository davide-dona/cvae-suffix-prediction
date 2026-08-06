from dataclasses import dataclass

from Declare4Py.ProcessModels.DeclareModel import DeclareModel

from src.inference import Generation
from src.logs.declare import conformance_rate
from src.metrics import ScalarMetrics, mean


@dataclass(frozen=True)
class ConformanceScores(ScalarMetrics):
    """The scores of one prefix's generated samples."""
    conformance_mean: float   # the mean over a prefix's samples
    conformance_point: float  # the suffix written from the mean of `p(z | prefix)`


def score_conformance(
    generation: Generation,
    *,
    model: DeclareModel,
    consider_vacuity: bool,
) -> ConformanceScores:
    """
    Check one prefix's generated suffixes and its point prediction.
    
    Args:
        generation: The model's answer for one prefix, decoded into the log's own units.
        model: The declarative model to check against, from `load_declare_model`.
        consider_vacuity: Whether a constraint a trace never activates counts as satisfied.
    Returns:
        The prefix's conformance. A prefix with no samples scores 0.0 on `conformance_mean`,
        the worst it can be, rather than looking perfectly conformant.
    """
    prefix = generation.prefix_activities

    def rate(suffix_activities: list[str]) -> float:
        return conformance_rate(
            prefix + suffix_activities, model=model, consider_vacuity=consider_vacuity
        )

    return ConformanceScores(
        conformance_mean=mean([rate(sample.activities) for sample in generation.samples]),
        conformance_point=rate(generation.point.activities),
    )
