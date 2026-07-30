import math


# Cyclical annealing: https://github.com/haofuml/cyclical_annealing
def cyclical_linear_weight(
    step: int,
    *,
    total_steps: int,
    n_cycles: int,
    ratio: float,
    start: float,
    stop: float,
) -> float:
    """
    The KL weight one optimizer step is given under a cyclical annealing schedule.

    Training is divided into `n_cycles` equal cycles. Each ramps the weight linearly from
    `start` to `stop` over its first `ratio`, then holds it at the top for the rest of it, so
    a run repeatedly gives the posterior a stretch of cheap latent capacity and then pays for
    it again.

    The schedule is a function of the step rather than a precomputed array because it is read
    once per step and `total_steps` runs to six figures.

    Args:
        step: The step to weight, counted from 0.
        total_steps: Length of the schedule, that is `training.max_steps`.
        n_cycles: Number of cycles to fit into `total_steps`.
        ratio: Fraction of each cycle spent ramping up, the rest being held at the top.
        start: Weight every cycle ramps up from.
        stop: Weight every cycle ramps up to, and is held at for the rest of the cycle.
    Returns:
        The weight to apply at `step`.
    """
    period = total_steps / n_cycles
    ramp_steps = period * ratio
    position = step % period  # how far into its own cycle this step is

    if position >= ramp_steps:
        return stop
    return start + (stop - start) * (position / ramp_steps)


def validate_schedule(*, total_steps: int, n_cycles: int, ratio: float, stop: float) -> None:
    """
    Check that a schedule reaches its full weight at least once, and say so if it does not.

    A schedule whose cycles are shorter than their own ramps, or shorter than a single step,
    never trains against the full KL term at all. Left unchecked that only shows up as a run
    that quietly optimized the wrong objective, so it is raised here, before the first step.

    Args:
        total_steps, n_cycles, ratio, stop: The schedule to check, as
            `cyclical_linear_weight` takes them.
    Raises:
        ValueError: If no step of the schedule reaches `stop`.
    """
    period = total_steps / n_cycles
    # A cycle holds the steps `0 .. ceil(period) - 1` of its own, and holds at `stop` from the
    # end of its ramp on, so it tops out only if its last step is past that ramp.
    if math.ceil(period) - 1 < period * ratio:
        raise ValueError(
            f'a {n_cycles}-cycle schedule over {total_steps} steps with ratio={ratio} never '
            f'reaches the full weight of {stop}: use fewer cycles, or a smaller ratio'
        )
