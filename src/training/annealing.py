# Cyclical annealing: https://github.com/haofuml/cyclical_annealing
def cyclical_linear_weight(
    step: int,
    *,
    period_steps: int,
    ratio: float,
    start: float,
    stop: float,
) -> float:
    """
    The KL weight one optimizer step is given under a cyclical annealing schedule.

    Every `period_steps` steps the weight ramps linearly from `start` to `stop` over the first
    `ratio` of the cycle, then holds at the top for the rest of it, so a run repeatedly gives
    the posterior a stretch of cheap latent capacity and then pays for it again.

    The period is a length rather than a fraction of the step budget: `training.max_steps` is a
    ceiling a run rarely reaches, and deriving the cycle from it would mean that changing how
    long a run is allowed to go on for silently reshapes what it optimizes. `ratio < 1` is what
    guarantees every cycle reaches `stop` at all, and `LossConfig` asserts it.

    The schedule is a function of the step rather than a precomputed array because it is read
    once per step and a run's budget runs to five figures.

    Args:
        step: The step to weight, counted from 0.
        period_steps: Length of one cycle, in optimizer steps.
        ratio: Fraction of each cycle spent ramping up, the rest being held at the top.
        start: Weight every cycle ramps up from.
        stop: Weight every cycle ramps up to, and is held at for the rest of the cycle.
    Returns:
        The weight to apply at `step`.
    """
    ramp_steps = period_steps * ratio
    position = step % period_steps  # how far into its own cycle this step is

    if position >= ramp_steps:
        return stop
    return start + (stop - start) * (position / ramp_steps)
