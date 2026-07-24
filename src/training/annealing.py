import numpy as np


# Cyclical annealing: https://github.com/haofuml/cyclical_annealing
def cyclical_linear_weights(
    n_iter: int,
    n_cycles: int,
    ratio: float,
    start: float,
    stop: float,
) -> np.ndarray:
    """
    Build a cyclical annealing schedule, one weight per iteration.
    Every cycle ramps the weight linearly from `start` to `stop` over the first `ratio`
    of the cycle, then holds it at the top for the rest of it.
    
    Args:
        n_iter: Length of the schedule, that is the number of training epochs.
        n_cycles: Number of cycles to fit into `n_iter`.
        ratio: Fraction of each cycle spent ramping up, the rest being held at the top.
        start: Weight every cycle ramps up from.
        stop: Weight every cycle ramps up to, and is held at for the rest of the cycle.
    Returns:
        The weight to apply at each one of the `n_iter` iterations.
    Raises:
        ValueError: If the cycles are so short, or their ramps so long, that no epoch
            ever reaches the full weight.
    """
    L = np.ones(n_iter) * stop
    period = n_iter/n_cycles
    step = (stop-start)/(period*ratio) # linear schedule

    for c in range(n_cycles):
        v, i = start, 0

        while v <= stop and (int(i+c*period) < n_iter):
            L[int(i+c*period)] = v
            v += step
            i += 1

    # A schedule that never tops out leaves early stopping and best model selection with
    # nothing to compare, and the run would only fail once training is over.
    if not np.isclose(L, stop).any():
        raise ValueError(
            f'a {n_cycles}-cycle schedule over {n_iter} epochs with ratio={ratio} never '
            f'reaches the full weight of {stop}: use fewer cycles, or a smaller ratio'
        )

    return L
