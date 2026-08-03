from src.configs.schema import EarlyStoppingConfig


class EarlyStopper:
    """
    Stop training once its validation loss has stopped improving. Every validation counts.
    What it is handed is the run's selection score, which is derived from free-running generation,
    making it independent from the KL annealing weights.

    Parameters:
        config: Early stopping configuration.
    """

    def __init__(self, config: EarlyStoppingConfig):
        self.patience = config.patience
        self.min_delta_perc = config.min_delta_perc
        self.counter = 0
        self.min_validation_score = float('inf')

    def update(self, val_score: float) -> bool:
        """Record one validation result and report whether training should stop."""
        # A real improvement resets the counter; anything else, plateaus included, spends it.
        if val_score < self.min_validation_score * (1.0 - self.min_delta_perc):
            self.counter = 0
        else:
            self.counter += 1

        # The best is tracked separately from what resets the counter, so a run of improvements
        # too small to count is still measured against the best of them rather than the first.
        self.min_validation_score = min(self.min_validation_score, val_score)

        return self.counter >= self.patience
