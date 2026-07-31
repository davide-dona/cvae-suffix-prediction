from src.configs.schema import EarlyStoppingConfig


class EarlyStopper:
    """
    Stop training once its validation loss has stopped improving.

    Every validation counts. What it is handed is the prior-path loss, which carries no KL
    term and so does not move with the annealing weight, meaning any two validations of a run
    are comparable with each other.

    Parameters:
        config: Early stopping configuration.
    """

    def __init__(self, config: EarlyStoppingConfig):
        self.patience = config.patience
        self.min_delta_perc = config.min_delta_perc
        self.counter = 0
        self.min_validation_loss = float('inf')

    def update(self, val_loss: float) -> bool:
        """Record one validation result and report whether training should stop."""
        # If the validation loss has improved, reset the counter and update the minimum
        if val_loss < self.min_validation_loss:
            self.min_validation_loss = val_loss
            self.counter = 0
        # Otherwise, if it has not improved by at least the minimum delta, increment it
        elif val_loss > self.min_validation_loss * (1.0 + self.min_delta_perc):
            self.counter += 1

        return self.counter >= self.patience
