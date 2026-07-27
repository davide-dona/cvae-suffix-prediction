from src.configs.schema import EarlyStoppingConfig


class EarlyStopper:
    """
    Stop training once its validation loss has stopped improving.
    Only epochs the caller marks comparable are counted: while the KL weight is still
    ramping, the validation loss is not comparable from one epoch to the next.
    Parameters:
        config: Early stopping configuration.
    """

    def __init__(self, config: EarlyStoppingConfig):
        self.patience = config.patience
        self.min_delta_perc = config.min_delta_perc
        self.counter = 0
        self.min_validation_loss = float('inf')

    def update(self, val_loss: float, comparable: bool) -> bool:
        """Record one epoch's validation result and report whether training should stop."""
        # An epoch whose loss cannot be compared tells us nothing about a plateau
        if not comparable:
            self.counter = 0
            return False

        # If the validation loss has improved, reset the counter and update the minimum
        if val_loss < self.min_validation_loss:
            self.min_validation_loss = val_loss
            self.counter = 0
        # Otherwise, if it has not improved by at least the minimum delta, increment it
        elif val_loss > self.min_validation_loss * (1.0 + self.min_delta_perc):
            self.counter += 1

        return self.counter >= self.patience
