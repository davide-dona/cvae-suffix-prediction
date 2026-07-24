import os
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Union

from src.configs.schema import EarlyStoppingConfig


@dataclass
class _StopperState:
    """The state tracked across a single training run for early stopping purposes."""
    counter: int = 0
    min_validation_loss: float = float('inf')


class EarlyStopper:
    """
    Stop training once its validation loss has stopped improving.
    Only epochs evaluating the full loss function are considered: while the KL term is
    being annealed the validation loss is not comparable from one epoch to the next.

    Parameters:
        config: Early stopping configuration.
        full_kl_weight: The weight of the KL term when it is fully applied.
        output_dir: The directory to write debug information to.
    """
    def __init__(
        self,
        config: EarlyStoppingConfig,
        full_kl_weight: float,
        output_dir: Union[str, Path],
    ):
        self.patience = config.patience
        self.min_delta_perc = config.min_delta_perc
        self.full_kl_weight = full_kl_weight
        self.output_dir = output_dir
        self.debug = config.debug

        if self.debug:
            self.debug_file_path = os.path.join(self.output_dir, 'early-stop-debug.txt')

        self.state = _StopperState()

    def update(self, epoch: int, val_loss: float, w_kl: float) -> bool:
        """Record one epoch's validation result and report whether training should stop."""
        # If the KL term is not fully applied, reset the counter and do not stop
        if not math.isclose(w_kl, self.full_kl_weight):
            self.state.counter = 0
            should_stop = False
        else:
            # If the validation loss has improved, reset the counter and update the minimum
            if val_loss < self.state.min_validation_loss:
                self.state.min_validation_loss = val_loss
                self.state.counter = 0
            # Otherwise, if it has not improved by at least the minimum delta, increment it
            elif val_loss > (self.state.min_validation_loss + self.min_delta_perc * self.state.min_validation_loss):
                self.state.counter += 1

            should_stop = self.state.counter >= self.patience
            if should_stop:
                print(f'Early stopping: validation loss has not improved for {self.patience} epochs')

        if self.debug:
            with open(self.debug_file_path, mode='a') as f:
                print(
                    f'[{epoch}] EarlyStopper: w_kl={w_kl:.2f}, val_loss={val_loss:.2f}; '
                    f'counter={self.state.counter}, min_delta_perc={self.min_delta_perc}, '
                    f'min_delta={(self.state.min_validation_loss * self.min_delta_perc):.2f}, '
                    f'should_stop={should_stop}',
                    file=f,
                )

        return should_stop
