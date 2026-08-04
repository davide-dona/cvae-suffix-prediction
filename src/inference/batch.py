from dataclasses import dataclass
import numpy as np

from src.configs.schema import InferenceConfig
from src.datasets.dataset import SuffixItem
from src.model import TransformerCVAE


@dataclass(frozen=True)
class BatchGeneration:
    """One batch's free-running generation, moved to the CPU, with the ground truth beside it.

    `true_lengths` counts events only: the EOT closing a complete suffix is dropped, so a
    truth compares directly against a sample cut at its length.
    """
    activities: np.ndarray            # int64, [batch_size, num_samples, steps]
    lengths: np.ndarray               # int64, [batch_size, num_samples]
    remaining_time: np.ndarray        # float32, [batch_size, num_samples], normalized in [0, 1]
    point_activities: np.ndarray      # int64, [batch_size, steps]
    point_lengths: np.ndarray         # int64, [batch_size]
    point_remaining_time: np.ndarray  # float32, [batch_size], normalized in [0, 1]
    true_activities: np.ndarray       # int64, [batch_size, seq_len]
    true_lengths: np.ndarray          # int64, [batch_size], EOT dropped
    true_remaining_time: np.ndarray   # float32, [batch_size], normalized in [0, 1]

    def samples(self, index: int) -> list[list[int]]:
        """The generated activity indices of one prefix, each sample cut at its length."""
        return [
            self.activities[index, sample, : self.lengths[index, sample]].tolist()
            for sample in range(self.activities.shape[1])
        ]

    def point(self, index: int) -> list[int]:
        """The point prediction's activity indices for one prefix, cut at its length."""
        return self.point_activities[index, : self.point_lengths[index]].tolist()

    def truth(self, index: int) -> list[int]:
        """The ground-truth activity indices of one prefix, EOT dropped."""
        return self.true_activities[index, : self.true_lengths[index]].tolist()


def generation_batch_size(inference: InferenceConfig, upper_bound: int) -> int:
    """How many prefixes to hand the decoder at once, to protect its memory.

    Each prefix expands into `num_samples` rows, so the batch size is
    `generation_rows // num_samples`: the row count stays bounded however `num_samples` is
    configured. Capped at `upper_bound` and floored at 1.

    Args:
        inference: Provides `generation_rows` (the row budget) and `num_samples`.
        upper_bound: Hard ceiling on the batch size, typically `data.batch_size`.
    Returns:
        The batch size, at least 1.
    """
    return max(1, min(upper_bound, inference.generation_rows // inference.num_samples))



def generate_batch(
    model: TransformerCVAE, batch: SuffixItem, *, num_samples: int
) -> BatchGeneration:
    """Generate `num_samples` suffixes per prefix of one batch, and the point prediction beside them.

    The one call into the raw `model.generate`: everything downstream (validation scoring and
    the generations file alike) reads the generation through the lengths resolved here. The
    second pass costs one prefix's worth of decoding against `num_samples`, and it is what makes
    the report say what the model answers as well as what it draws.

    Args:
        model: The model to generate with, already in eval mode.
        batch: A batch from `SuffixDataset`, already on the model's device.
        num_samples: How many suffixes to draw per prefix.
    Returns:
        The generation, the point prediction and the ground truth, on the CPU.
    """
    generated = model.generate(item=batch, num_samples=num_samples)
    point = model.generate(item=batch, num_samples=1, sample_latent=False)

    # `suffix_len` counts the EOT closing a complete suffix, which is a marker and not an
    # event; a truncated suffix has none to drop.
    last_position = (batch.suffix_len - 1).unsqueeze(dim=1)  # [batch_size, 1]
    ends_with_eot = (
        batch.suffix.activities.gather(dim=1, index=last_position).squeeze(dim=1)
        == model.eot_activity_index
    )  # [batch_size]
    true_lengths = batch.suffix_len - ends_with_eot.long()  # [batch_size]

    return BatchGeneration(
        activities=generated.activities.cpu().numpy(),
        lengths=generated.lengths.cpu().numpy(),
        remaining_time=generated.remaining_time.cpu().numpy(),
        point_activities=point.activities.squeeze(dim=1).cpu().numpy(),
        point_lengths=point.lengths.squeeze(dim=1).cpu().numpy(),
        point_remaining_time=point.remaining_time.squeeze(dim=1).cpu().numpy(),
        true_activities=batch.suffix.activities.cpu().numpy(),
        true_lengths=true_lengths.cpu().numpy(),
        true_remaining_time=batch.remaining_time.cpu().numpy(),
    )