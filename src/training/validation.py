from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
import wandb
from torch.utils.data import DataLoader

from src.datasets.codec import DatasetCodec
from src.evaluation.scores import AccuracyScores, ConformanceScores, DistributionScores
from src.inference.generate import generate_batch
from src.logs import ContinuationIndex
from src.logs.declare import ConformanceChecker
from src.scalar_metrics import Owner
from src.suffixes import ActivityCodes
from src.training.kl import LatentMetrics
from src.training.loss import Loss
from src.visualization.catalogue import TABLES

if TYPE_CHECKING:
    from src.model import SuffixModel

# Which report table, if any, answers with each metric, read off the same catalogue a figure or a
# table is composed from: the wandb chart a run is watched on and the paper's own tables read the
# same grouping by construction. A metric no table holds (a diagnostic, e.g. `sample_diversity` or
# the `_ae_mean` columns) falls into a namespace of its own instead.
_TABLE_OF_METRIC = {entry.key: table.name for table in TABLES for entry in table.columns}
_DIAGNOSTICS = 'diagnostics'
# Where the distributional scores read over the prefixes a report reads them on are logged, so
# the criterion a checkpoint is selected on and the one a report holds can be read against each
# other over a run without either being renamed.
_COMPARABLE = 'comparable'


@dataclass(frozen=True, slots=True)
class GenerationMetrics:
    """What one generation pass measured: how close the suffixes are to the ground truth, whether
    they are traces the process allows, and how the set of them compares against every
    continuation the validation split took.

    The three families a training run reads. Accuracy is the curve a run is watched on, and
    `DistributionScores.emsc` is the score a checkpoint is selected on.

    The distributional family is carried twice: over every prefix, which is what a checkpoint is
    selected on, and over the prefixes the log ran often enough for what followed them to be a
    distribution, which is what a report holds. The two are the same scores over two populations,
    so a run shows whether the checkpoint it kept is the one the reported criterion would have
    kept.
    """

    accuracy: AccuracyScores
    conformance: ConformanceScores
    distribution: DistributionScores
    # The same as `distribution`, over the prefixes `DistributionScores.comparable` admits
    comparable: DistributionScores

    def log(self, step: int) -> None:
        """Log every model-owned score to the active W&B run, namespaced by the report table it
        answers (`accuracy-point`, `fidelity`, `generative-accuracy`, `calibration`) or
        `diagnostics` for the ones no table holds.

        A `Owner.LOG` field (e.g. `suffix_length`, `reference_diversity`) is a property of the
        fixed slice this run generates for, constant across every validation of one run, so it is
        dropped here rather than logged as a flat line: it already sits in every report and
        per-prefix file.

        Args:
            step: The training step this pass scores.
        """
        # Conformance has no report table of its own (`visualization.md`: a whole-split mean says
        # nothing a reader can act on there, so it is drawn by length instead), but it is still one
        # of the two goals a run is judged on, so it keeps a namespace of its own rather than
        # falling into `diagnostics` beside unrelated per-run diagnostics. The comparable scores
        # keep one for the same reason a table would: they are the same metrics over a different
        # population and would otherwise overwrite the ones above them.
        families = (
            (self.accuracy, None),
            (self.conformance, 'conformance'),
            (self.distribution, None),
            (self.comparable, _COMPARABLE),
        )
        payload = {}
        for family, table_namespace in families:
            for declaration in type(family).metrics():
                if declaration.owner is Owner.LOG:
                    continue
                namespace = table_namespace or _TABLE_OF_METRIC.get(declaration.key, _DIAGNOSTICS)
                payload[f'{namespace}/{declaration.key}'] = getattr(family, declaration.key)
        wandb.log(payload, step=step)


@torch.no_grad()
def validate(
    model: SuffixModel, loader: DataLoader, *, step: int, device: torch.device
) -> tuple[Loss, LatentMetrics | None]:
    """
    Run one pass over `loader` without learning from it.
    Args:
        model: The model to evaluate. Put in evaluation mode here, and left in it.
        loader: The dataloader to iterate over. Its batches are `SplitTrace`s.
        step: The training step this pass scores, for a model whose loss anneals a term over
            the run.
        device: The device to run the computations on.
    Returns:
        The loss terms of the pass and what the latent carried, both averaged over the traces
        of the split. The latter is None where the model has no latent.
    """
    model.eval()

    totals = Loss()
    latent_totals: LatentMetrics | None = None
    for batch in loader:
        batch = batch.to(device)
        output = model(batch)
        _, metrics, latent = model.compute_loss(output, batch, step=step)
        totals += metrics
        if latent is not None:
            latent_totals = latent if latent_totals is None else latent_totals + latent

    traces = len(loader.dataset)
    return totals / traces, None if latent_totals is None else latent_totals / traces


@torch.no_grad()
def validate_generation(
    model: SuffixModel,
    loader: DataLoader,
    *,
    num_samples: int,
    codec: DatasetCodec,
    index: ContinuationIndex,
    checker: ConformanceChecker,
    device: torch.device,
) -> GenerationMetrics:
    """
    Generate suffixes from the prefixes in `loader` and compare them to the ground truth, to the
    declarative model, and to every continuation the split was observed to take.

    Scored through the same three families the final report is built from, and each prefix is
    answered with the same number of suffixes. What differs is which split is read and how much of
    it, so a training curve sits on a report's scale without being a report's number.

    The distributional family is returned twice, over every prefix and over the prefixes a report
    reads it on. The checkpoint is selected on the first, which is what every checkpoint in
    `outputs/` was selected on; the second is logged beside it so a run says whether the two
    criteria would have kept the same step, which is what deciding to move the selection needs and
    a report cannot show.

    Args:
        model: The model to evaluate. Put in evaluation mode here, and left in it.
        loader: The prefixes to generate for, from a `TraceDataset`.
        num_samples: Suffixes to draw per prefix. The spread across them is what
            `sample_diversity` measures, and `generate` puts `len(batch) * num_samples` rows
            through the decoder at once, so it is also what the caller sizes its batches by.
        codec: The codec the split was encoded through, read here to put the
            generations back into the log's own units. Passed rather than read off
            `loader.dataset`, which is a `Subset` wherever the split is bigger than the slice
            validated on.
        index: The continuations the validation split takes after each of its prefixes. The
            validation split's and never the test split's: selecting a checkpoint against the
            test split's continuations would fold the held-out set into what gets kept.
        checker: The declarative model to check generated suffixes against.
        device: The device to run the computations on.
    Returns:
        The metrics of the pass, averaged over prefixes.
    """
    model.eval()

    # The same codebook the index was seeded from, so a generated suffix is spelled the way the
    # continuations it is scored against are and nothing is translated per prefix.
    codes = ActivityCodes.of(codec.activity.names)

    generations = [
        generation
        for batch in loader
        for generation in generate_batch(
            model=model,
            batch=batch.to(device),
            num_samples=num_samples,
            codec=codec,
            codes=codes,
        )
    ]
    distribution = [DistributionScores.of(one, index=index) for one in generations]
    return GenerationMetrics(
        accuracy=AccuracyScores.mean([AccuracyScores.of(one) for one in generations]),
        conformance=ConformanceScores.mean(
            [ConformanceScores.of(one, checker=checker) for one in generations]
        ),
        distribution=DistributionScores.mean(distribution),
        comparable=DistributionScores.mean([one for one in distribution if one.comparable]),
    )
