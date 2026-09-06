from dataclasses import dataclass

from src.evaluation import Axis
from src.evaluation.scores import METRICS
from src.scalar_metrics import Direction
from src.visualization.catalogue.entry import MetricEntry


@dataclass(frozen=True)
class Table:
    """One comparison table: what its file is called, which breakdown it summarizes (always the
    overall one, a table comparing runs over their whole split), the sentence its caption has to
    carry, and the columns it holds.

    A table answers one of the questions the paper asks of a model, so the columns are the metrics
    that question is settled on and no others. Rows are a block per log, one per model within it.

    The headers carry no unit. A name and a name plus `[days]` set to different heights in an
    equal-width column, which is what left the header row ragged, so the units are stated once in
    `note` and written into the caption instead of four times over the columns.
    """

    name: str
    axis: Axis
    note: str
    columns: tuple[MetricEntry, ...]

    def __post_init__(self) -> None:
        undirected = [
            entry.metric.key for entry in self.columns if entry.metric.direction is Direction.NONE
        ]
        if undirected:
            raise ValueError(
                f'the {self.name} table holds {", ".join(undirected)}, which have no better value '
                f'and so no cell a reader could rank or the emphasis could mark. A property of the '
                f"log is drawn in FIGURES as the log's own series rather than tabulated."
            )


# One table per question a mean can answer. Fidelity is one of the two goals stated for the model;
# the two accuracy tables are neither of them, but the comparison against a deterministic
# baseline (`accuracy-point`) and the check against the prefix's own ground truth
# (`accuracy-generative`) on the ground the literature already reads a model on. `calibration` asks
# of the same draws whether the spread they claim is the spread they have.
#
# Conformance, the other goal, is not tabulated: every model of every log sits within a few points
# of the log's own conformance, where a mean over the whole split says nothing a reader can act on.
# It is drawn by `conformance-by-suffix-length` against that log's own line instead, where the
# lengths and the bands say where a model really parts from the process.
#
# A point estimate beats the mean of ten stochastic draws almost by construction, so the two never
# share a column; where a table holds both, each column says which of them it is. CRPS is the one
# sample-side score that number is fair against, being exactly the absolute error where the draws
# are a single value, which is why it and the `_ae_point` columns are on one scale even though they
# sit in two tables.
#
# Every column names a metric with a direction. One without has no best value, so its cell is a
# number a reader cannot rank and the emphasis can never mark; those are the log's own values,
# which `FIGURES` draws as the log's own series, and the diagnostics of a run, which no page draws.
TABLES = (
    # The single suffix each model writes from the mean of `p(z | prefix)`, against the one the log
    # actually took.
    Table(
        name='accuracy-point',
        axis=Axis.OVERALL,
        note='Mean absolute error per prefix; length in events, times in days.',
        columns=(
            MetricEntry(METRICS['dls_point'], 'DLS'),
            MetricEntry(METRICS['length_ae_point'], 'Length'),
            MetricEntry(METRICS['remaining_time_ae_point_days'], 'Rem. time'),
            MetricEntry(METRICS['cycle_time_ae_point_days'], 'Event time'),
        ),
    ),
    # Fidelity: how close the distribution a model draws from is to the one the log continued with.
    # EMSC compares the two as stochastic languages; the three W1 columns compare the marginals
    # a suffix carries beyond its activities. The event-time column groups the cycle times by the
    # activity they precede, where its counterpart in `accuracy-point` reads them by position:
    # a position only means something once the control flow is right, which EMSC already asks.
    #
    # Precision and recall are the same comparison read in either direction and are deliberately
    # asymmetric, as `scores/distribution.py` sets them out: precision asks whether a draw is a
    # continuation the log ever took, counting a hit whether the log took it once or five hundred
    # times, where recall is weighted by how often each continuation occurred and so is the share
    # of what actually happens that the model reproduces.
    #
    # The sampled DLS is not a fidelity column: it is the expected accuracy of one draw, which a
    # model collapsed onto a mode maximizes, so it does not settle this question. It settles no
    # other either, which is why no table holds it and only `dls-by-length` draws it, against the
    # point estimate.
    Table(
        name='fidelity',
        axis=Axis.OVERALL,
        note='W1 is the 1-Wasserstein distance; length in events, times in days.',
        columns=(
            MetricEntry(METRICS['emsc'], 'EMSC'),
            MetricEntry(METRICS['continuation_precision'], 'Precision'),
            MetricEntry(METRICS['continuation_recall'], 'Recall'),
            MetricEntry(METRICS['length_wasserstein'], 'Length W1'),
            MetricEntry(METRICS['remaining_time_wasserstein_days'], 'Rem. time W1'),
            MetricEntry(METRICS['activity_time_wasserstein_days'], 'Event time W1'),
        ),
    ),
    # What the draws themselves deliver against the one continuation the prefix actually took,
    # rather than against the distribution `fidelity` reads them on or through the single answer
    # `accuracy-point` writes. Hit rate is exact-match, so it does not overlap `accuracy-point`'s
    # edit distance; CRPS is the same samples read on the three marginals a suffix carries beyond
    # its activities.
    #
    # The sampled DLS is not here. It is `E[d(draw, truth)]`, the expected accuracy of one draw,
    # which a model collapsed onto a mode maximizes, so a mean distance to one observation is the
    # wrong shape of question for a set of draws; `dls-by-length` still draws it against the point
    # estimate as a diagnostic. CRPS is that question asked properly, and being exactly the absolute
    # error for a single draw it is the one sample-side score that reads on the same scale as its
    # `accuracy-point` counterpart.
    Table(
        name='accuracy-generative',
        axis=Axis.OVERALL,
        note='Hit rate@k is the share of prefixes whose first k samples include the exact ground '
        'truth suffix, @any the share whose samples include it at all, and hit share the share of '
        "a prefix's samples that were exactly it. CRPS is the continuous ranked probability score "
        'of the samples; length in events, times in days.',
        columns=(
            MetricEntry(METRICS['hit_rate_at_1'], 'Hit rate@1'),
            MetricEntry(METRICS['hit_rate_at_5'], 'Hit rate@5'),
            MetricEntry(METRICS['hit_rate_at_10'], 'Hit rate@10'),
            MetricEntry(METRICS['hit_rate_any'], 'Hit rate@any'),
            MetricEntry(METRICS['hit_share'], 'Hit share'),
            MetricEntry(METRICS['length_crps'], 'Length CRPS'),
            MetricEntry(METRICS['remaining_time_crps_days'], 'Rem. time CRPS'),
            MetricEntry(METRICS['cycle_time_crps_days'], 'Event time CRPS'),
        ),
    ),
    # Whether the spread the samples claim is honest, which no other table asks:
    # `accuracy-generative` says how good the draws are and `fidelity` how close their distribution
    # is to the log's, but neither says whether a model's own interval means what it says.
    #
    # A cell is the empirical coverage less the level it was taken at, so 0 is a calibrated model
    # and the sign says which way a miss goes: negative is over-confident, its interval too narrow,
    # positive over-dispersed. Read as a gap rather than as a share because a coverage's best value
    # is its nominal level rather than 1.0, which no direction but `ZERO` can express and which is
    # what lets the emphasis mark the model closest to calibrated.
    #
    # An interval read off 100 draws is narrower than the one it estimates, so a calibrated model
    # lands a point or two below 0 rather than on it (`scores/accuracy.py` measures how far). The
    # bias follows the draw count, which every model here shares, so the column ranks models even
    # where its zero is not quite the zero.
    #
    # Three levels rather than one because a gap lives in `[-level, 1 - level]` and so is not
    # symmetric about what it can show: at 95 an over-dispersed model has 0.05 of room and an
    # over-confident one 0.95, and at 50 the two are even. The high levels are what resolve a model
    # whose intervals are too narrow, the low ones a model whose intervals are too wide.
    Table(
        name='calibration',
        axis=Axis.OVERALL,
        note="Each cell is the empirical coverage of the samples' central interval less the level "
        'it covers, so 0 is calibrated, negative over-confident and positive over-dispersed. An '
        'interval read off a finite sample is narrow, which costs a calibrated model a point or '
        'two on every column.',
        columns=(
            MetricEntry(METRICS['length_coverage_gap_50'], 'Len.@50'),
            MetricEntry(METRICS['length_coverage_gap_75'], 'Len.@75'),
            MetricEntry(METRICS['length_coverage_gap_95'], 'Len.@95'),
            MetricEntry(METRICS['remaining_time_coverage_gap_50'], 'Rem.@50'),
            MetricEntry(METRICS['remaining_time_coverage_gap_75'], 'Rem.@75'),
            MetricEntry(METRICS['remaining_time_coverage_gap_95'], 'Rem.@95'),
            MetricEntry(METRICS['cycle_time_coverage_gap_50'], 'Event@50'),
            MetricEntry(METRICS['cycle_time_coverage_gap_75'], 'Event@75'),
            MetricEntry(METRICS['cycle_time_coverage_gap_95'], 'Event@95'),
        ),
    ),
)
