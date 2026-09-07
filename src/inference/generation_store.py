from collections.abc import Sequence
from pathlib import Path
from types import TracebackType
from typing import Self

import pyarrow as pa
import pyarrow.parquet as pq

from src.configs.schema import SamplingConfig
from src.identity import RunIdentity, read_run_identity, read_vocabulary, stamped, with_vocabulary
from src.inference.generation import DecodedEvents, Draws, Generation

# Which prefix a row answers, and so what the rows of two runs of one log are matched on. A cut is
# a case and a length, and the pair is unique within a file.
type PrefixKey = tuple[str, int]

# How the activity head was read, for a file whose model drew from it, and absent for one whose
# model read it at its mode. A run's identity does not settle this: the sampler is chosen after
# training and can be changed without the weights moving, so two files of one run are told apart
# by nothing else.
_SAMPLING = b'sampling'

# One run of activities, one character each. A suffix is a string rather than a list of names: the
# names live once in the file's vocabulary metadata, and an edit distance reads a string directly.
_SUFFIX = pa.string()

# One run's cycle time before each of its activities. Timestamps are these accumulated, so they are
# not written a second time. float32 because that is what the model emits: `denormalize` widens to
# float64 on the way out, and storing that width would double the largest column of the file to
# carry digits the decoder never produced.
_CYCLE_TIMES = pa.list_(pa.field(name='element', type=pa.float32()))

# The schema of the Parquet file that holds a run's generations. One row per prefix: the samples
# nest inside it, so nothing describing the prefix is written once per sample.
_SCHEMA = pa.schema(
    [
        ('case_id', pa.large_string()),
        ('prefix_len', pa.int64()),
        # The events before the cut, which a constraint over the whole trace is checked against.
        ('prefix_activities', _SUFFIX),
        # The distinct suffixes drawn for this prefix, each written once however many draws landed
        # on it, and which of them each draw took, in the order they were drawn. The decoder is
        # deterministic given `z`, so a repeated suffix is one answer the model gave twice;
        # `hit_rate_at_k` reads the first k of `generated_draws`, and a mean over the draws is the
        # weighted mean over the distinct suffixes.
        ('generated_suffixes', pa.list_(pa.field(name='element', type=_SUFFIX))),
        ('generated_draws', pa.list_(pa.field(name='element', type=pa.int16()))),
        # Still one entry per draw, in draw order. Two draws of one suffix came from different `z`
        # and the decoder wrote each its own times, so these do not fold the way the activities do.
        (
            'generated_cycle_time_minutes',
            pa.list_(pa.field(name='element', type=_CYCLE_TIMES)),
        ),
        (
            'generated_remaining_time_minutes',
            pa.list_(pa.field(name='element', type=pa.float32())),
        ),
        # The suffix written from the mean of `p(z | prefix)`: the model's single answer, drawn once
        # per prefix and the only column comparable against a model that does not sample.
        ('point_activities', _SUFFIX),
        ('point_cycle_time_minutes', _CYCLE_TIMES),
        ('point_remaining_time_minutes', pa.float32()),
        ('true_activities', _SUFFIX),
        ('true_cycle_time_minutes', _CYCLE_TIMES),
        ('true_remaining_time_minutes', pa.float32()),
    ]
)

# The two columns that identify a prefix, which is all `prefix_keys` reads.
_KEY_COLUMNS = ['case_id', 'prefix_len']

_COMPRESSION = 'zstd'

# Worth the write time: generation is GPU-bound over hours, where the whole file costs under a
# minute more to compress at this level and comes out a tenth smaller than at the default.
_COMPRESSION_LEVEL = 9

# The float columns, named as Parquet names their leaves. Byte-stream-split splits a float into its
# four byte planes before compressing, so the exponents of a column line up and zstd has something
# repetitive to find; on the cycle times, which are continuous and share nothing as whole values, it
# is the difference between compressing and not.
_FLOAT_LEAVES = [
    'generated_cycle_time_minutes.list.element.list.element',
    'generated_remaining_time_minutes.list.element',
    'point_cycle_time_minutes.list.element',
    'point_remaining_time_minutes',
    'true_cycle_time_minutes.list.element',
    'true_remaining_time_minutes',
]


class GenerationWriter:
    """A generations file, open for writing, one block per batch.

    Used as a context manager: the file's footer is written when it closes, so a run that dies
    mid-generation leaves nothing readable rather than a file that lies about its length.
    """

    def __init__(
        self,
        path: Path,
        run: RunIdentity,
        *,
        vocabulary: Sequence[str],
        sampling: SamplingConfig | None,
    ) -> None:
        """
        Args:
            path: The file to write, its directory already made, from `paths.GENERATIONS.prepare`.
                Overwritten if it already exists.
            run: The run these generations come from, stamped into the file so evaluation can read
                it back instead of guessing at it.
            vocabulary: The activity names the suffixes are spelled on, in code order, from
                `ActivityCodes.vocabulary`. Written into the file so it says what its own
                characters mean.
            sampling: How the activity head was read, for a model that draws from it, or None for
                one that reads it at its mode. Written in for the same reason the vocabulary is:
                the sampler is chosen after training, so the run's identity alone does not say
                which one produced this file.
        """
        schema = with_vocabulary(_SCHEMA, vocabulary)
        if sampling is not None:
            schema = schema.with_metadata(
                (schema.metadata or {}) | {_SAMPLING: sampling.model_dump_json()}
            )
        schema = stamped(schema, run)
        self._writer = pq.ParquetWriter(
            where=path,
            schema=schema,
            compression=_COMPRESSION,
            compression_level=_COMPRESSION_LEVEL,
            # Dictionary encoding takes precedence over byte-stream-split wherever it is left on,
            # and a column of continuous cycle times has no dictionary worth building, so the two
            # are set together.
            use_dictionary=False,
            use_byte_stream_split=_FLOAT_LEAVES,
        )

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self._writer.close()

    def write(self, generations: list[Generation]) -> None:
        """Write one batch's generations as one block of the file, one row per prefix.

        Args:
            generations: The model's answers, in the order `generate_batch` returned them.
        """
        # Dicts here because they are what arrow's constructor takes, keyed by the schema's own
        # names.
        rows = [
            {
                'case_id': generation.case_id,
                'prefix_len': generation.prefix_len,
                'prefix_activities': generation.prefix_activities,
                'generated_suffixes': list(generation.samples.suffixes),
                'generated_draws': list(generation.samples.taken),
                'generated_cycle_time_minutes': [
                    events.cycle_time_minutes for events in generation.samples.events
                ],
                'generated_remaining_time_minutes': [
                    events.remaining_time_minutes for events in generation.samples.events
                ],
                'point_activities': generation.point.activities,
                'point_cycle_time_minutes': generation.point.cycle_time_minutes,
                'point_remaining_time_minutes': generation.point.remaining_time_minutes,
                'true_activities': generation.truth.activities,
                'true_cycle_time_minutes': generation.truth.cycle_time_minutes,
                'true_remaining_time_minutes': generation.truth.remaining_time_minutes,
            }
            for generation in generations
        ]
        self._writer.write_table(table=pa.Table.from_pylist(mapping=rows, schema=_SCHEMA))


class Generations:
    """A generations file, open for reading.

    Used as a context manager, and opened once per process that reads it: the scoring pool gives
    each of its workers its own, since a block is decoded where it is scored and nothing but the
    handful of floats it reduces to comes back.
    """

    def __init__(self, path: Path) -> None:
        """
        Args:
            path: The generations file to read, from `python -m pipelines.generate`.
        """
        self._parquet = pq.ParquetFile(path)

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self._parquet.close()

    @property
    def run(self) -> RunIdentity:
        """Which run wrote this file, which is what its report is named after.

        Raises:
            ValueError: If the file carries no identity, and so predates the one it should name
                itself by.
        """
        return read_run_identity(self._parquet)

    @property
    def vocabulary(self) -> tuple[str, ...]:
        """The activity names this file spells its suffixes on, in code order.

        What `ActivityCodes.of` seeds back into a codebook, and what a reader compares against the
        continuation index's own before it scores a single prefix.

        Raises:
            ValueError: If the file carries none, and so predates the vocabulary it should name
                itself by.
        """
        return read_vocabulary(self._parquet.schema_arrow)

    @property
    def blocks(self) -> int:
        """How many blocks the file holds, which is how many units of work scoring it splits into.

        A block is what one call to `GenerationWriter.write` wrote, held as a Parquet row group. A
        prefix cannot cross one, since a row holds one, which is what makes a block an independent
        unit: what it costs to read and to score is set by the batch a run wrote rather than by the
        size of the split.
        """
        return self._parquet.num_row_groups

    @property
    def prefixes(self) -> int:
        """How many prefixes the file answers, one per row."""
        return self._parquet.metadata.num_rows

    def prefix_keys(self) -> list[PrefixKey]:
        """Which prefixes this file answers, without decoding a single suffix.

        Returns:
            The key of every row, in the order the file holds them, which is the order a walk of
            its blocks scores them in. Only the two columns that identify a prefix are read, which
            is cheap even on a file of a quarter of a million rows.
        """
        table = self._parquet.read(columns=_KEY_COLUMNS)
        return list(
            zip(
                table.column('case_id').to_pylist(),
                table.column('prefix_len').to_pylist(),
                strict=True,
            )
        )

    def block(self, block: int) -> list[Generation]:
        """Read one block of the file back.

        Args:
            block: Which block to decode, in `range(self.blocks)`.
        Returns:
            The generation for each prefix of the block, in the order they were written, exactly as
            `generate_batch` produced them. Read a column at a time: Arrow converts a whole column
            to Python in one call, and the lists that come back are the plain lists `DecodedEvents`
            promises, so nothing is copied a second time and the block itself is dropped as soon as
            this returns.
        """
        table = self._parquet.read_row_group(block)
        columns = {name: table.column(name).to_pylist() for name in table.schema.names}

        generations = []
        for position in range(table.num_rows):
            suffixes = columns['generated_suffixes'][position]
            taken = columns['generated_draws'][position]
            generations.append(
                Generation(
                    case_id=columns['case_id'][position],
                    prefix_activities=columns['prefix_activities'][position],
                    samples=Draws(
                        suffixes=tuple(suffixes),
                        taken=tuple(taken),
                        events=[
                            DecodedEvents(
                                activities=suffixes[index],
                                cycle_time_minutes=cycle_time_minutes,
                                remaining_time_minutes=remaining_time_minutes,
                            )
                            for index, cycle_time_minutes, remaining_time_minutes in zip(
                                taken,
                                columns['generated_cycle_time_minutes'][position],
                                columns['generated_remaining_time_minutes'][position],
                                strict=True,
                            )
                        ],
                    ),
                    point=DecodedEvents(
                        activities=columns['point_activities'][position],
                        cycle_time_minutes=columns['point_cycle_time_minutes'][position],
                        remaining_time_minutes=columns['point_remaining_time_minutes'][position],
                    ),
                    truth=DecodedEvents(
                        activities=columns['true_activities'][position],
                        cycle_time_minutes=columns['true_cycle_time_minutes'][position],
                        remaining_time_minutes=columns['true_remaining_time_minutes'][position],
                    ),
                )
            )
        return generations
