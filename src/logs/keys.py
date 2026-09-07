from enum import StrEnum


class Split(StrEnum):
    """The three splits preprocessing cuts a log into, named as their files on disk are.

    Iterating the members is what says a dataset has been preprocessed and what a pipeline
    reads, so the three names are written down here and nowhere else.
    """

    TRAIN = 'train'
    VAL = 'val'
    TEST = 'test'


# Canonical column names used throughout preprocessing, training and the
# baseline methods, following the pm4py naming convention.
CASE_KEY = 'case:concept:name'
ACTIVITY_KEY = 'concept:name'
RESOURCE_KEY = 'org:resource'
TIMESTAMP_KEY = 'time:timestamp'
# Attributes added by pipelines/preprocess.py.
# Minutes since the previous event of the same case, read through `DatasetCodec.cycle_time`
CYCLE_TIME_KEY = 'cycle_time'
# Minutes since the first event of the same case, offered to the encoders the same way
CASE_ELAPSED_KEY = 'ts_start'
# Minutes until the end of the case. Predicted by the decoder
REMAINING_TIME_KEY = 'rtime'
# The calendar position of the event, offered to the encoders the same way. Cyclical: sin/cos of
# the day of the week and of the second of the day, so the encoders read the wrap-around (Sunday
# to Monday, midnight to midnight) rather than a raw count that treats it as a jump.
DAY_SIN_KEY = 'day_sin'
DAY_COS_KEY = 'day_cos'
SECONDS_SIN_KEY = 'seconds_sin'
SECONDS_COS_KEY = 'seconds_cos'
# The lower bound of the cut points a case may be split at.
# Normally 1; a case crossing the train/test separation have its cut points narrowed to
# the first one after the separation, making the split leak-proof.
MIN_PREFIX_KEY = 'min_prefix_len'

# Special tokens used by the encoders and decoder. Each follows the vocabulary of whichever
# categorical channel carries it, so the same marker serves activities, resources and features.
SOS_TOKEN = 'SOS'  # Start Of Suffix

EOT_TOKEN = 'EOT'  # End Of Trace

PAD_TOKEN = 'PAD'  # Padding (Used to align sequences to the same length in a batch)

UNK_TOKEN = 'UNK'  # Unknown (Used to represent values not seen during training)

MISSING_FEATURE = '<MISSING>'  # Value used to represent missing features in the input data


# Separator used by every CSV log this project writes. A raw log is read with whatever
# separator it was downloaded with, declared as `data.separator`.
CSV_SEPARATOR = ';'
