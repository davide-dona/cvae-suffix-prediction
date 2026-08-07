# Canonical column names used throughout preprocessing, training and the
# baseline methods, following the pm4py naming convention.
CASE_KEY = 'case:concept:name'
ACTIVITY_KEY = 'concept:name'
RESOURCE_KEY = 'org:resource'
TIMESTAMP_KEY = 'time:timestamp'
# Attributes added by pipelines/preprocess.py.
# Minutes since the previous event of the same case. Read by the encoders; never predicted
EVENT_DELTA_KEY = 'ts_prev'
# Minutes since the first event of the same case. Read by the encoders; never predicted
CASE_ELAPSED_KEY = 'ts_start'
# Minutes until the end of the case. Predicted by the decoder
REMAINING_TIME_KEY = 'rtime'
# The calendar position of the event, offered to the encoders through `data.event_features`
# rather than as channels of their own.
DAY_IN_WEEK_KEY = 'day_in_week'  # 0 (Monday) .. 6 (Sunday)
SECONDS_IN_DAY_KEY = 'seconds_in_day'  # 0 .. 86399

# Special tokens used by the encoders and decoder. Each follows the vocabulary of whichever
# categorical channel carries it, so the same marker serves activities, resources and features.
SOS_TOKEN = 'SOS'  # Start Of Suffix

EOT_TOKEN = 'EOT'  # End Of Trace

PAD_TOKEN = 'PAD'  # Padding (Used to align sequences to the same length in a batch)

UNK_TOKEN = 'UNK'  # Unknown (Used to represent values not seen during training)

MISSING_FEATURE = '<MISSING>'  # Value used to represent missing features in the input data


# Separator used by every raw and processed CSV log in this project.
CSV_SEPARATOR = ';'
