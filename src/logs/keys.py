# Canonical column names used throughout preprocessing, training and the
# baseline methods, following the pm4py naming convention.
CASE_KEY = 'case:concept:name'
ACTIVITY_KEY = 'concept:name'
RESOURCE_KEY = 'org:resource'
TIMESTAMP_KEY = 'time:timestamp'
LABEL_KEY = 'case:label'
# Attributes added by pipelines/preprocess.py
# Minutes since the start of the case. Read by the encoders; never predicted
CASE_OFFSET_KEY = 'relative_timestamp_from_start'
# Minutes since the previous event of the same case. Read by the encoders; never predicted
EVENT_DELTA_KEY = 'relative_timestamp_from_previous_activity'
# Minutes until the end of the case. Predicted by the decoder
REMAINING_TIME_KEY = 'remaining_time_to_case_end'

# Special TOKENS used by the encoders and decoder.
SOS_ACTIVITY = 'SOS'            # Start Of Suffix

EOT_ACTIVITY = 'EOT'            # End Of Trace Activity
EOT_RESOURCE = 'EOT-RES'        # End Of Trace Resource

PADDING_ACTIVITY = 'PAD'        # Padding Activity (Used to align sequences to the same length in a batch)
PADDING_RESOURCE = 'PAD-RES'    # Padding Resource (Used to align sequences to the same length in a batch)

UNK_ACTIVITY = 'UNK'            # Unknown Activity (Used to represent activities not seen during training)
UNK_RESOURCE = 'UNK-RES'        # Unknown Resource (Used to represent resources not seen during training)

MISSING_FEATURE = '<MISSING>'   # Value used to represent missing features in the input data


# Separator used by every raw and processed CSV log in this project.
CSV_SEPARATOR = ';'
