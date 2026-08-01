# Canonical column names used throughout preprocessing, training and the
# baseline methods, following the pm4py naming convention.
CASE_KEY = 'case:concept:name'
ACTIVITY_KEY = 'concept:name'
RESOURCE_KEY = 'org:resource'
TIMESTAMP_KEY = 'time:timestamp'
LABEL_KEY = 'case:label'

# Special tokens appended to the activity and resource vocabularies. They are part of
# the encoding, not of the data, and are identical for every dataset and every split.
EOT_ACTIVITY = 'EOT'
PADDING_ACTIVITY = 'PAD'
SOS_ACTIVITY = 'SOS'
UNK_ACTIVITY = 'UNK'
EOT_RESOURCE = 'EOT-RES'
PADDING_RESOURCE = 'PAD-RES'
SOS_RESOURCE = 'SOS-RES'
UNK_RESOURCE = 'UNK-RES'

# Case attribute added by pipelines/preprocess.py (see add_case_offset) and
# always fed to the model.
CASE_OFFSET_KEY = 'relative_timestamp_from_start'

# Minutes since the previous activity in the same case. Read by the encoders; never predicted
EVENT_DELTA_KEY = 'relative_timestamp_from_previous_activity'

# minutes from this event to the last event of its case. This is the time quantity the
# model predicts.
REMAINING_TIME_KEY = 'remaining_time_to_case_end'

# Separator used by every raw and processed CSV log in this project.
CSV_SEPARATOR = ';'
