from datetime import datetime

import pandas as pd

from make_datasets import CreateDataset
from utils import calculate_num_features

# from models import lightgbm, logreg

inp_folder = "../data/unzipped_files"

dataset = CreateDataset(inp_folder)

(_, _, df_microbiology, df_diagnosis, _, df_labevents) = dataset.import_tables()

# roll up all events
df_all_events_by_admission = dataset.generate_all_events_by_admission(df_microbiology, df_labevents)

### add sepis events
df_all_events_by_admission_w_labels = dataset.generate_sepsis_event(
    df_all_events_by_admission, df_diagnosis
)

# train, validation and test split
train_set, validation_set, test_set = dataset.train_validation_test_split(
    df_all_events_by_admission_w_labels
)

# training sequence
train_seqs = dataset.generate_sequence_data(train_set[0])
val_seqs = dataset.generate_sequence_data(validation_set[0])
test_seqs = dataset.generate_sequence_data(test_set[0])

# labels
train_labels = list(train_set[1].astype(int))
val_labels = list(validation_set[1].astype(int))
test_labels = list(test_set[1].astype(int))

# number of features
num_features = calculate_num_features(list(train_set[0]["FEATURE_ID"]))

# generate torch dataset
train_loader = dataset.generate_torch_dataset_loaders(train_seqs, train_labels, num_features)
val_loader = dataset.generate_torch_dataset_loaders(val_seqs, val_labels, num_features)
test_loader = dataset.generate_torch_dataset_loaders(test_seqs, test_labels, num_features)

# import pdb

# pdb.set_trace()

# logreg(train_input, y)

# lightgbm(train_input, y)

# patient124 = df_all_events_by_admission[df_all_events_by_admission['SUBJECT_ID']==124]
# print(list(patient124['FEATURE_ID']))
# print(df_all_events_by_admission.head())
# sepsis_df_after = df_all_events_by_admission[df_all_events_by_admission['SEPSIS']==1]
# featureID2_sepsis_list = list(sepsis_df_after['FEATURE_ID'])

# TODO: 5. Make a visit sequence dataset as a List of patient Lists of visit Lists
# TODO: Visits for each patient must be sorted in chronological order.


# TODO: 6. Make patient-id List and label List also.
# TODO: The order of patients in the three List output must be consistent.

"""icu_id = df_diag_admit_mort['SUBJECT_ID'].to_list()
labels _ sepsis =  df_diag_admit_mort['MORTALITY'].to_list()
seq_data = df_diag_admit_mort['FEATURE_ID'].to_list()"""
"""patient_ids = [0, 1, 2]
labels = [1, 0, 1]
seq_data = [[[0, 1], [2]], [[1, 3, 4], [2, 5]], [[3], [5]]]"""
