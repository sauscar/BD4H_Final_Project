from datetime import datetime

import pandas as pd

from make_datasets import CreateDataset

# from models import lightgbm, logreg

inp_folder = "./data/unzipped_files"

dataset = CreateDataset(inp_folder)

(
    df_icustays,
    df_patients,
    df_microbiology,
    df_diagnosis,
    df_procedures,
    df_labevents,
) = dataset.import_tables()


### TODO convert_icd9 on diagnosis and procedures table
# df_diagnosis["FEATURE"] = df_diagnosis["ICD9_CODE"].apply(convert_icd9)
# print(df_diagnosis.head())


### TODO Append all tables df_MICROBIOLOGY, df_labevents, df_diagnosis, df_procedures
### MAKE sure they all have four fields 'SUBJECT_ID','HADM_ID','ITEMID' OR ICD CODE,'CHARTTIME'

df_all_events_by_admission = dataset.generate_all_events_by_admission(df_microbiology, df_labevents)

# training sequence
train_seqs = dataset.generate_sequence_data(df_all_events_by_admission)

### sepis events
labels = dataset.generate_sepsis_event(df_all_events_by_admission, df_diagnosis)

import pdb

pdb.set_trace()

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
