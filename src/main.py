from datetime import datetime

import pandas as pd

from create_datasets import import_tables
from model_prep import prepare
from models import logreg
from utils import build_codemap, convert_icd9, read_table

inp_folder = "../data/unzipped_files"


df_icustays, df_patients, df_MICROBIOLOGY, df_diagnosis, df_procedures = import_tables(inp_folder)


### TODO convert_icd9 on diagnosis and procedures table
df_diagnosis["FEATURE_ID"] = df_diagnosis["ICD9_CODE"].apply(convert_icd9)
# print(df_diagnosis.head())


df_procedures["FEATURE_ID"] = df_procedures["ICD9_CODE"].apply(convert_icd9)
df_procedures = df_procedures[["SUBJECT_ID", "HADM_ID", "FEATURE_ID"]]
# print(df_procedures.head())

### TODO split sepsis from diagnosis table
filter_condition1 = df_diagnosis["ICD9_CODE"] == "99592"
filter_condition2 = df_diagnosis["ICD9_CODE"] == "99591"

# print("ORIGINAL DIAGNOSIS TABLE:",df_diagnosis.count())

df_sepsis = df_diagnosis[filter_condition1 | filter_condition2]
# print("SEPSIS TABLE:",df_sepsis.count())

df_diagnosis = df_diagnosis[~(filter_condition1 | filter_condition2)]
df_diagnosis = df_diagnosis[["SUBJECT_ID", "HADM_ID", "FEATURE_ID"]]
# print("DIAGNOSIS TABLE:",df_diagnosis.count())


### TODO Append all tables df_MICROBIOLOGY, df_labevents, df_diagnosis, df_procedures
### MAKE sure they all have four fields 'SUBJECT_ID','HADM_ID','ITEMID' OR ICD CODE,'CHARTTIME'
df_MICROBIOLOGY = df_MICROBIOLOGY.rename(columns={"SPEC_ITEMID": "FEATURE_ID"})[
    ["SUBJECT_ID", "HADM_ID", "FEATURE_ID"]
]

list_of_dfs = [df_diagnosis, df_procedures, df_MICROBIOLOGY]

df_all_events = pd.concat(list_of_dfs)
# print("df_MICROBIOLOGY TABLE:",df_MICROBIOLOGY.count())
# print("df_procedures TABLE:",df_procedures.count())
# print("DIAGNOSIS TABLE:",df_diagnosis.count())
# print("all TABLE:",df_all_events.count())


### ERIMA TODO ::: ADD LABEVENTS

df_all_events = df_all_events.dropna()
### TODO build codemap for all ITEMID + features
codemap = build_codemap(df_all_events["FEATURE_ID"])

# del codemap['nan']

# df_all_events = df_all_events[~(df_all_events['FEATURE_ID']=='nan')]
df_all_events = df_all_events.dropna()

df_all_events["FEATURE_ID2"] = df_all_events["FEATURE_ID"].map(codemap)

df_all_events["FEATURE_ID2"] = df_all_events["FEATURE_ID2"].astype("Int64")
df_all_events = df_all_events.dropna()

# TODO: 4. Group the visits for the same patient and admission
# df_all_events = df_all_events.sort_values(by=['SUBJECT_ID', 'ADMITTIME']) ## LATER SEQUENCE IF WE FIND A WAY

df_all_events2 = (
    df_all_events.groupby(["SUBJECT_ID", "HADM_ID"])["FEATURE_ID2"].apply(list).reset_index()
)
# df_all_events2 = df_all_events2.groupby(['SUBJECT_ID'])["FEATURE_ID2"].apply(list).reset_index()              ### GROUP ON PATIENTS to get LIST of LISTS
# print(df_all_events2.head())

### TODO MERGE sepsis at hadmid
df_sepsis["SEPSIS"] = 1
# print(df_sepsis.head())

df_all_events2 = df_all_events2.merge(
    df_sepsis[["SUBJECT_ID", "HADM_ID", "SEPSIS"]],
    how="left",
    left_on=["SUBJECT_ID", "HADM_ID"],
    # left_on = ['SUBJECT_ID'],
    right_on=["SUBJECT_ID", "HADM_ID"],
)
# right_on = ['SUBJECT_ID'])


df_all_events2["SEPSIS"] = df_all_events2["SEPSIS"].fillna(0)
df_all_events2 = df_all_events2.dropna()

print(df_all_events2.head())
# X = [[48, 49, 50, 265, 1213]]
X = list(df_all_events2["FEATURE_ID2"])
# y = [0]
y = list(df_all_events2["SEPSIS"].astype("Int64"))

length_features = len(codemap)
max_value = max(df_all_events2["FEATURE_ID2"])
print(length_features)
print(max_value)
train_input = prepare(X, length_features)
logreg(train_input, y)

# patient124 = df_all_events2[df_all_events2['SUBJECT_ID']==124]
# print(list(patient124['FEATURE_ID2']))
# print(df_all_events2.head())
# sepsis_df_after = df_all_events2[df_all_events2['SEPSIS']==1]
# featureID2_sepsis_list = list(sepsis_df_after['FEATURE_ID2'])

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
