from utils import read_table
from datetime import datetime
from create_datasets import import_tables
import pandas as pd
from utils import build_codemap
from utils import convert_icd9

inp_folder = '../data'



df_icustays, df_patients, df_MICROBIOLOGY, df_diagnosis, df_procedures, df_labevents = import_tables(inp_folder)

### ERIMA REMOVE BLANK HADM_IDs


### TODO convert_icd9 on diagnosis and procedures table
df_diagnosis['FEATURE_ID'] = df_diagnosis['ICD9_CODE'].apply(convert_icd9)
print(df_diagnosis.head())


df_procedures['FEATURE_ID'] = df_procedures['ICD9_CODE'].apply(convert_icd9)
df_procedures = df_procedures[['SUBJECT_ID', 'HADM_ID','FEATURE_ID']]
print(df_procedures.head())

### TODO split sepsis from diagnosis table
filter_condition1 = df_diagnosis['ICD9_CODE'] == '99592' 
filter_condition2 = df_diagnosis['ICD9_CODE'] == '99591'

print("ORIGINAL DIAGNOSIS TABLE:",df_diagnosis.count())

df_sepsis = df_diagnosis[filter_condition1|filter_condition2]
print("SEPSIS TABLE:",df_sepsis.count())

df_diagnosis = df_diagnosis[~(filter_condition1|filter_condition2)]
df_diagnosis = df_diagnosis[['SUBJECT_ID', 'HADM_ID','FEATURE_ID']]
print("DIAGNOSIS TABLE:",df_diagnosis.count())


### COUBLE CHECK SEPSIS IS NOT IN DIAGNOSIS TABLE


### TODO Append all tables df_MICROBIOLOGY, df_labevents, df_diagnosis, df_procedures
### MAKE sure they all have four fields 'SUBJECT_ID','HADM_ID','ITEMID' OR ICD CODE,'CHARTTIME'
df_MICROBIOLOGY = df_MICROBIOLOGY.rename(columns = {'SPEC_ITEMID': 'FEATURE_ID'})[['SUBJECT_ID', 'HADM_ID','FEATURE_ID']]

df_labevents = df_labevents.rename(columns = {'ITEMID': 'FEATURE_ID'})[['SUBJECT_ID', 'HADM_ID','FEATURE_ID']]

list_of_dfs = [df_diagnosis, df_procedures,df_MICROBIOLOGY,df_labevents]

df_all_events = pd.concat(list_of_dfs)
print("df_MICROBIOLOGY TABLE:",df_MICROBIOLOGY.count())
print("df_procedures TABLE:",df_procedures.count())
print("DIAGNOSIS TABLE:",df_diagnosis.count())
print("LAB EVENTS TABLE:",df_labevents.count())
print("all TABLE:",df_all_events.count())


### ERIMA TODO ::: ADD LABEVENTS


### TODO build codemap for all ITEMID + features
codemap = build_codemap(df_all_events['FEATURE_ID'])
#print(codemap)
df_all_events['FEATURE_ID2'] = df_all_events['FEATURE_ID'].map(codemap)

# TODO: 4. Group the visits for the same patient and admission
#df_all_events = df_all_events.sort_values(by=['SUBJECT_ID', 'ADMITTIME']) ## LATER SEQUENCE IF WE FIND A WAY

df_all_events2 = df_all_events.groupby(['SUBJECT_ID', 'HADM_ID'])["FEATURE_ID2"].apply(list).reset_index()
print(df_all_events2.head())

### TODO MERGE sepsis at hadmid
df_sepsis['SEPSIS'] = 1
print(df_sepsis.head())

df_all_events2 = df_all_events2.merge(df_sepsis[['SUBJECT_ID','HADM_ID','SEPSIS']],
					how='left',
					left_on = ['SUBJECT_ID','HADM_ID'],
					right_on = ['SUBJECT_ID','HADM_ID'])

df_all_events2['SEPSIS'] = df_all_events2['SEPSIS'].fillna(0)
print(df_all_events2.head())
df_all_events2



# TODO: 5. Make a visit sequence dataset as a List of patient Lists of visit Lists
# TODO: Visits for each patient must be sorted in chronological order.


# TODO: 6. Make patient-id List and label List also.
# TODO: The order of patients in the three List output must be consistent.

'''icu_id = df_diag_admit_mort['SUBJECT_ID'].to_list()
labels _ sepsis =  df_diag_admit_mort['MORTALITY'].to_list()
seq_data = df_diag_admit_mort['FEATURE_ID'].to_list()'''
'''patient_ids = [0, 1, 2]
labels = [1, 0, 1]
seq_data = [[[0, 1], [2]], [[1, 3, 4], [2, 5]], [[3], [5]]]'''