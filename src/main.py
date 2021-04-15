from utils import read_table
from datetime import datetime
from create_datasets import import_tables
import pandas as pd

inp_folder = '../data'

df_icustays, df_patients, df_MICROBIOLOGY, df_labevents, df_diagnosis, df_procedures = import_tables(inp_folder)


### TODO convert_icd9 diagnosis and procedures table

### TODO Append all tables df_MICROBIOLOGY, df_labevents, df_diagnosis, df_procedures
### MAKE sure they all have four fields 'SUBJECT_ID','HADM_ID','ITEMID' OR ICD CODE,'CHARTTIME'

### TODO build codemap for all ITEMID + features

### TODO merge the 