from utils import read_table
from datetime import datetime

inp_folder = '../data/mimic-unzipped'



def import_tables(inp_folder):

    ####ICU STAYS

    filename = 'ICUSTAYS.csv'
    df_icustays = read_table(inp_folder,filename)

    unq_ICU_patients = df_icustays['SUBJECT_ID'].unique().tolist()
    df_icustays['INDEX_DATE'] = df_icustays['INTIME'] 
    df_icustays = df_icustays[['SUBJECT_ID','HADM_ID','ICUSTAY_ID','INDEX_DATE','OUTTIME']]

    ## PATIENTS
    df_patients = read_table(inp_folder,filename = 'PATIENTS.csv')
    df_patients = df_patients[df_patients['SUBJECT_ID'].isin(unq_ICU_patients)]
    df_patients = df_patients[['SUBJECT_ID','GENDER','DOB']]
    print('FILTERED RECORDS in ', df_patients.shape)

    ### MICROBIOLOGY
    df_MICROBIOLOGY = read_table(inp_folder,filename = 'MICROBIOLOGYEVENTS.csv')
    df_MICROBIOLOGY = df_MICROBIOLOGY[df_MICROBIOLOGY['SUBJECT_ID'].isin(unq_ICU_patients)]
    df_MICROBIOLOGY = df_MICROBIOLOGY[['SUBJECT_ID','HADM_ID','CHARTDATE','SPEC_ITEMID']]
    print('FILTERED RECORDS in ', df_MICROBIOLOGY.shape)

    ### LABEVENTS
    df_labevents = read_table(inp_folder,filename = 'LABEVENTS.csv')
    df_labevents = df_labevents[df_labevents['SUBJECT_ID'].isin(unq_ICU_patients)]
    df_labevents = df_labevents[df_labevents['FLAG']=='abnormal']
    df_labevents = df_labevents[['SUBJECT_ID','HADM_ID','ITEMID','CHARTTIME']]
    print('FILTERED RECORDS in ', df_labevents.shape)

    ### DIAGNOSIS
    #TODO: read diagnosis data OSCAR
    df_diagnosis = read_table(inp_folder,filename = 'DIAGNOSES_ICD.csv')
    df_diagnosis = df_diagnosis[['SUBJECT_ID','HADM_ID','SEQ_NUM','ICD9_CODE']]

    
    ### PROCEDURE
    df_procedures = read_table(inp_folder,filename = 'PROCEDURES_ICD.csv')
    df_procedures = df_procedures[['SUBJECT_ID','HADM_ID','SEQ_NUM','ICD9_CODE']]

    return df_icustays, df_patients, df_MICROBIOLOGY, df_diagnosis, df_procedures, df_labevents



