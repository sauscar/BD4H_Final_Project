import pdb
from datetime import datetime

import pandas as pd
import pyspark.sql.functions as F
from pyspark.sql import SparkSession
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, TensorDataset
from utils.utils import (
    build_codemap,
    calculate_num_features,
    create_sequence_data,
    event_collate_fn,
    read_table,
    read_table_spark,
)

inp_folder = "../data/unzipped_files"


class CreateDataset:

    BATCH_SIZE = 1
    NUM_WORKERS = 0

    def __init__(self, inp_folder):
        self.inp_folder = inp_folder

    def set_icustays(self):
        self.icustays_file = "ICUSTAYS.csv"
        self.icustays_colums = ["SUBJECT_ID", "HADM_ID", "ICUSTAY_ID", "INTIME", "OUTTIME"]

    def set_patients(self, df_icustays):
        self.patients_file = "PATIENTS.csv"
        self.unq_ICU_patients = df_icustays.select("SUBJECT_ID").distinct()
        self.patients_colums = ["SUBJECT_ID", "GENDER", "DOB", "DOD"]

    def set_microbiology(self):
        self.microbiology_file = "MICROBIOLOGYEVENTS.csv"
        self.microbiology_names = "D_ITEMS.csv"
        self.microbiology_columns = [
            "SUBJECT_ID",
            "HADM_ID",
            "CHARTDATE",
            "SPEC_ITEMID",
            # "SPEC_TYPE_DESC",
            # "ORG_ITEMID",
            # "ISOLATE_NUM",
            # "AB_ITEMID",
            # "INTERPRETATION",
        ]

    def set_labevents(self):
        self.labevents_file = "LABEVENTS.csv"
        self.labevents_columns = [
            "SUBJECT_ID",
            "HADM_ID",
            "ITEMID",
            "FLAG",
            "CHARTTIME",
        ]

    def set_diagnosis_icd(self):
        self.diagnosis_icd_file = "DIAGNOSES_ICD.csv"
        self.diagnosis_icd_columns = ["SUBJECT_ID", "HADM_ID", "SEQ_NUM", "ICD9_CODE"]

    def set_procedures(self):
        self.procedures_file = "PROCEDURES_ICD.csv"
        self.procedures_columns = ["SUBJECT_ID", "HADM_ID", "SEQ_NUM", "ICD9_CODE"]

    def import_tables(self):
        """Import data from CSV using spark"""
        spark = SparkSession.builder.appName("Sepsis_Prediction").getOrCreate()
        spark.conf.set("park.sql.execution.arrow.pyspark.enabled", "true")

        ####ICU STAYS
        self.set_icustays()
        df_icustays = read_table_spark(spark, inp_folder, self.icustays_file, self.icustays_colums)
        df_icustays = df_icustays.withColumnRenamed("INTIME", "INDEX_DATE")

        ## PATIENTS
        self.set_patients(df_icustays)
        df_patients = read_table_spark(spark, inp_folder, self.patients_file, self.patients_colums)
        df_patients = df_patients.join(
            self.unq_ICU_patients, df_patients["SUBJECT_ID"] == self.unq_ICU_patients["SUBJECT_ID"]
        )
        print("FILTERED RECORDS in ", df_patients.count())

        ### MICROBIOLOGY
        self.set_microbiology()
        microbiology_names = read_table_spark(spark, inp_folder, self.microbiology_names)
        df_microbiology = read_table_spark(
            spark, inp_folder, self.microbiology_file, self.microbiology_columns
        )
        df_microbiology = df_microbiology.join(
            self.unq_ICU_patients,
            df_microbiology["SUBJECT_ID"] == self.unq_ICU_patients["SUBJECT_ID"],
        )

        print("FILTERED RECORDS in ", df_microbiology.count())

        ### LABEVENTS
        self.set_labevents()
        df_labevents = read_table(inp_folder, self.labevents_file)
        df_labevents = df_labevents[df_labevents["FLAG"] == "abnormal"]
        print("FILTERED RECORDS in ", df_labevents.shape)

        ### DIAGNOSIS
        self.set_diagnosis_icd()
        df_diagnosis = read_table_spark(
            spark, inp_folder, self.diagnosis_icd_file, self.diagnosis_icd_columns
        )

        ### PROCEDURE
        self.set_procedures()
        df_procedures = read_table_spark(
            spark, inp_folder, self.procedures_file, self.procedures_columns
        )

        # convert to pandas DataFrame
        (df_icustays, df_patients, df_microbiology, df_diagnosis, df_procedures) = (
            df_icustays.toPandas(),
            df_patients.toPandas(),
            df_microbiology.toPandas(),
            df_diagnosis.toPandas(),
            df_procedures.toPandas(),
        )

        for df in [df_microbiology, df_diagnosis, df_labevents, df_procedures]:
            df.dropna(subset=["HADM_ID"], inplace=True)
            df["SUBJECT_ID"] = df["SUBJECT_ID"].astype(int)
            df["HADM_ID"] = df["HADM_ID"].astype(int)

        return (
            df_icustays,
            df_patients,
            df_microbiology,
            df_diagnosis,
            df_procedures,
            df_labevents,
        )

    def train_validation_test_split(self, df_all_events_by_admission, ratio=[0.8, 0.1, 0.1]):
        """perform a train, validation and test split based on the input ratio"""
        # separate sepsis and non_sepsis cases
        sepsis = df_all_events_by_admission[df_all_events_by_admission["SEPSIS"] == 1]
        non_sepsis = df_all_events_by_admission[df_all_events_by_admission["SEPSIS"] == 0]

        X_sepsis = sepsis.drop("SEPSIS", axis=1)
        Y_sepsis = sepsis["SEPSIS"]

        X_non_sepsis = non_sepsis.drop("SEPSIS", axis=1)
        Y_non_sepsis = non_sepsis["SEPSIS"]

        # split the train/val test on both sepsis and non-sepsis cases
        (X_train_val_sepsis, X_test_sepsis, Y_train_val_sepsis, Y_test_sepsis) = train_test_split(
            X_sepsis, Y_sepsis, test_size=ratio[-1], random_state=7
        )

        (
            X_train_val_no_sepsis,
            X_test_no_sepsis,
            Y_train_val_no_sepsis,
            Y_test_no_sepsis,
        ) = train_test_split(X_non_sepsis, Y_non_sepsis, test_size=ratio[-1], random_state=7)

        # collect the test set
        test_set = (
            pd.concat([X_test_sepsis, X_test_no_sepsis]),
            pd.concat([Y_test_sepsis, Y_test_no_sepsis]),
        )

        # split the train val on both sepsis and non-sepsis cases
        (X_train_sepsis, X_val_sepsis, Y_train_sepsis, Y_val_sepsis) = train_test_split(
            X_train_val_sepsis, Y_train_val_sepsis, test_size=ratio[1], random_state=7
        )

        (X_train_no_sepsis, X_val_no_sepsis, Y_train_no_sepsis, Y_val_no_sepsis) = train_test_split(
            X_train_val_no_sepsis, Y_train_val_no_sepsis, test_size=ratio[1], random_state=7
        )

        # collect the training set
        train_set = (
            pd.concat([X_train_sepsis, X_train_no_sepsis]),
            pd.concat([Y_train_sepsis, Y_train_no_sepsis]),
        )
        # collect the validation set
        validation_set = (
            pd.concat([X_val_sepsis, X_val_no_sepsis]),
            pd.concat([Y_val_sepsis, Y_val_no_sepsis]),
        )

        return train_set, validation_set, test_set

    def generate_sepsis_event(self, df_all_events_by_admission, df_diagnosis):
        """Generate sepis event"""
        # create df_sepsis
        filter_condition1 = df_diagnosis["ICD9_CODE"] == "99592"
        filter_condition2 = df_diagnosis["ICD9_CODE"] == "99591"
        filter_condition3 = df_diagnosis["ICD9_CODE"] == "78552"
        df_sepsis = df_diagnosis[filter_condition1 | filter_condition2 | filter_condition3]

        df_sepsis["HADM_ID"] = df_sepsis["HADM_ID"].astype(int)
        df_sepsis = df_sepsis.drop_duplicates(subset=["SUBJECT_ID", "HADM_ID"])
        df_sepsis["SEPSIS"] = 1
        print("NUMBER OF SEPSIS:", len(df_sepsis))

        # join df_all_events_by_admission and create 0 and 1 indicator
        df_all_events_by_admission = df_all_events_by_admission.merge(
            df_sepsis[["SUBJECT_ID", "HADM_ID", "SEPSIS"]],
            how="left",
            left_on=["SUBJECT_ID", "HADM_ID"],
            right_on=["SUBJECT_ID", "HADM_ID"],
        )
        df_all_events_by_admission["SEPSIS"] = df_all_events_by_admission["SEPSIS"].fillna(0)
        df_all_events_by_admission = df_all_events_by_admission.dropna()
        # y = list(df_all_events_by_admission["SEPSIS"].astype("Int64"))

        return df_all_events_by_admission

    def generate_all_events_by_admission(self, df_microbiology, df_labevents, df_icustays):
        """Convert to sequence events data based on 1 day window"""

        df_microbiology = df_microbiology[
            ["SUBJECT_ID", "HADM_ID", "SPEC_ITEMID", "CHARTDATE"]
        ].iloc[:, 1:]

        df_microbiology = df_microbiology.rename(
            columns={"SPEC_ITEMID": "FEATURE", "CHARTDATE": "CHARTTIME"}
        )

        df_labevents = df_labevents[["SUBJECT_ID", "HADM_ID", "CHARTTIME", "ITEMID"]]
        df_labevents = df_labevents.rename(columns={"ITEMID": "FEATURE"})

        list_of_dfs = [df_microbiology, df_labevents]

        df_all_events = pd.concat(list_of_dfs)

        # build codemap for all ITEMID + features
        self.codemap = build_codemap(df_all_events["FEATURE"])

        df_all_events["FEATURE_ID"] = df_all_events["FEATURE"].map(self.codemap)

        # if the feature_id is not in code map, drop it
        df_all_events.dropna(inplace=True)

        df_all_events["FEATURE_ID"] = df_all_events["FEATURE_ID"].astype(int)
        df_all_events["HADM_ID"] = df_all_events["HADM_ID"].astype(int)
        # first seen events
        df_first_seen = (
            pd.DataFrame(df_all_events.groupby("HADM_ID")["CHARTTIME"].min())
            .rename(columns={"CHARTTIME": "FIRST_CHARTTIME"})
            .reset_index()
        )
        df_all_events = pd.merge(df_all_events, df_first_seen, on="HADM_ID", how="left")

        # create the time seq
        df_all_events["TIME_SEQ"] = (
            pd.to_datetime(df_all_events["CHARTTIME"])
            - pd.to_datetime(df_all_events["FIRST_CHARTTIME"])
        ).dt.days

        ### START ICUSTAY STUFF
        # convert HADM in icustays to int and select relevant data
        df_all_events["CHARTTIME"] = pd.to_datetime(df_all_events["CHARTTIME"])

        df_icustays = df_icustays[["HADM_ID", "ICUSTAY_ID", "INDEX_DATE"]]
        df_icustays["INDEX_DATE"] = pd.to_datetime(df_icustays["INDEX_DATE"])
        df_icustays["HADM_ID"] = df_icustays["HADM_ID"].astype(int)
        joined = df_all_events.merge(df_icustays, on="HADM_ID", how="inner")
        all_events_filtered_icu = joined[joined["CHARTTIME"] < joined["INDEX_DATE"]]

        #### END ICUSTAY STUFF
        df_all_events = all_events_filtered_icu.sort_values(
            by=["SUBJECT_ID", "HADM_ID", "TIME_SEQ"]
        )

        # create rolled up sequence in a df column
        df_all_events_by_time_seq = (
            df_all_events.groupby(["SUBJECT_ID", "HADM_ID", "TIME_SEQ"])["FEATURE_ID"]
            .apply(list)
            .reset_index()
        )
        # convert datatype to int
        df_all_events_by_admission = (
            df_all_events_by_time_seq.groupby(["SUBJECT_ID", "HADM_ID"])["FEATURE_ID"]
            .apply(list)
            .reset_index()
        )
        df_all_events_by_admission.dropna(inplace=True)
        print("NUMBER OF ADMISSIONS:", len(df_all_events_by_admission))
        return df_all_events_by_admission

    def generate_sequence_data(self, df_all_events_by_admission):
        """
        Create sequence data based on events sequence
            Example for 5 total feautures:
            Input: [[3], [0, 2], [4, 1]]
            Ouput: [[0, 0, 0, 1, 0, 0], [1, 0, 1, 0, 0, 0], [0, 1, 0, 0, 1, 0]]
        """
        # pdb.set_trace()
        # create sequence data
        seqs = [
            create_sequence_data(seq, len(self.codemap))
            for seq in list(df_all_events_by_admission["FEATURE_ID"])
        ]
        return seqs

    def generate_torch_dataset_loaders(self, seqs, labels, num_features):
        """Generate a torch dataset object using the input sequence, labels and num_features"""
        # generate SequenceWithLabelDataset object
        dataset = SequenceWithLabelDataset(seqs, labels, num_features)
        # generate torch data_loader
        data_loader = DataLoader(
            dataset=dataset,
            batch_size=self.BATCH_SIZE,
            shuffle=True,
            collate_fn=event_collate_fn,
            num_workers=self.NUM_WORKERS,
        )
        return data_loader


class SequenceWithLabelDataset(Dataset):
    def __init__(self, seqs, labels, num_features):
        """
        Args:
                seqs (list): list of patients (list) of visits (list) of codes (int) that contains visit sequences
                labels (list): list of labels (int)
                num_features (int): number of total features available
        """

        if len(seqs) != len(labels):
            raise ValueError("Seqs and Labels have different lengths")

        self.labels = labels
        self.num_features = num_features
        self.seqs = seqs

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        # returns will be wrapped as List of Tensor(s) by DataLoader
        return self.seqs[index], self.labels[index]
