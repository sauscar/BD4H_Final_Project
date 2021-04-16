import pandas as pd


def read_table(inp_folder, filename):
    path = inp_folder + "/" + filename
    df = pd.read_csv(path)
    print("******", filename)
    print("TOTAL RECORDS in ", df.shape)
    return df


def read_table_spark(spark_session, inp_folder, filename, cols):
    path = inp_folder + "/" + filename
    spark_df = spark_session.read.csv(path, header=True)
    spark_df_filtered = spark_df.select(*cols)

    print("******", filename)
    print(
        f"TOTAL RECORDS AFTER FILTER ({spark_df_filtered.count()}, {len(spark_df_filtered.columns)})"
    )

    return spark_df_filtered


def build_codemap(df_icd9):
    """
	:return: Dict of code map {main-digits of ICD9: unique feature ID}
	"""
    uniqueICD9 = df_icd9.unique()
    uniqueICD9 = uniqueICD9[~pd.isnull(uniqueICD9)]

    codeList = list(uniqueICD9)
    indexList = list(range(0, len(codeList)))

    codemap = dict(zip(codeList, indexList))
    return codemap


def convert_icd9(icd9_object):

    icd9_str = str(icd9_object)

    # TODO: Extract the the first 3 or 4 alphanumeric digits prior to the decimal point from a given ICD-9 code.
    # TODO: Read the homework description carefully.
    if icd9_str[0] == "E":
        converted = icd9_str[0:4]
    else:
        converted = icd9_str[0:3]
    # print(icd9_str)

    return converted
