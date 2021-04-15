
def convert_icd9(icd9_object):
	"""
	:param icd9_object: ICD-9 code (Pandas/Numpy object).
	:return: extracted main digits of ICD-9 code
	"""
	icd9_str = str(icd9_object)

	# TODO: Extract the the first 3 or 4 alphanumeric digits prior to the decimal point from a given ICD-9 code.
	# TODO: Read the homework description carefully.
	if icd9_str[0] == "E":
		converted = icd9_str[0:4]
	else:
		converted = icd9_str[0:3]
	#print(icd9_str)

	return converted


def build_codemap(df_icd9, transform):
	"""
	:return: Dict of code map {main-digits of ICD9: unique feature ID}
	"""
	# TODO: We build a code map using ONLY train data. Think about how to construct validation/test sets using this.
	#print("ORIGINAL SHAPE:",df_icd9.shape)

	df_icd9 = df_icd9.dropna()
	df_digits = df_icd9['ICD9_CODE'].apply(transform)

	uniqueICD9 = df_digits.unique()
	uniqueICD9 = uniqueICD9[~pd.isnull(uniqueICD9)]
	#print("UNIQUE SHAPE:",uniqueICD9.shape)
	#print("UNIQUE type:",type(uniqueICD9))

	codeList = list(uniqueICD9)
	indexList = list(range(0,len(codeList)))
	#indexList2 = [round(x) for x in indexList]
	#print("Code List length:",len(codeList))
	
	codemap = dict(zip(codeList,indexList))
	#print(codemap)
	return codemap


def create_dataset(path, codemap, transform):
	"""
	:param path: path to the directory contains raw files.
	:param codemap: 3-digit ICD-9 code feature map
	:param transform: e.g. convert_icd9
	:return: List(patient IDs), List(labels), Visit sequence data as a List of List of List.
	"""
	# TODO: 1. Load data from the three csv files
	# TODO: Loading the mortality file is shown as an example below. Load two other files also.
	df_mortality = pd.read_csv(os.path.join(path, "MORTALITY.csv"))
	df_admissions = pd.read_csv(os.path.join(path, "ADMISSIONS.csv"))
	df_diagnosis = pd.read_csv(os.path.join(path, "DIAGNOSES_ICD.csv"))
	
	#print(df_mortality.head(1))
	#print(df_admissions.head(5))
	#print(df_diagnosis.head(1))

	# TODO: 2. Convert diagnosis code in to unique feature ID.
	# TODO: HINT - use 'transform(convert_icd9)' you implemented and 'codemap'.
	df_diagnosis['ICD9_CODE2'] = df_diagnosis['ICD9_CODE'].apply(convert_icd9)
	df_diagnosis['FEATURE_ID'] = df_diagnosis['ICD9_CODE2'].map(codemap)
	#print("BEFORE NA:",df_diagnosis.shape)
	df_diagnosis = df_diagnosis.dropna()
	#print("after NA:",df_diagnosis.shape)
	df_diagnosis['FEATURE_ID'] = df_diagnosis['FEATURE_ID'].astype(int)
	
	#print(df_diagnosis.head(5))

	# TODO: 3. Group the diagnosis codes for the same visit.
	#df_diag_group = df_diagnosis.groupby("HADM_ID")["featureID"].count()
	df_diag_admit = df_diagnosis[['SUBJECT_ID','HADM_ID','FEATURE_ID']].merge(df_admissions[['SUBJECT_ID','HADM_ID','ADMITTIME']],
					how='left',
					left_on = ['SUBJECT_ID','HADM_ID'],
					right_on = ['SUBJECT_ID','HADM_ID'])
	#print(df_diag_admit.head(5))
	df_diag_admit2 = df_diag_admit.groupby(["SUBJECT_ID","HADM_ID","ADMITTIME"])["FEATURE_ID"].apply(list).reset_index()
	#print(df_diag_admit2.head(5))
	
	# TODO: 4. Group the visits for the same patient.
	df_diag_admit3 = df_diag_admit2.sort_values(by=['SUBJECT_ID', 'ADMITTIME'])
	df_diag_admit4 = df_diag_admit3.groupby(["SUBJECT_ID"])["FEATURE_ID"].apply(list).reset_index()
	#print(df_diag_admit4.head(5))
	
	# TODO: 5. Make a visit sequence dataset as a List of patient Lists of visit Lists
	# TODO: Visits for each patient must be sorted in chronological order.
	df_diag_admit_mort =  df_diag_admit4.merge(df_mortality[['SUBJECT_ID','MORTALITY']],
					how='left',
					left_on = ['SUBJECT_ID'],
					right_on = ['SUBJECT_ID'])

	df_diag_admit_mort['MORTALITY'].replace(np.nan,0,inplace=True)
	print(df_diag_admit_mort.shape)

	# TODO: 6. Make patient-id List and label List also.
	# TODO: The order of patients in the three List output must be consistent.
	patient_ids = df_diag_admit_mort['SUBJECT_ID'].to_list()
	labels =  df_diag_admit_mort['MORTALITY'].to_list()
	seq_data = df_diag_admit_mort['FEATURE_ID'].to_list()
	'''patient_ids = [0, 1, 2]
	labels = [1, 0, 1]
	seq_data = [[[0, 1], [2]], [[1, 3, 4], [2, 5]], [[3], [5]]]'''
	return patient_ids, labels, seq_data
