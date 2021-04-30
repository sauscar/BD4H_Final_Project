import os
from datetime import datetime

import pandas as pd

from make_datasets import CreateDataset
from utils import calculate_num_features, train, evaluate
from models import VariableRNN
import torch
import torch.nn as nn
import torch.optim as optim


# from models import lightgbm, logreg

NUM_EPOCHS = 5
USE_CUDA = False


device = torch.device("cuda" if torch.cuda.is_available() and USE_CUDA else "cpu")
torch.manual_seed(1)
if device.type == "cuda":
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False

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

#### NEW STUFF MODEL TRAINING IS BELOW ####

model = VariableRNN(num_features)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())

model.to(device)
criterion.to(device)

best_val_acc = 0.0
train_losses, train_accuracies = [], []
valid_losses, valid_accuracies = [], []
for epoch in range(NUM_EPOCHS):
	train_loss, train_accuracy = train(model, device, train_loader, criterion, optimizer, epoch)
	valid_loss, valid_accuracy, valid_results = evaluate(model, device, val_loader, criterion)

	train_losses.append(train_loss)
	valid_losses.append(valid_loss)

	train_accuracies.append(train_accuracy)
	valid_accuracies.append(valid_accuracy)

	is_best = valid_accuracy > best_val_acc  # let's keep the model that has the best accuracy, but you can also use another metric.
	if is_best:
		best_val_acc = valid_accuracy
		# torch.save(model, os.path.join(PATH_OUTPUT, "MyVariableRNN.pth"), _use_new_zipfile_serialization = False)

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
