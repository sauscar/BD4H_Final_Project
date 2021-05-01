import os
from datetime import datetime

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from make_datasets import CreateDataset
from models import VariableRNN
from utils import calculate_num_features, evaluate, train

# from models import lightgbm, logreg

NUM_EPOCHS = 5
USE_CUDA = False
PATH_OUTPUT = "../output/"
os.makedirs(PATH_OUTPUT, exist_ok=True)


device = torch.device("cuda" if torch.cuda.is_available() and USE_CUDA else "cpu")
torch.manual_seed(1)
if device.type == "cuda":
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

inp_folder = "../data/unzipped_files"

dataset = CreateDataset(inp_folder)

(df_icustays, _, df_microbiology, df_diagnosis, _, df_labevents) = dataset.import_tables()

# roll up all events
df_all_events_by_admission = dataset.generate_all_events_by_admission(
    df_microbiology, df_labevents, df_icustays
)

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

num_features = calculate_num_features(list(df_all_events_by_admission["FEATURE_ID"]))
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
# train_losses, train_accuracies, train_recalls = [], [], []
train_losses, train_accuracies = [], []
# valid_losses, valid_accuracies, valid_recalls = [], [], []
valid_losses, valid_accuracies = [], []


for epoch in range(NUM_EPOCHS):
    # train_loss, train_accuracy, train_recall = train(model, device, train_loader, criterion, optimizer, epoch)
    # valid_loss, valid_accuracy, valid_results, valid_recall = evaluate(model, device, val_loader, criterion)

    train_loss, train_accuracy = train(model, device, train_loader, criterion, optimizer, epoch)
    valid_loss, valid_accuracy, valid_results = evaluate(model, device, val_loader, criterion)

    train_losses.append(train_loss)
    valid_losses.append(valid_loss)

    train_accuracies.append(train_accuracy)
    valid_accuracies.append(valid_accuracy)

    # train_recalls.append(train_recall)
    # valid_recalls.append(valid_recall)

    is_best = (
        valid_accuracy > best_val_acc
    )  # let's keep the model that has the best accuracy, but you can also use another metric.
    if is_best:
        best_val_acc = valid_accuracy
        torch.save(
            model,
            os.path.join(PATH_OUTPUT, "VariableRNN.pth"),
            _use_new_zipfile_serialization=False,
        )

best_model = torch.load(os.path.join(PATH_OUTPUT, "VariableRNN.pth"))


def predict_sepsis(model, device, data_loader):
    model.eval()
    probas = []
    with torch.no_grad():
        for i, (input, target) in enumerate(data_loader):
            if isinstance(input, tuple):
                input = tuple([e.to(device) if type(e) == torch.Tensor else e for e in input])
            else:
                input = input.to(device)
            output = model(input)
            y_pred = output.numpy()[0][1]
            probas.append(y_pred)

    return probas


test_prob = predict_sepsis(best_model, device, test_loader)
