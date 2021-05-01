import os
from datetime import datetime

import pandas as pd

from etl import build_data_loaders
from evaluate_model import display_test_metrics
from train_model import train_rnn_model
from utils.utils import (
    calculate_num_features,
    evaluate,
    plot_confusion_matrix,
    plot_learning_curves,
    train,
)

# get train, validation and test data loaders
train_loader, val_loader, test_loader, num_features = build_data_loaders()

# train RNN model
best_model, train_losses, valid_loss, train_accuracies, valid_accuracies = train_rnn_model(
    train_loader, val_loader, num_features
)

# display test metrics
display_test_metrics(best_model, test_loader)

# plot learning curves
plot_learning_curves(train_losses, valid_loss, train_accuracies, valid_accuracies)

# plot confusion matrix
