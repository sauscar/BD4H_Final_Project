import os

from etl import build_data_loaders
from evaluate_model import display_test_metrics
from train_model import train_rnn_model
from utils.utils import plot_confusion_matrix, plot_learning_curves

NUM_EPOCHS = 5
BATCH_SIZE = 1
USE_CUDA = False
PATH_OUTPUT = "../output/"
os.makedirs(PATH_OUTPUT, exist_ok=True)

# get train, validation and test data loaders
train_loader, val_loader, test_loader, num_features = build_data_loaders(BATCH_SIZE)

# train RNN model
best_model, train_losses, valid_loss, train_accuracies, valid_accuracies, results = train_rnn_model(
    train_loader, val_loader, num_features, NUM_EPOCHS, USE_CUDA, PATH_OUTPUT
)

# display test metrics
display_test_metrics(best_model, test_loader)

# plot learning curves
plot_learning_curves(train_losses, valid_loss, train_accuracies, valid_accuracies)

class_names = ["No Sepsis", "Sepsis"]
plot_confusion_matrix(results, class_names, model_type="RNN for Sepsis Prediction")
