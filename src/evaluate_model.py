import pdb

import numpy as np
import torch
import torch.nn as nn
from sklearn import metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score
from utils.utils import plot_confusion_matrix


def predict_sepsis_probabiltiies(model, device, data_loader):
    """
    Returns the predicted probability of sepsis
    Input: model, device and data_loader
    Output: predicted probability and labels as lists
    """
    model.eval()
    probas = []
    labels = []

    # set to no_grad for evaluation
    with torch.no_grad():
        # evaluate each batch
        for i, (input, label) in enumerate(data_loader):
            # put all tensors to device
            input = tuple([e.to(device) if type(e) == torch.Tensor else e for e in input])
            # call softmax and get probability
            softmax = nn.Softmax(dim=1)
            sepsis_prob = softmax(model(input)).detach().to("cpu").numpy().tolist()
            sepsis_prob_actual = [x[1] for x in sepsis_prob]
            probas.append(sepsis_prob_actual)

            label = label.detach().to("cpu").tolist()
            labels.append(label)

    return probas, labels


def display_test_metrics(model, test_loader, prob_threshold=0.3):
    """
    displays the performance metrics on test sets
    """
    # get test probabilities
    device = "cpu"
    test_probs, labels = predict_sepsis_probabiltiies(model, device, test_loader)
    test_probs_flat = list(np.concatenate(test_probs).flat)
    labels_flat = list(np.concatenate(labels).flat)
    test_binary = [int(prob > prob_threshold) for prob in test_probs_flat]

    # precision, recall and accuracy
    precision = precision_score(labels_flat, test_binary, zero_division=1)
    recall = recall_score(labels_flat, test_binary, zero_division=1)
    acc = accuracy_score(labels_flat, test_binary)
    print("Precision for the test: ", precision)
    print("Recall for test: ", recall)
    print("Accuracy for test: ", acc)

    class_names = ["No Sepsis", "Sepsis"]
    results = list(zip(labels_flat, test_binary))
    plot_confusion_matrix(results, class_names, model_type="RNN for Sepsis Prediction")
