import torch
import torch.nn as nn
from sklearn import metrics


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
            sepsis_prob = softmax(model(input)).detach().to("cpu").numpy()[0].tolist()[1]

            probas.append(sepsis_prob)
            labels.append(label)

    return probas, labels


def display_test_metrics(model, test_loader, prob_threshold=0.5):
    """
        displays the performance metrics on test sets
    """
    # get test probabilities
    device = "cpu"
    test_probs, labels = predict_sepsis_probabiltiies(model, device, test_loader)

    import pdb

    pdb.set_trace()
    # convert to cut offs at prob_threshold

