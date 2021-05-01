import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class VariableRNN(nn.Module):
    """
    RNN model that takes in varying length of sequence
    """

    def __init__(self, dim_input):
        super(VariableRNN, self).__init__()
        # You may use the input argument 'dim_input', which is basically the number of features
        self.fc32 = nn.Linear(dim_input, 50)
        self.gru = nn.GRU(
            input_size=50,
            hidden_size=16,
            num_layers=3,
            dropout=0.15,
            batch_first=True,
        )
        self.fc2 = nn.Linear(16, 2)

    def forward(self, input_tuple):
        # build architecture
        x, lengths = input_tuple
        # pass x through the first layer
        x = F.relu(self.fc32(x))
        # create packed sequence as input to the lstm
        packed_x = pack_padded_sequence(x, lengths, batch_first=True)
        # pass padded pack sequence through the lstm layer
        packed_x, _ = self.gru(packed_x)
        # unpack the padded sequence from lstm output
        x, _ = pad_packed_sequence(packed_x, batch_first=True)
        # pass through the second layer
        x = self.fc2(x[:, -1, :])
        # x = F.softmax(x,dim=1)

        return x
