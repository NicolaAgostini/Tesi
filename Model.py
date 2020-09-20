from torch import nn
import torch
from torch.nn.init import normal, constant
import numpy as np
from torch.nn import functional as F


class BaselineModel(torch.nn.Module):
    def __init__(self, batch_size, seq_len, input_size, dropout=0.8, num_classes=106):
        super(BaselineModel, self).__init__()
        self.lstm = torch.nn.LSTM(input_size[0], 1024, 1, batch_first=True)
        """
        self.branches = torch.nn.ModuleList([torch.nn.LSTM(input_size[0], 1024, 1, batch_first=True),
                                             torch.nn.LSTM(input_size[1], 1024, 1, batch_first=True)])
        """

        """
        self.branches = nn.ModuleDict({
            "rgb": torch.nn.LSTM(input_size[0], 1024, seq_len),  # input of lstm is 1024 (vector of input), hidden units are 1024, num layers is 14 (6 enc + 8 dec)
            "flow": torch.nn.LSTM(input_size[1], 1024, seq_len),  # output of each LSTM will be batch_size * temp_sequence=14 * hidden = 1024
            "obj": torch.nn.LSTM(input_size[2], 1024, seq_len)
        })
        """
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.dropout = torch.nn.Dropout(dropout)
        self.fc = torch.nn.Linear(1024*1, num_classes)  # without seq_len because i want my output on every timestamp from 0 to 2s of observations

        #self.fc = torch.nn.Linear(1024*3, num_classes)
        self.num_classes = num_classes

    def forward(self, feat):  # input will be batch_size * sequence length * input_dim
        """
        :param feat: input as a list [rgb, flow, obj] where each is size [batch_size, 14, len(rgb)]
        :return:
        """

        # LSTM forward
        x = []
        ### For multiple branches ###
        """
        for i, j in enumerate(feat):
            #print("INPUT " + str(i))
            #print(j.size())
            x_mod, hid = self.branches[i](j)  # x_mod has shapes [batch_size, 14, lstm_hidden_size=1024]
            x.append(x_mod)
            #print(x_mod.size())

        #print(x)
        # Concatenate
        x = torch.cat(x, -1)  # x has shape [batch_size, 14, 3 * lstm_hidden_size]
        #print(x.size())
        """


        x, _ = self.lstm(feat[0])  # for single branch

        # Take last time samples
        x = x[:, -8:, :]  # x has shape [batch_size, 8, 3 * lstm_hidden_size]

        # Dropout
        x = self.dropout(x)  # apply dropout against overfitting

        # Fully connected
        y = self.fc(x)  # output y has shape [batch_size, 8, num_classes]
        #print(y.size())

        '''
        For example, if each feature input is sampled every 0.25, then
        x[:, 0, :] is the prediction at 0.0 [sec]
        x[:, 1, :] is the prediction at 0.25 [sec]
        x[:, 2, :] is the prediction at 0.5 [sec]
        etc...
        '''

        return y



