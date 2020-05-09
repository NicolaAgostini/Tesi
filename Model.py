from torch import nn
import torch
from torch.nn.init import normal, constant
import numpy as np
from torch.nn import functional as F

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class BaselineModel(torch.nn.Module):
    def __init__(self, batch_size, seq_len, input_size, dropout=0.2, num_classes=106):
        super(BaselineModel, self).__init__()

        self.branches = torch.nn.ModuleList([torch.nn.LSTM(input_size[0], 1024, seq_len),
                         torch.nn.LSTM(input_size[1], 1024, seq_len),
                         torch.nn.LSTM(input_size[2], 1024, seq_len)])
        """
        self.branches = nn.ModuleDict({
            "rgb": torch.nn.LSTM(input_size[0], 1024, seq_len),  # input of lstm is 1024 (vector of input), hidden units are 1024, num layers is 14 (6 enc + 8 dec)
            "flow": torch.nn.LSTM(input_size[1], 1024, seq_len),  # output of each LSTM will be batch_size * temp_sequence=14 * hidden = 1024
            "obj": torch.nn.LSTM(input_size[2], 1024, seq_len)
        })
        """
        self.batch_size = batch_size
        self.dropout = torch.nn.Dropout(dropout)
        self.fc = torch.nn.Linear(1024*3, num_classes)  # without seq_len because i want my output on every timestamp from 0 to 2s of observations
        #self.dropout = torch.nn.Dropout(dropout)
        #self.fc = torch.nn.Linear(1024*3, num_classes)
        self.num_classes = num_classes

    def forward(self, feat, hidden):  # input will be batch_size * sequence length * input_dim
        '''
        input feat: list like {key: np.ndarray of shape [batch_size, 14, len(key)] for key in modalities}  where
                    key€{rgb, flow, obj} and if key="rgb" => len(key) = 1024

        '''

        # LSTM forward
        x = []
        """
        ### first type of input ###
        
        for key, value in feat.items():
            x_mod, hidden = self.branches[key](value)  # x_mod has shapes [batch_size, 14, lstm_hidden_size=1024]
            #print(x_mod.size())
            x.append(x_mod)  # append to a list
        """
        new_hid = []
        for key in range(len(feat)):
            #print(key)
            #print(feat[key])
            #print(" === ")
            x_mod, hid = self.branches[key](feat[key], hidden[key])  # x_mod has shapes [batch_size, 14, lstm_hidden_size=1024]
            # print(x_mod.size())
            x.append(x_mod)  # append to a list
            new_hid.append(hid)
            #x.append(x_mod)


        #print(np.shape(x))
        # Concatenate
        x = torch.cat(x, -1)  # x has shape [batch_size, 14, 3 * lstm_hidden_size]

        #print("otput lstm" + str(x.size()))

        # Take last time samples
        x = x[:, -8:, :]  # x has shape [batch_size, 8, 3 * lstm_hidden_size]

        # Dropout
        x = self.dropout(x)  # apply dropout otherwise ll encounter overfitting
        #x = x.view(self.batch_size, -1)  # prepare input to FC linear
        #print(x.size())
        # Fully connected
        x = self.fc(x)  # output x has shape [batch_size, 8, num_classes]
        #x = torch.nn.functional.softmax(x, -1)  # transform last layer (106 vector) to probabilities sum to one

        '''
        For example, if each feature input is sampled every 0.25, then
        x[:, 0, :] is the prediction at 0.0 [sec]
        x[:, 1, :] is the prediction at 0.25 [sec]
        x[:, 2, :] is the prediction at 0.5 [sec]
        etc...
        '''

        return x, new_hid

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        hid = []
        hid.append((weight.new(14, 14, 1024).zero_().to(device),
                  weight.new(14, 14, 1024).zero_().to(device)))
        hid.append((weight.new(14, 14, 1024).zero_().to(device),
                    weight.new(14, 14, 1024).zero_().to(device)))
        hid.append((weight.new(14, 14, 1024).zero_().to(device),
                    weight.new(14, 14, 1024).zero_().to(device)))



        return hid
