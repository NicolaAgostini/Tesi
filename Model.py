from torch import nn
import torch
from torch.nn.init import normal, constant
import numpy as np
from torch.nn import functional as F



class LSTM(nn.Module):
    """"An LSTM implementation """
    def __init__(self, feat_in, feat_out=1024, num_layers=1, dropout=0, num_class=106):
        """
            feat_in: input feature size
            feat_out: output feature size
            num_layers: number of layers
            dropout: dropout probability
        """
        super(LSTM, self).__init__()
        self.input_dim = feat_in
        self.hidden_dim = feat_out  #hidden è la dimensione dell'output dell'lstm che poi sarà l'input della FC
        self.num_layers = num_layers  # numero di livelli(iterazioni) di una singola lstm

        # simply create an LSTM with the given parameters
        self.lstm = nn.LSTM(feat_in, feat_out, num_layers=num_layers, dropout=dropout)
        #self.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(self.hidden, num_class))
        #self.linear = nn.Linear(self.hidden_dim, output_dim)

    def forward(self, seq):
        """
        Iterates over the input and pass the hidden and cell state to the next LSTM
        :param seq:
        :return: the list of (hidden state, cell state) of the LSTM over the input sequence of frames
        """
        last_cell = None
        last_hid = None
        hid = []
        cell = []
        for i in range(seq.shape[0]):  # for every time steps
            el = seq[i, ...].unsqueeze(0)
            if last_cell is not None:
                _, (last_hid, last_cell) = self.lstm(el, (last_hid, last_cell))  # the next LSTM takes as inputs the
                # frame at time t and also the hidden and cell state of the previous LSTM
            else:
                _, (last_hid, last_cell) = self.lstm(el)
            hid.append(last_hid)
            cell.append(last_cell)

        return torch.stack(hid, 0), torch.stack(cell, 0)


class LSTMROLLING(nn.Module):
    def __init__(self, num_class, feat_in, hidden, dropout=0.8, depth=1):


        super(LSTMROLLING, self).__init__()
        self.feat_in = feat_in
        self.dropout = nn.Dropout(dropout)
        self.hidden = hidden
        self.rolling_lstm = LSTM(feat_in, hidden, num_layers=depth, dropout=dropout if depth > 1 else 0)
        #self.unrolling_lstm = nn.LSTM(feat_in, hidden, num_layers=depth, dropout=dropout if depth > 1 else 0)
        #self.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden, num_class))

    def forward(self, inputs):
        # permute the inputs for compatibility with the LSTM
        inputs = inputs.permute(1, 0, 2)

        # pass the frames through the rolling LSTM
        # and get the hidden (x) and cell (c) states at each time-step
        x, c = self.rolling_lstm(self.dropout(inputs))  # x, c are the hidden and cell states of all the lstm during the time steps
        out = x.contiguous()  # batchsize x timesteps x hidden


        return out  # return the hidden state for every V_i


class RLSTMFusion(nn.Module):
    def __init__(self, branches, hidden):
        """
            branches: list of pre-trained branches
            hidden: size of hidden vectors of the branches
        """
        super(RLSTMFusion, self).__init__()
        self.branches = nn.ModuleList(branches)

        # input size for the MATT network
        # given by 1(only the hidden state) * num_branches * hidden_size
        in_size = len(self.branches) * hidden

        # final fusion of branches
        self.final = nn.Linear(in_size, 106)

    def forward(self, inputs):
        """inputs: tuple containing the inputs to the single branches
        :return: a 106 vector with largest number in predicted future actions
        """
        preds = []

        # for each branch
        for i in range(len(inputs)):
            # feed the inputs to the LSTM and get the scores y
            prediction = self.branches[i](inputs[i])
            preds.append(prediction)  # preds now has a list of branches*(number of preds)*(hidden=1024)
        y = torch.stack(preds, 2)  # stack predictions along dimension 2
        y = y.contiguous().view(y.size(0), -1)  # concatenate the branches obtaining batch* # preds * 3*1024

        y_final = self.final(y)

        return y_final


