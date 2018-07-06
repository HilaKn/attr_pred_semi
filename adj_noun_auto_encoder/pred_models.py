import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch
import numpy as np
from config import *


class ModelTypes:
    MLP = "MLP"

def model_factory(model_name):
    model = ""
    if model_name == ModelTypes.MLP:
        if USE_DROPOUT:
            model = MLPNetDropouts()
        else:
            model = MLPNet()
    return model


class MLPNet(nn.Module):
    def __init__(self):
        super(MLPNet, self).__init__()
        self.linear_input = nn.Linear(D_IN, D_HIDDEN)
        torch.nn.init.xavier_uniform_(self.linear_input.weight)
        self.hidden_linears = nn.ModuleList([nn.Linear(D_HIDDEN,D_HIDDEN) for i in xrange(0,HIDDEN_LAYERS-1)])
        for hidden_lin in self.hidden_linears:
            torch.nn.init.xavier_uniform_(hidden_lin.weight)
        self.out_layer = nn.Linear(D_HIDDEN,D_OUT)
        torch.nn.init.xavier_uniform_(self.out_layer.weight)

    def forward(self, x):
        x = F.tanh(self.linear_input(x))
        for i, linear in enumerate(self.hidden_linears):
            x = F.tanh(linear(x))
        x = self.out_layer(x)
        return x


class MLPNetDropouts(MLPNet):
    def __init__(self):
        super(MLPNetDropouts, self).__init__()

    def forward(self, x):
        x = F.tanh(self.linear_input(x))
        x = F.dropout(x, p=DROPOUT_RATE, training=self.training )
        for i, linear in enumerate(self.hidden_linears):
            x = F.tanh(linear(x))
            x = F.dropout(x, p=DROPOUT_RATE, training=self.training )
        x = self.out_layer(x)
        return x
