"""
This module contains decoder networks that convert latent vectors into target predictions
"""

from torch import nn

from nmrpred.utils.torch_util import FCOps

class DenseNetOutput(nn.Module):
    def __init__(self, in_channels, dropout):
        super().__init__()
        self.contraction = FCOps(in_channels, 128, dropout)
        self.output_layer = nn.Linear(128, 1)

    def forward(self, inputs):
        x = self.contraction(inputs)
        y = self.output_layer(x)
        return y

class MLPOutput(nn.Module):
    def __init__(self, n_hidden_neurons, dropout):
        super().__init__()
        hidden_layers = []
        for i in range(len(n_hidden_neurons) - 1):
            hidden_layers.append(FCOps(n_hidden_neurons[i], n_hidden_neurons[i + 1], dropout, use_batchnorm=False))
        hidden_layers.append(nn.Linear(n_hidden_neurons[-1], 1))
        self.hidden_layers = nn.ModuleList(hidden_layers)

    def forward(self, x):
        for i in range(len(self.hidden_layers)):
            x = self.hidden_layers[i](x)
        return x

class AttentionMask(nn.Module):
    def __init__(self, n_neurons, dropout):
        super().__init__()
        hidden_layers = []
        for i in range(len(n_neurons) - 2):
            hidden_layers.append(FCOps(n_neurons[i], n_neurons[i + 1], dropout, use_batchnorm=False))
        hidden_layers.append(nn.Linear(n_neurons[-2], n_neurons[-1]))
        self.hidden_layers = nn.ModuleList(hidden_layers)

    def forward(self, x):
        for i in range(len(self.hidden_layers)):
            x = self.hidden_layers[i](x)
        return x
