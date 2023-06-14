'''
Multiple Layer Perceptron model for AEV-based representation 
'''
from torch import nn
from torch.nn import ReLU
from nmrpred.layers import Dense


class MLP(nn.Module):
    def __init__(self, dropout):
        super().__init__()
        self.hiddens = nn.ModuleList([Dense(384, 128, activation=ReLU(), dropout=dropout),
                                      Dense(128, 128, activation=ReLU(), dropout=dropout),
                                      Dense(128, 128, activation=ReLU(), dropout=dropout),
                                      Dense(128, 1)])

    def forward(self, input):
        x = input["aev"]
        for i in range(len(self.hiddens)):
            x = self.hiddens[i](x)
        return x

class AEVMLP(nn.Module):
    def __init__(self, n_neurons, dropout):
        super().__init__()
        hidden_layers = []
        for i in range(len(n_neurons) - 2):
            hidden_layers.append(Dense(n_neurons[i], n_neurons[i + 1], activation=ReLU(), dropout=dropout))
        hidden_layers.append(nn.Linear(n_neurons[-2], n_neurons[-1]))
        self.hidden_layers = nn.ModuleList(hidden_layers)

    def forward(self, inputs):
        x = inputs['aev']
        for i in range(len(self.hidden_layers)):
            x = self.hidden_layers[i](x)
        return x