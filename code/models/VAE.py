import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):

    def __init__(self, layers_dim, activation_func):
        super(Encoder, self).__init__()
        self.layers = self._build(layers_dim)
        self.activation_func = activation_func

    def _build(self, layers_dim):
        layers = []
        for i in range(1, len(layers_dim)):
            layer = nn.Linear(layers_dim[i-1], layers_dim[i])
            layers.append(layer)
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.layers(x)
        x = self.activation_func(x)
        return x


