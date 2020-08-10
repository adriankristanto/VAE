import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, channels, kernels, strides, paddings, activation_func):
        super(Encoder, self).__init__()
        self.layers = self._build(channels, kernels, strides, paddings, activation_func)
    
    def _build(self, channels, kernels, strides, paddings, activation_func):
        layers = []
        for i in range(1, len(channels)):
            layer = nn.Conv2d(channels[i-1], channels[i], kernels[i], strides[i], paddings[i])
            layers.append(layer)
            layers.append(activation_func)
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.layers(x)
        return x
