import torch
import torch.nn as nn

class Encoder(nn.Module):

    def __init__(self, channels, kernels, strides, paddings, batch_norm, activation, hidden_dim, z_dim):
        super(Encoder, self).__init__()
        self.layers = self._build(channels, kernels, strides, paddings, batch_norm, activation)
        self.mean = nn.Linear(hidden_dim, z_dim)
        self.log_var = nn.Linear(hidden_dim, z_dim)
    
    def _build(self, channels, kernels, strides, paddings, batch_norm, activation):
        layers = []
        for i in range(1, len(channels)):
            layer = nn.Conv2d(
                in_channels=channels[i-1],
                out_channels=channels[i],
                kernel_size=kernels[i],
                stride=strides[i],
                padding=paddings[i],
                # if we use batch norm, then we don't need to use bias
                # else, we can use bias
                bias=not batch_norm
            )
            layers.append(layer)
            # if batch_norm == True and this is not the final layer, append batch_norm layer
            if batch_norm and i < len(channels) - 1:
                layers.append(nn.BatchNorm2d(channels[i]))
            # add activation function
            layers.append(activation)
        return nn.Sequential(*layers)
    
    def sampling(self, mean, log_var):
        # reparameterisation trick
        sigma = torch.exp(log_var / 2)
        epsilon = torch.randn_like(sigma)
        return mean + sigma * epsilon

    def forward(self, x):
        x = self.layers(x)
        mean = self.mean(x)
        log_var = self.log_var(x)
        z = self.sampling(mean, log_var)
        return mean, log_var, z
