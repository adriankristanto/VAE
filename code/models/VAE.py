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
            layers.append(self.activation_func)
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.layers(x)
        return x


class Decoder(nn.Module):
    
    def __init__(self, layers_dim, activation_func):
        super(Decoder, self).__init__()
        self.layers = self._build(layers_dim)
        self.activation_func = activation_func
    
    def _build(self, layers_dim):
        layers = []
        for i in range(1, len(layers_dim)):
            layer = nn.Linear(layers_dim[i-1], layers_dim[i])
            layers.append(layer)
            layers.append(self.activation_func)
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.layers(x)
        return x


class VAE(nn.Module):

    def __init__(self, encoder_dim, encoder_activation, z_dim, decoder_dim, decoder_activation, output_dim,  output_activation):
        super(VAE, self).__init__()
        self.encoder = Encoder(layers_dim=encoder_dim, activation_func=encoder_activation)
        self.latent_layer = nn.Linear(encoder_dim[-1], z_dim)
        self.decoder = Decoder(layers_dim=decoder_dim, activation_func=decoder_activation)
        self.output_layer = nn.Linear(decoder_dim[-1], output_dim)
        self.output_activation = output_activation
    
    def sampling(self, mean, log_var):
        sigma = torch.exp(log_var / 2)
        # epsilon has the same shape as sigma
        epsilon = torch.randn_like(sigma)
        z = mean + sigma * epsilon
        return z
    
    def forward(self, x):
        x = self.encoder(x)
        mean, log_var = self.latent_layer(x), self.latent_layer(x)
        z = sampling(mean, log_var)
        x = self.decoder(z)
        x = self.output_layer(x)
        x = self.output_activation(x)
        return x