import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):

    def __init__(self, layers_dim, activation_func):
        super(Encoder, self).__init__()
        self.layers = self._build(layers_dim, activation_func)

    def _build(self, layers_dim, activation_func):
        layers = []
        for i in range(1, len(layers_dim)):
            layer = nn.Linear(layers_dim[i-1], layers_dim[i])
            layers.append(layer)
            layers.append(activation_func)
        print(layers)
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.layers(x)
        return x


class Decoder(nn.Module):
    
    def __init__(self, layers_dim, internal_activation, output_activation):
        super(Decoder, self).__init__()
        self.layers = self._build(layers_dim, internal_activation, output_activation)
    
    def _build(self, layers_dim, internal_activation, output_activation):
        layers = []
        for i in range(1, len(layers_dim)):
            layer = nn.Linear(layers_dim[i-1], layers_dim[i])
            layers.append(layer)
            layers.append(internal_activation if i < len(layers_dim) - 1 else output_activation)
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.layers(x)
        return x


class VAE(nn.Module):

    def __init__(self, encoder_dim, encoder_activation, z_dim, decoder_dim, decoder_activation, output_activation):
        super(VAE, self).__init__()
        self.encoder = Encoder(layers_dim=encoder_dim, activation_func=encoder_activation)
        self.latent_layer = nn.Linear(encoder_dim[-1], z_dim)
        self.decoder = Decoder(layers_dim=decoder_dim, internal_activation=decoder_activation, output_activation=output_activation)
    
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
        return x

if __name__ == "__main__":
    vae = VAE([784, 400], nn.LeakyReLU(), 20, [20, 400, 784], nn.LeakyReLU(), nn.Sigmoid())
    print(vae)