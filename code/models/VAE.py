# reference: https://wiseodd.github.io/techblog/2016/12/10/variational-autoencoder/
import torch
import torch.nn as nn

class Encoder(nn.Module):

    def __init__(self, layers_dim, activation_func):
        super(Encoder, self).__init__()
        self.layers = self._build(layers_dim, activation_func)

    def _build(self, layers_dim, activation_func):
        layers = []
        for i in range(1, len(layers_dim)):
            layer = nn.Linear(layers_dim[i-1], layers_dim[i])
            layers.append(layer)
            # append internal activation layer
            layers.append(activation_func)
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
            # append internal activation layer if it's not the output layer
            # otherwise, add the output activation layer
            layers.append(internal_activation if i < len(layers_dim) - 1 else output_activation)
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.layers(x)
        return x


class VAE(nn.Module):

    def __init__(self, encoder_dim, encoder_activation, z_dim, decoder_dim, decoder_activation, output_activation):
        """
        Fully Connected VAE

        Attributes:

        encoder_dim: an array of integers containing the number of nodes from the input layer up until the layer
        before the latent layer, e.g. [784, 400] will create an encoder with one layer nn.Linear(784, 400)

        encoder_activation: activation function to be used after each encoder layer

        z_dim: an integer specifying the dimension of the latent space

        decoder_dim: an array of integers containing the number of nodes from the input layer (the z_dim) up until 
        the output layer, e.g. [20, 400, 784] will create a decoder with 2 layers nn.Linear(20, 400) and nn.Linear(400,784)
        where 20 is the z_dim

        decoder_activation: activation function to be used after each decoder layer expect for the output layer

        output_activation: activation function to be used on the decoder output layer
        """
        super(VAE, self).__init__()
        self.encoder = Encoder(layers_dim=encoder_dim, activation_func=encoder_activation)
        self.fc1 = nn.Linear(encoder_dim[-1], z_dim)
        self.fc2 = nn.Linear(encoder_dim[-1], z_dim)
        self.decoder = Decoder(layers_dim=decoder_dim, internal_activation=decoder_activation, output_activation=output_activation)
    
    def sampling(self, mean, log_var):
        # sample from standard normal distribution
        sigma = torch.exp(log_var / 2)
        # epsilon has the same shape as sigma
        epsilon = torch.randn_like(sigma)
        z = mean + sigma * epsilon
        return z
    
    def forward(self, x):
        x = self.encoder(x)
        mean, log_var = self.fc1(x), self.fc2(x)
        z = self.sampling(mean, log_var)
        x = self.decoder(z)
        return mean, log_var, x


if __name__ == "__main__":
    vae = VAE([784, 400], nn.LeakyReLU(), 20, [20, 400, 784], nn.LeakyReLU(), nn.Sigmoid())
    # vae(torch.randn((1, 784)))
    print(vae)