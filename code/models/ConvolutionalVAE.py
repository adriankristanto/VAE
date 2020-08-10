import torch
import torch.nn as nn
import numpy as np

class Encoder(nn.Module):
    def __init__(self, channels, kernels, strides, paddings, activation_func):
        super(Encoder, self).__init__()
        self.conv = self._build(channels, kernels, strides, paddings, activation_func)
    
    def _build(self, channels, kernels, strides, paddings, activation_func):
        layers = []
        for i in range(1, len(channels)):
            layer = nn.Conv2d(channels[i-1], channels[i], kernels[i], strides[i], paddings[i])
            layers.append(layer)
            layers.append(activation_func)
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.conv(x)
        return x

class Decoder(nn.Module):
    def __init__(self, channels, kernels, strides, paddings, internal_activation, output_activation):
        super(Decoder, self).__init__()
        self.conv = self._build(channels, kernels, strides, paddings, internal_activation, output_activation)
    
    def _build(self, channels, kernels, strides, paddings, internal_activation, output_activation):
        layers = []
        for i in range(len(channels) - 1):
            layer = nn.ConvTranspose2d(channels[i], channels[i+1], kernels[i], strides[i], paddings[i])
            layers.append(layer)
            # the final index represent the image channel, thus, there is no activation
            layers.append(internal_activation if i < len(channels) - 2 else output_activation)
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.conv(x)
        return x

class ConvolutionalVAE(nn.Module):
    def __init__(self, 
        input_shape,
        e_channels, e_kernels, e_strides, e_paddings, e_activation_func, 
        z_dim,
        d_channels, d_kernels, d_strides, d_paddings, d_internal_activation, d_output_activation
    ):
        super(ConvolutionalVAE, self).__init__()
        # create a convolution encoder
        self.encoder = Encoder(e_channels, e_kernels, e_strides, e_paddings, e_activation_func)
        self.z_dim = z_dim
        # create a convolution decoder
        self.decoder = Decoder(d_channels, d_kernels, d_strides, d_paddings, d_internal_activation, d_output_activation)
        # get the shape to be used for creating the fc layer in both encode and decode function
        self.unflatten_shape = self._initialize(input_shape)
    
    def _initialize(self, input_shape):
        x = self.encoder(torch.unsqueeze(torch.randn(input_shape), 0))
        return x.shape[1:]
    
    def sampling(self, mean, log_var):
        sigma = torch.exp(log_var / 2)
        epsilon = torch.randn_like(sigma)
        return mean + sigma * epsilon
    
    def encode(self, x):
        x = self.encoder(x)
        flatten_shape = np.prod(self.unflatten_shape)
        x = x.view(-1, flatten_shape)
        mean, log_var = nn.Linear(flatten_shape, self.z_dim)(x), nn.Linear(flatten_shape, self.z_dim)(x)
        z = self.sampling(mean, log_var)
        return mean, log_var, z
    
    def decode(self, z):
        flatten_shape = np.prod(self.unflatten_shape)
        x = nn.Linear(self.z_dim, flatten_shape)(z)
        x = x.view(-1, *self.unflatten_shape)
        x = self.decoder(x)
        return x
    
    def forward(self, x):
        mean, log_var, z = self.encode(x)
        x = self.decode(z)
        return mean, log_var, x


if __name__ == "__main__":
    vae = ConvolutionalVAE(
        (1, 28, 28),
        [1, 32, 64, 128, 128], 
        [None, 3, 2, 2, 3], 
        [None, 1, 2, 2, 1], 
        [None, 1, 0, 0, 1], 
        nn.LeakyReLU(),
        20,
        [128, 128, 64, 32, 1],
        [3, 2, 2, 3, None],
        [1, 2, 2, 1, None],
        [1, 0, 0, 1, None],
        nn.LeakyReLU(),
        nn.Sigmoid()
    )
    vae(torch.randn((1, 1, 28, 28)))
    print(vae)