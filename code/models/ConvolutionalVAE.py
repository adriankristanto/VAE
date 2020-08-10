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

class Decoder(nn.Module):
    def __init__(self, channels, kernels, strides, paddings, internal_activation, output_activation):
        super(Decoder, self).__init__()
        self.layers = self._build(channels, kernels, strides, paddings, internal_activation, output_activation)
    
    def _build(self, channels, kernels, strides, paddings, internal_activation, output_activation):
        layers = []
        for i in range(len(channels) - 1):
            layer = nn.ConvTranspose2d(channels[i], channels[i+1], kernels[i], strides[i], paddings[i])
            layers.append(layer)
            layers.append(internal_activation if i < len(channels) - 1 else output_activation)
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.layers(x)
        return x

class ConvolutionalVAE(nn.Module):
    def __init__(self, 
        e_channels, e_kernels, e_strides, e_paddings, e_activation_func, 
        z_dim,
        d_channels, d_kernels, d_strides, d_paddings, d_internal_activation, d_output_activation
    ):
        super(ConvolutionalVAE, self).__init__()
        self.encoder = Encoder(e_channels, e_kernels, e_strides, e_paddings, e_activation_func)
        self.z_dim = z_dim
        self.decoder = Decoder(d_channels, d_kernels, d_strides, d_paddings, d_internal_activation, d_output_activation)
    
    def sampling(self, mean, log_var):
        sigma = torch.exp(log_var / 2)
        epsilon = torch.randn_like(sigma)
        return mean + sigma * epsilon
    
    def forward(self, x):
        x = self.encoder(x)
        print(x.shape)
        flatten_shape = x.shape[1] * x.shape[2] * x.shape[3]
        unflatten_shape = x.shape[1:]
        # flatten
        x = x.view(-1, flatten_shape)
        mean, log_var = nn.Linear(flatten_shape, self.z_dim)(x), nn.Linear(flatten_shape, self.z_dim)(x)
        z = self.sampling(mean, log_var)
        # connect z to a fc layer
        z = nn.Linear(self.z_dim, flatten_shape)(z)
        z = z.view(-1, *unflatten_shape)
        x = self.decoder(z)
        return x


if __name__ == "__main__":
    vae = ConvolutionalVAE(
        [1, 32, 64, 128, 256], 
        [None, 3, 2, 2, 2], 
        [None, 1, 2, 2, 2], 
        [None, 1, 0, 0, 0], 
        nn.LeakyReLU(),
        20,
        [256, 128, 64, 32, 1],
        [2, 2, 2, 3, None],
        [2, 2, 2, 1, None],
        [0, 0, 0, 1, None],
        nn.LeakyReLU(),
        nn.Sigmoid()
    )
    vae(torch.randn((1, 1, 28, 28)))
    print(vae)