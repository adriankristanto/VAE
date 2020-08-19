import torch
import torch.nn as nn
import torchvision
from datetime import datetime
from models.ConvolutionalVAE import ConvolutionalVAE 
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# load the model
MODEL_PATH = os.path.dirname(os.path.realpath(__file__)) + '/../saved_models/'
MODEL_NAME = 'convolutionalvae-model-epoch50.pth'
Z_DIM = 20
net = ConvolutionalVAE(
    input_shape=(1, 28, 28),
    e_channels=[1, 32, 64, 128, 128],
    e_kernels=[None, 3, 2, 2, 3],
    e_strides=[None, 1, 2, 2, 1],
    e_paddings=[None, 1, 0, 0, 1],
    e_activation_func=nn.LeakyReLU(),
    z_dim=Z_DIM,
    d_channels=[128, 128, 64, 32, 1],
    d_kernels=[3, 2, 2, 3, None],
    d_strides=[1, 2, 2, 1, None],
    d_paddings=[1, 0, 0, 1, None],
    d_internal_activation=nn.LeakyReLU(),
    d_output_activation=nn.Sigmoid()
)
checkpoint = torch.load(MODEL_PATH + MODEL_NAME, map_location=device)
net.load_state_dict(checkpoint.get('net_state_dict'))

# generate new image
sample = torch.randn((1, Z_DIM))
print(f'Sample: {sample}\n')
sample = net.decode(sample)
filename = datetime.now().strftime("%d_%m_%Y_%H%M%S")
torchvision.utils.save_image(sample, f'{filename}.png')