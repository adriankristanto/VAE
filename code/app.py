import torch
import torch.nn as nn
import torchvision
from models.VAE import VAE
import os
from datetime import datetime

device = torch.device("gpu" if torch.cuda.is_available() else "cpu")

# load the model
MODEL_PATH = os.path.dirname(os.path.realpath(__file__)) + '/../saved_models/'
MODEL_NAME = 'vae-model-epoch50.pth'
Z_DIM = 20
net = VAE(
    encoder_dim=[28 * 28 * 1, 512, 256, 128], 
    encoder_activation=nn.LeakyReLU(), 
    z_dim=Z_DIM, 
    decoder_dim=[20, 128, 256, 512, 784], 
    decoder_activation=nn.LeakyReLU(),
    output_activation=nn.Sigmoid()
)
checkpoint = torch.load(MODEL_PATH + MODEL_NAME, map_location=device)
net.load_state_dict(checkpoint.get('net_state_dict'))

# generate new image
sample = torch.randn((1, Z_DIM))
print(f'Sample: {sample}\n')
sample = net.decoder(sample)
sample = sample.view(-1, *[1, 28, 28])
filename = datetime.now().strftime("%d_%m_%Y_%H%M%S")
torchvision.utils.save_image(sample, f'{filename}.png')