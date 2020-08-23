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
Z_DIM = 100
net = ConvolutionalVAE(
    input_shape=(1, 32, 32),
    e_channels=[1, 32, 64, 128, 256, 512],
    e_kernels=[None, 4, 4, 4, 4, 4],
    e_strides=[None, 2, 2, 2, 2, 2],
    e_paddings=[None, 1, 1, 1, 1, 1],
    e_activation_func=nn.LeakyReLU(),
    z_dim=Z_DIM,
    d_channels=[512, 256, 128, 64, 32, 1],
    d_kernels=[4, 4, 4, 4, 4, None],
    d_strides=[2, 2, 2, 2, 2, None],
    d_paddings=[1, 1, 1, 1, 1, None],
    d_internal_activation=nn.LeakyReLU(),
    d_output_activation=nn.Sigmoid()
)
# reference: https://discuss.pytorch.org/t/solved-keyerror-unexpected-key-module-encoder-embedding-weight-in-state-dict/1686/3
# loading a model that was wrapped by nn.DataParallel for training
checkpoint = torch.load(MODEL_PATH + MODEL_NAME, map_location=device)
old_G_state_dict = checkpoint.get('net_state_dict')
# if the model was wrapped by nn.DataParallel
if 'module.' in list(old_G_state_dict.keys())[0]:
    new_G_state_dict = OrderedDict()
    for key, value in old_G_state_dict.items():
        # remove "module." from each key
        name = key[7:]
        new_G_state_dict[name] = value
    # load the newly created state dict
    net.load_state_dict(new_G_state_dict)
else:
    net.load_state_dict(old_G_state_dict)

# generate new image
sample = torch.randn((1, Z_DIM))
print(f'Sample: {sample}\n')
sample = net.decode(sample)
filename = datetime.now().strftime("%d_%m_%Y_%H%M%S")
torchvision.utils.save_image(sample, f'{filename}.png')