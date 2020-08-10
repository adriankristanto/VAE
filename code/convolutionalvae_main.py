import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from tqdm import tqdm
import os
from models.ConvolutionalVAE import ConvolutionalVAE
from datetime import datetime

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Current device: {device}', flush=True)

# 1. load the data
DATA_PATH = os.path.dirname(os.path.realpath(__file__)) + '/../data/'
BATCH_SIZE = 128

train_transform = transforms.Compose([
    transforms.ToTensor()
])

test_transform = transforms.Compose([
    transforms.ToTensor()
])

trainset = datasets.MNIST(root=DATA_PATH, train=True, transform=train_transform, download=True)
testset = datasets.MNIST(root=DATA_PATH, train=False, transform=test_transform, download=True)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)
testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False)

print(f"""
Total training data: {len(trainset)}
Total testing data: {len(testset)}
Total data: {len(trainset) + len(testset)}
""", flush=True)


# 2. instantiate the model
Z_DIM = 20
# def __init__(self, 
#         e_channels, e_kernels, e_strides, e_paddings, e_activation_func, 
#         z_dim,
#         d_channels, d_kernels, d_strides, d_paddings, d_internal_activation, d_output_activation
#     ):
net = VAE(
    e_channels=[1, 32, 64, 128, 128],
    e_kernels=[None, 3, 2, 2, 3],
    e_strides=[None, 1, 2, 2, 1],
    e_paddings=[None, 1, 0, 0, 1],
    e_activation_function=nn.LeakyReLU(),
    z_dim=Z_DIM,
    d_channels=[128, 128, 64, 32, 1],
    d_kernels=[3, 2, 2, 3, None],
    d_strides=[1, 2, 2, 1, None],
    d_paddings=[1, 0, 0, 1, None],
    d_internal_activation=nn.LeakyReLU(),
    d_output_activation=nn.Sigmoid()
)

multigpu = False
if torch.cuda.device_count() > 1:
    print(f'Number of GPUs: {torch.cuda.device_count()}\n', flush=True)
    net = nn.DataParallel(net)
    multigpu = True

net.to(device)


# 3. define the loss function
def vae_loss(x_reconstructed, x_original, mean, log_var):
    reconstruction_loss = F.mse_loss(x_reconstructed, x_original, reduction='sum')
    kl_divergence = -0.5 * torch.sum(1 = log_var - (mean ** 2) - torch.exp(log_var))
    return reconstruction_loss + kl_divergence


# 4. define the optimizer
LEARNING_RATE = 0.001
optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)


# 5. train the model
MODEL_DIRPATH = os.path.dirname(os.path.realpath(__file__)) + '/../saved_models/'
GENERATED_DIRPATH = os.path.dirname(os.path.realpath(__file__)) + '/../generated_images/'
CONTINUE_TRAIN = False
CONTINUE_TRAIN_NAME = MODEL_DIRPATH + 'convolutionalvae-model-epoch10.pth'
EPOCH = 50
SAVE_INTERVAL = 5
SAMPLE = torch.randn((BATCH_SIZE, Z_DIM))

IMAGE_SIZE = (1, 28, 28)

next_epoch = 0
if CONTINUE_TRAIN:
    checkpoint = torch.load(CONTINUE_TRAIN_NAME)
    net.load_state_dict(checkpoint.get('net_state_dict'))
    optimizer.load_state_dict(checkpoint.get('optimizer_state_dict'))
    next_epoch = checkpoint.get('epoch')