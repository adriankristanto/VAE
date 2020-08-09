import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import tqdm
import os
from models.VAE import VAE

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Current device: {device}', flush=True)

# 1. load the training data
DATA_PATH = os.path.dirname(os.path.realpath(__file__)) + '/../data'
BATCH_SIZE = 64

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

# 2. instantiate the network model
net = VAE(
    encoder_dim=[28 * 28 * 1, 400], 
    encoder_activation=nn.LeakyReLU(), 
    z_dim=20, 
    decoder_dim=[20, 400, 784], 
    decoder_activation=nn.LeakyReLU(),
    output_activation=nn.Sigmoid()
)

if torch.cuda.device_count() > 1:
    print(f'Number of GPUs: {torch.cuda.device_count()}', flush=True)
    net = nn.DataParallel(net)

net.to(device)
