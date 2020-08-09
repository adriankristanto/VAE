import torch
import torch.nn as nn
import torch.nn.functional as F
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
DATA_PATH = os.path.dirname(os.path.realpath(__file__)) + '/../data/'
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
    print(f'Number of GPUs: {torch.cuda.device_count()}\n', flush=True)
    net = nn.DataParallel(net)

net.to(device)


# 3. define the loss function
def vae_loss(x_reconstructed, x_original):
    # reference: https://stats.stackexchange.com/questions/318748/deriving-the-kl-divergence-loss-for-vaes
    # reference: https://wiseodd.github.io/techblog/2016/12/10/variational-autoencoder/
    # reconstruction loss
    mse_loss = F.mse_loss(x_reconstructed, x_original)
    # KL divergence
    kl_divergence = -0.5 * torch.sum(1 + log_var - mean ** 2 - torch.exp(log_var))
    return mse_loss + kl_divergence


# 4. define the optimiser
LEARNING_RATE = 0.001
optimiser = optim.Adam(net.parameters(), lr=LEARNING_RATE)


# 5. train the model
MODEL_DIRPATH = os.path.dirname(os.path.realpath(__file__)) + '/../model/'
CONTINUE_TRAIN = False
CONTINUE_TRAIN_NAME = MODEL_DIRPATH + 'model-epoch10.pth'
EPOCH = 20
SAVE_INTERVAL = 5