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
net = VAE(
    input_shape=(1, 28, 28),
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
    kl_divergence = -0.5 * torch.sum(1 + log_var - (mean ** 2) - torch.exp(log_var))
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

def generate(sample, filename):
    net.eval()
    with torch.no_grad():
        sample = sample.to(device)
        sample = net.module.decode(sample) if multigpu else net.decode(sample)
        torchvision.utils.save_image(sample, filename)

generate(SAMPLE, GENERATED_DIRPATH + 'convolutionalvae_sample_0.png')

for epoch in range(next_epoch, EPOCH):
    running_loss = 0.0
    n = 0

    net.train()
    for train_data in tqdm(dataloader, desc=f'Epoch {epoch+1}/{EPOCH}'):
        inputs = train_data[0].to(device)
        optimizer.zero_grad()
        mean, log_var, outputs = net(inputs)
        loss = vae_loss(outputs, inputs, mean, log_var)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        n += len(inputs)
    
    generate(SAMPLE, GENERATED_DIRPATH + f'convolutionalvae_sample_{epoch+1}.png')

    if (epoch + 1) % SAVE_INTERVAL == 0:
        torch.save({
            'epoch' : epoch + 1,
            'net_state_dict' : net.state_dict(),
            'optimizer_state_dict' : optimizer.state_dict(),
        }, MODEL_DIRPATH + f'convolutionalvae-model-epoch{epoch + 1}.pth')
    
    print(f'Training loss: {running_loss / n}', flush=True)


# 6. test the model
RECONSTRUCTED_DIRPATH = os.path.dirname(os.path.realpath(__file__)) + '/../reconstructed_images/'

test_loss = 0.0
net.eval()
SEE_FLAG = True
with torch.no_grad():
    for test_data in tqdm(testloader):
        inputs = test_data[0].to(device)
        mean, log_var, outputs = net(inputs)
        test_loss += vae_loss(outputs, inputs, mean, log_var)
        if SEE_FLAG:
            SEE_FLAG = False
            filename = datetime.now().strftime("%d_%m_%Y_%H%M%S")
            torchvision.utils.save_image(inputs, RECONSTRUCTED_DIRPATH + f'convolutionalvae_real_{filename}.png', pad_value=1)
            torchvision.utils.save_image(outputs, RECONSTRUCTED_DIRPATH + f'convolutionalvae_reconstructed_{filename}.png', pad_value=1)
print(f'Test loss: {test_loss / len(testset)}')