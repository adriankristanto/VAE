import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from tqdm import tqdm
import os
from models.VAE import VAE
from datetime import datetime

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Current device: {device}', flush=True)

# 1. load the training data
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


# 2. instantiate the network model
Z_DIM = 20

net = VAE(
    encoder_dim=[28 * 28 * 1, 512, 256, 128], 
    encoder_activation=nn.LeakyReLU(), 
    z_dim=Z_DIM, 
    decoder_dim=[20, 128, 256, 512, 784], 
    decoder_activation=nn.LeakyReLU(),
    output_activation=nn.Sigmoid()
)

multigpu = False
if torch.cuda.device_count() > 1:
    print(f'Number of GPUs: {torch.cuda.device_count()}\n', flush=True)
    net = nn.DataParallel(net)
    multigpu = True

net.to(device)


# 3. define the loss function
def vae_loss(x_reconstructed, x_original, mean, log_var):
    # reference: https://stats.stackexchange.com/questions/318748/deriving-the-kl-divergence-loss-for-vaes
    # reference: https://wiseodd.github.io/techblog/2016/12/10/variational-autoencoder/
    # reconstruction loss
    # reduction='mean' doesn't seem to make the network learn
    mse_loss = F.mse_loss(x_reconstructed, x_original, reduction='sum')
    # KL divergence
    kl_divergence = -0.5 * torch.sum(1 + log_var - (mean ** 2) - torch.exp(log_var))
    return mse_loss + kl_divergence


# 4. define the optimiser
LEARNING_RATE = 0.001
optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)


# 5. train the model
MODEL_DIRPATH = os.path.dirname(os.path.realpath(__file__)) + '/../saved_models/'
GENERATED_DIRPATH = os.path.dirname(os.path.realpath(__file__)) + '/../generated_images/'
CONTINUE_TRAIN = False
CONTINUE_TRAIN_NAME = MODEL_DIRPATH + 'vae-model-epoch10.pth'
EPOCH = 50
SAVE_INTERVAL = 5
# for generation
SAMPLE = torch.randn((BATCH_SIZE, Z_DIM))

# bug note: before passing to torchvision.utils.save_image, unflatten to (1, 28, 28) NOT to (28, 28, 1)
IMAGE_SIZE = (1, 28, 28)
FLATTEN_SIZE = 28 * 28 * 1

# for training continuation
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
        # reference: https://stackoverflow.com/questions/50442000/dataparallel-object-has-no-attribute-init-hidden
        sample = net.module.decoder(sample) if multigpu else net.decoder(sample)
        # unflatten the image to be shown as a generated image
        sample = sample.view(BATCH_SIZE, *IMAGE_SIZE)
        torchvision.utils.save_image(sample, filename)

# generate data after visualisation
generate(SAMPLE, GENERATED_DIRPATH + 'vae_sample_0.png')

for epoch in range(next_epoch, EPOCH):
    running_loss = 0.0
    n = 0

    net.train()
    print(f'Currently training: {net.training}', flush=True)
    for train_data in tqdm(trainloader, desc=f'Epoch {epoch + 1}/{EPOCH}'):
        inputs = train_data[0].to(device)
        inputs = inputs.view(-1, FLATTEN_SIZE)
        # bug note: make sure to not forget zero the gradients at each iteration
        optimizer.zero_grad()
        # flatten the image as it is used as an input to a FC layer
        mean, log_var, outputs = net(inputs)
        loss = vae_loss(outputs, inputs, mean, log_var)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        n += len(inputs)
    
    # reference: https://github.com/pytorch/examples/blob/master/vae/main.py
    generate(SAMPLE, GENERATED_DIRPATH + f'vae_sample_{epoch+1}.png')
    
    if (epoch + 1) % SAVE_INTERVAL == 0:
        torch.save({
            # since the currect epoch has been completed, save the next epoch
            'epoch' : epoch + 1,
            'net_state_dict' : net.state_dict(),
            'optimizer_state_dict' : optimizer.state_dict(),
        }, MODEL_DIRPATH + f'vae-model-epoch{epoch + 1}.pth')

    print(f'Training Loss: {running_loss / n}')


# 6. test the model
# reference: https://github.com/pytorch/examples/blob/master/vae/main.py
RECONSTRUCTED_DIRPATH = os.path.dirname(os.path.realpath(__file__)) + '/../reconstructed_images/'

test_loss = 0.0
net.eval()
SEE_FLAG = True
with torch.no_grad():
    for test_data in tqdm(testloader):
        inputs = test_data[0].to(device)
        inputs = inputs.view(-1, FLATTEN_SIZE)
        mean, log_var, outputs = net(inputs)
        test_loss += vae_loss(outputs, inputs, mean, log_var)
        if SEE_FLAG:
            SEE_FLAG = False
            real = inputs.view(-1, *IMAGE_SIZE)
            reconstructed = outputs.view(-1, *IMAGE_SIZE)
            filename = datetime.now().strftime("%d_%m_%Y_%H%M%S")
            torchvision.utils.save_image(real, RECONSTRUCTED_DIRPATH + f'vae_real_{filename}.png', pad_value=1)
            torchvision.utils.save_image(reconstructed, RECONSTRUCTED_DIRPATH + f'vae_reconstructed_{filename}.png', pad_value=1)

print(f'Test loss: {test_loss / len(testset)}')
