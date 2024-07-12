import os
import sys
import csv
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image
from torch.autograd import Variable
import functools

# Paths and hyperparameters
username = 'mcgregor'
DATA_DIR = f'/home/{username}/CUMGAN/cumulant_gan_cifar10_python3_tf2.2/data/cifar-10-batches-py'
if len(DATA_DIR) == 0:
    raise Exception('Please specify path to data directory in gan_cifar.py!')

BATCH_SIZE = 64
GEN_BS_MULTIPLE = 2
DIM_G = 128
DIM_D = 128
OUTPUT_DIM = 3072
ITERS = int(sys.argv[1])
alpha = float(sys.argv[2])
rev = int(sys.argv[3])
LR = float(sys.argv[4])
INCEPTION_FREQUENCY = 100 # How frequently to calculate Inception score

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Set random seed for reproducibility
torch.manual_seed(999)

def nonlinearity(x):
    return F.relu(x)

def Normalize(name, inputs, labels=None):
    if name == 'batchnorm':
        return nn.BatchNorm2d(inputs.size(1)).to(device)(inputs)
    elif name == 'layernorm':
        return nn.LayerNorm(inputs.size()[1:]).to(device)(inputs)
    else:
        return inputs

class ResidualBlock(nn.Module):
    def __init__(self, input_dim, output_dim, filter_size, resample=None, normalization=True):
        super(ResidualBlock, self).__init__()
        self.resample = resample
        self.shortcut = nn.Conv2d(input_dim, output_dim, kernel_size=1, stride=1, padding=0)
        self.conv1 = nn.Conv2d(input_dim, output_dim, filter_size, stride=1, padding=filter_size//2)
        self.conv2 = nn.Conv2d(output_dim, output_dim, filter_size, stride=1, padding=filter_size//2)
        self.normalize = normalization
        self.batchnorm1 = nn.BatchNorm2d(input_dim)
        self.batchnorm2 = nn.BatchNorm2d(output_dim)

    def forward(self, x):
        shortcut = self.shortcut(x)
        if self.normalize:
            x = self.batchnorm1(x)
        x = nonlinearity(x)
        x = self.conv1(x)
        if self.normalize:
            x = self.batchnorm2(x)
        x = nonlinearity(x)
        x = self.conv2(x)
        return shortcut + x

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.fc = nn.Linear(128, 4*4*DIM_G)
        self.res_block1 = ResidualBlock(DIM_G, DIM_G, 3, resample='up')
        self.res_block2 = ResidualBlock(DIM_G, DIM_G, 3, resample='up')
        self.res_block3 = ResidualBlock(DIM_G, DIM_G, 3, resample='up')
        self.conv = nn.Conv2d(DIM_G, 3, 3, stride=1, padding=1)

    def forward(self, noise):
        x = self.fc(noise)
        x = x.view(-1, DIM_G, 4, 4)
        x = self.res_block1(x)
        x = self.res_block2(x)
        x = self.res_block3(x)
        x = nonlinearity(x)
        x = self.conv(x)
        return torch.tanh(x)

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Conv2d(3, DIM_D, 3, stride=1, padding=1)
        self.res_block1 = ResidualBlock(DIM_D, DIM_D, 3, resample='down')
        self.res_block2 = ResidualBlock(DIM_D, DIM_D, 3, resample='down')
        self.res_block3 = ResidualBlock(DIM_D, DIM_D, 3, resample=None)
        self.fc = nn.Linear(DIM_D * 4 * 4, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = nonlinearity(x)
        x = self.res_block1(x)
        x = self.res_block2(x)
        x = self.res_block3(x)
        x = nonlinearity(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

def f_alpha_star(y, alpha):
    return (torch.relu(y)**(alpha/(alpha-1.0)) * (alpha-1.0)**(alpha/(alpha-1.0)) / alpha + 1 / (alpha*(alpha-1.0)))

# Create the generator and discriminator models
generator = Generator().to(device)
discriminator = Discriminator().to(device)

# Optimizers
gen_opt = optim.Adam(generator.parameters(), lr=LR, betas=(0, 0.9))
disc_opt = optim.Adam(discriminator.parameters(), lr=LR, betas=(0, 0.9))

# DataLoader
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

trainset = datasets.CIFAR10(root=DATA_DIR, train=True, download=True, transform=transform)
trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

# Training Loop
for iteration in range(ITERS):
    for i, data in enumerate(trainloader, 0):
        real_data, _ = data
        real_data = real_data.to(device)

        # Train Discriminator
        noise = torch.randn(BATCH_SIZE, 128, device=device)
        fake_data = generator(noise)

        disc_real = discriminator(real_data).mean()
        disc_fake = discriminator(fake_data).mean()

        if alpha == 0:  # WGAN-GP loss
            disc_cost = disc_fake - disc_real
        elif alpha == 1:
            if rev == 1:
                disc_cost = disc_fake - torch.log(torch.exp(disc_real).mean())
            elif rev == 0:
                disc_cost = disc_real - torch.log(torch.exp(disc_fake).mean())
        elif alpha == -1:
            if rev == 1:
                disc_cost = disc_fake - torch.relu(disc_real).mean()
            elif rev == 0:
                disc_cost = disc_real - torch.relu(disc_fake).mean()
        else:  # reverse generalized alpha GAN
            if rev == 1:
                disc_cost = disc_fake - f_alpha_star(disc_real, alpha).mean()
            elif rev == 0:
                disc_cost = disc_real - f_alpha_star(disc_fake, alpha).mean()

        disc_opt.zero_grad()
        disc_cost.backward()
        disc_opt.step()

        # Train Generator
        noise = torch.randn(BATCH_SIZE, 128, device=device)
        fake_data = generator(noise)
        disc_fake = discriminator(fake_data).mean()

        if alpha == 0:
            gen_cost = -disc_fake
        elif alpha == 1:
            if rev == 1:
                gen_cost = disc_fake
            elif rev == 0:
                gen_cost = -torch.log(torch.exp(disc_fake).mean())
        elif alpha == -1:
            if rev == 1:
                gen_cost = disc_fake
            elif rev == 0:
                gen_cost = -torch.relu(disc_fake).mean()
        else:
            if rev == 1:
                gen_cost = disc_fake
            elif rev == 0:
                gen_cost = -f_alpha_star(disc_fake, alpha).mean()

        gen_opt.zero_grad()
        gen_cost.backward()
        gen_opt.step()

        # Print progress
        if i % 100 == 0:
            print(f"Iteration {iteration}/{ITERS}, Batch {i}/{len(trainloader)}, Gen Loss: {gen_cost.item()}, Disc Loss: {disc_cost.item()}")

    # Save generated images
    if iteration % INCEPTION_FREQUENCY == 0:
        with torch.no_grad():
            fake_images = generator(torch.randn(100, 128, device=device)).detach().cpu()
        save_image(fake_images, f'cifar_resnet_sn/{sess_name}/samples_{iteration}.png', nrow=10, normalize=True)

    # Save model checkpoints
    torch.save(generator.state_dict(), f'cifar_resnet_sn/{sess_name}/generator_{iteration}.pth')
    torch.save(discriminator.state_dict(), f'cifar_resnet_sn/{sess_name}/discriminator_{iteration}.pth')
