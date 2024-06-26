import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os
from torchvision.utils import save_image
import numpy as np
from scipy.stats import entropy
import sys
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt


# Add the path to the system path
current_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from models.GAN_CIFAR10_torch import *

# Constants
BATCH_SIZE = 64 # Critic batch size
GEN_BS_MULTIPLE = 2 # Generator batch size, as a multiple of BATCH_SIZE
DIM_G = 128 # Generator dimensionality
DIM_D = 128 # Critic dimensionality
NORMALIZATION_G = True # Use batchnorm in generator?
NORMALIZATION_D = False # Use batchnorm (or layernorm) in critic? This doesn't do anything at the moment.
OUTPUT_DIM = 3072 # Number of pixels in cifar10 (32*32*3)
DECAY = True # Whether to decay LR over learning
INCEPTION_FREQUENCY = 100 # How frequently to calculate Inception score
j=0

LR = 2e-4 # Initial learning rate
ITERS = 150
alpha = 0
rev = 0

# WGAN-GP parameter
CONDITIONAL = False # Whether to train a conditional or unconditional model
ACGAN = False # If CONDITIONAL, whether to use ACGAN or "vanilla" conditioning
ACGAN_SCALE = 1. # How to scale the critic's ACGAN loss relative to WGAN loss
ACGAN_SCALE_G = 0.1 # How to scale generator's ACGAN loss relative to WGAN loss

if CONDITIONAL and (not ACGAN) and (not NORMALIZATION_D):
    print ("WARNING! Conditional model without normalization in D might be effectively unconditional!")
device = 'cuda' if torch.cuda.is_available() else 'cpu'


def generate_image(generator, frame):
    generator.eval()
    n_samples = 12
    fixed_noise = torch.randn(n_samples, 128, device=device)
    fixed_labels = torch.tensor(np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9] * 10), dtype=torch.long, device=device)
    with torch.no_grad():
        samples = generator(n_samples, fixed_labels, fixed_noise).detach().cpu()
        samples = ((samples + 1) * 127.5).clamp(0, 255).to(torch.uint8)
        samples = samples.view(n_samples, 3, 32, 32)
        samples = samples.permute(0, 2, 3, 1)

        n_rows = (n_samples + 3) // 4
        fig, axs = plt.subplots(
            nrows=n_rows,
            ncols=4,
            figsize=(8, 2*n_rows),
            subplot_kw={'xticks': [], 'yticks': [], 'frame_on': False}
        )
        for i, axis in enumerate(axs.flat[:n_samples]):
            axis.imshow(samples[i], cmap='binary')
        plt.subplots_adjust(wspace=0.1, hspace=0.1)
        plt.savefig(f'samples/gen_images_{frame}.png')
        # plt.show()
        plt.close(fig)

    generator.train()


def train(generator, discriminator, gen_opt, disc_opt, train_loader):
    disc_costs = np.zeros(ITERS)
    gen_costs = np.zeros(ITERS)
    
    generator.train()
    discriminator.train()
    
    for iteration in tqdm(range(ITERS)):
        gen_loss = 0
        disc_loss = 0
        
        for real_data, labels in train_loader:
            real_data = real_data.to(device)
            labels = labels.to(device)
            
            # Generator forward pass
            noise = torch.randn(BATCH_SIZE, DIM_G, device=device)
            fake_data = generator(BATCH_SIZE, labels, noise)
            
            # Discriminator forward pass
            disc_real, _ = discriminator(real_data, labels)
            disc_fake, _ = discriminator(fake_data, labels)
            
            # Calculate discriminator loss
            if alpha == 0: # Standard WGAN loss
                disc_cost_real = disc_real.mean()
                disc_cost_fake = disc_fake.mean()
                disc_cost = disc_cost_fake - disc_cost_real
            elif alpha == 1:
                if rev == 0:
                    disc_cost_real = disc_real.mean()
                    disc_cost_fake = torch.log(torch.exp(disc_fake).mean())
                    disc_cost = disc_cost_fake - disc_cost_real
                elif rev == 1:
                    disc_cost_real = torch.log(torch.exp(disc_real).mean())
                    disc_cost_fake = disc_fake.mean()
                    disc_cost = disc_cost_fake - disc_cost_real
            elif alpha == -1: # alpha = infinity
                if rev == 0:
                    disc_cost_real = disc_real.mean()
                    disc_cost_fake = F.relu(disc_fake).mean()
                    disc_cost = disc_cost_fake - disc_cost_real
                elif rev == 1:
                    disc_cost_real = F.relu(disc_real).mean()
                    disc_cost_fake = disc_fake.mean()
                    disc_cost = disc_cost_fake - disc_cost_real
            else: # Reversed generalized alphaGAN
                if rev == 0:
                    disc_cost_real = disc_real.mean()
                    disc_cost_fake = f_alpha_star(disc_fake, alpha).mean()
                    disc_cost = disc_cost_fake - disc_cost_real
                elif rev == 1:
                    disc_cost_real = f_alpha_star(disc_real, alpha).mean()
                    disc_cost_fake = disc_fake.mean()
                    disc_cost = disc_cost_fake - disc_cost_real
            
            disc_loss += disc_cost.item()
            
            # add CONDITIONAL AND ACGAN code here
            
            # Update discriminator
            disc_opt.zero_grad()
            disc_cost.backward()
            disc_opt.step()
            
            for p in discriminator.parameters():
                p.data.clamp_(-0.01, 0.01)

            # Generator forward pass
            noise = torch.randn(BATCH_SIZE, DIM_G, device=device)
            fake_data = generator(BATCH_SIZE, labels, noise)
            disc_fake, _ = discriminator(fake_data, labels)
            
            # Calculate generator loss
            if alpha == 0: # Standard WGAN loss
                gen_cost =  -disc_fake.mean()
            elif alpha == 1:
                if rev == 0:
                    gen_cost = -torch.log(torch.exp(disc_fake).mean())
                elif rev == 1:
                    gen_cost = disc_fake.mean()
            elif alpha == -1: # alpha = infinity
                if rev == 0:
                    gen_cost = -F.relu(disc_fake).mean()
                elif rev == 1:
                    gen_cost = disc_fake.mean()
            else: # Reversed generalized alphaGAN
                if rev == 0:
                    gen_cost = -f_alpha_star(disc_fake, alpha).mean()
                elif rev == 1:
                    gen_cost = disc_fake.mean()
            
            gen_loss += gen_cost.item()
                    
            # Update generator
            gen_opt.zero_grad()
            gen_cost.backward()
            gen_opt.step()
        
        disc_costs[iteration] = disc_loss / len(train_loader)
        gen_costs[iteration] = gen_loss / len(train_loader)
        print(f'Iteration {iteration}, Generator loss: {gen_costs[iteration]}, Discriminator loss: {disc_costs[iteration]}')
        # Save checkpoints
        if iteration % 10 == 0:
            torch.save({
                'generator': generator.state_dict(),
                'discriminator': discriminator.state_dict(),
                'gen_opt': gen_opt.state_dict(),
                'disc_opt': disc_opt.state_dict(),
            }, f'cifar_resnet_sn/checkpoint_{iteration}.pth')
            
            generate_image(generator, iteration)
            # # Calculate and log the inception score
            # inception_mean, inception_std = get_inception_score(generator, 5000)
            # print(f"Inception score at iteration {iteration}: {inception_mean} ± {inception_std}")

    return disc_costs, gen_costs
    

def main():

    if not os.path.exists(f'cifar_resnet_sn/'):
        os.makedirs(f'cifar_resnet_sn/')

    # Initialize models
    generator = Generator(dim_g=DIM_G, output_dim=OUTPUT_DIM).to(device)
    discriminator = Discriminator(input_dim=3, dim_d=DIM_D).to(device)
    print(generator)
    print()
    print(discriminator)
    print('Using device:', device)
    # Optimizers
    gen_opt = optim.Adam(generator.parameters(), lr=LR, betas=(0., 0.9))
    disc_opt = optim.Adam(discriminator.parameters(), lr=LR, betas=(0., 0.9))

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    train_dataset = datasets.CIFAR10(root='data', train=True, transform=transform, download=True)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    d_loss, g_loss = train(generator, discriminator, gen_opt, disc_opt, train_loader)
    
    epoch_ax = np.arange(start=1, stop=ITERS+1, step=1)
    _, ax = plt.subplots(1, 2, figsize=(15, 5))

    ax[0].plot(epoch_ax, d_loss, color='blue')
    ax[0].set_xlim(1, ITERS)
    ax[0].set_title("Discriminator Loss vs Epoch")
    ax[0].grid()

    ax[1].plot(epoch_ax, g_loss, color='red')
    ax[1].set_xlim(1, ITERS)
    ax[1].set_title("Generator Loss vs Epoch")
    ax[1].grid()

    plt.tight_layout()
    plt.savefig('cifar_resnet_sn/loss_vs_epoch.png')
    plt.show()
    plt.close()    
    
if __name__ == "__main__":
    main()