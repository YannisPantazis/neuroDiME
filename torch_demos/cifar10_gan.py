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
from torchinfo import summary
import argparse
import time

current_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from models.GAN_CIFAR10_torch import *
from models.Divergences_CIFAR10_torch import *

start = time.perf_counter()

fl_act_func_CC = 'poly-softplus' # abs, softplus, poly-softplus
optimizer = "RMS" #Adam, RMS
save_frequency = 10 #generator samples are saved every save_frequency epochs
num_gen_samples_to_save = 5000

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

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def main():
    # read input arguments
    parser = argparse.ArgumentParser(description='Neural-based Estimation of Divergences between Gaussians')
    parser.add_argument('--method', default='KLD-DV', type=str, metavar='method',
                        help='values: IPM, KLD-DV, KLD-LT, squared-Hel-LT, chi-squared-LT, JS-LT, alpha-LT, Renyi-DV, Renyi-CC, rescaled-Renyi-CC, Renyi-CC-WCR')
    parser.add_argument('--disc_steps_per_gen_step', default=5, type=int)
    parser.add_argument('--batch_size', default=64, type=int, metavar='m')
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--epochs', default=100, type=int,
                        help='number of total epochs to run')
    parser.add_argument('--alpha', default=2.0, type=float, metavar='alpha')
    parser.add_argument('--Lip_constant', default=1.0, type=float, metavar='Lipschitz constant')
    parser.add_argument('--gp_weight', default=1.0, type=float, metavar='GP weight')
    parser.add_argument('--spectral_norm', choices=('True','False'), default='False')
    parser.add_argument('--bounded', choices=('True','False'), default='False')
    parser.add_argument('--reverse_order', choices=('True','False'), default='False')
    parser.add_argument('--use_GP', choices=('True','False'), default='False')



    parser.add_argument('--run_number', default=1, type=int, metavar='run_num')

    opt = parser.parse_args()
    opt_dict = vars(opt)
    print('parsed options:', opt_dict)

    mthd = opt_dict['method']
    disc_steps_per_gen_step = opt_dict['disc_steps_per_gen_step']
    m = opt_dict['batch_size']
    lr = opt_dict['lr']
    epochs = opt_dict['epochs']
    if mthd=="squared-Hel-LT":
        alpha=1./2.
    elif mthd=="chi-squared-LT":
        alpha=2.
    elif mthd=="chi_squared_HCR":
        alpha=2.
    else:
        alpha = opt_dict['alpha']
    L = opt_dict['Lip_constant']
    gp_weight = opt_dict['gp_weight']
    run_num=opt_dict['run_number']

    spec_norm = opt_dict['spectral_norm']=='True'
    bounded=opt_dict['bounded']=='True'
    reverse_order = opt_dict['reverse_order']=='True'
    use_GP = opt_dict['use_GP']=='True'

    print("Spectral_norm: "+str(spec_norm))
    print("Bounded: "+str(bounded))
    print("Reversed: "+str(reverse_order))
    print("Use Gradient Penalty: "+str(use_GP))


    generator = Generator(dim_g=DIM_G, output_dim=OUTPUT_DIM).to(device)
    discriminator = Discriminator(input_dim=3, dim_d=DIM_D).to(device)

    summary(generator)
    print()
    summary(discriminator)
    
    print('Using device:', device)
    
    if optimizer == 'RMS':
        gen_opt = optim.RMSprop(generator.parameters(), lr=LR)
        disc_opt = optim.RMSprop(discriminator.parameters(), lr=LR)
    elif optimizer == 'Adam':
        gen_opt = optim.Adam(generator.parameters(), lr=LR, betas=(0., 0.9))
        disc_opt = optim.Adam(discriminator.parameters(), lr=LR, betas=(0., 0.9))

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    train_dataset = datasets.CIFAR10(root='data', train=True, transform=transform, download=True)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

    # construct gradient penalty
    if use_GP:
        discriminator_penalty=Gradient_Penalty_1Sided(gp_weight, L)
    else:
        discriminator_penalty=None

    # construct divergence
    if mthd=="IPM":
        div_dense = IPM(discriminator, disc_opt, epochs, m, discriminator_penalty)

    if mthd=="KLD-LT":
        div_dense = KLD_LT(discriminator, disc_opt, epochs, m, discriminator_penalty)
        
    if mthd=="KLD-DV":
        div_dense = KLD_DV(discriminator, disc_opt, epochs, m, discriminator_penalty)

    if mthd=="squared-Hel-LT":
        div_dense = squared_Hellinger_LT(discriminator, disc_opt, epochs, m, discriminator_penalty)

    if mthd=="chi-squared-LT":
        div_dense = Pearson_chi_squared_LT(discriminator, disc_opt, epochs, m, discriminator_penalty)

    if mthd=="chi-squared-HCR":
        div_dense = Pearson_chi_squared_HCR(discriminator, disc_opt, epochs, m, discriminator_penalty)

    if mthd=="JS-LT":
        div_dense = Jensen_Shannon_LT(discriminator, disc_opt, epochs, m, discriminator_penalty)    

    if mthd=="alpha-LT":
        div_dense = alpha_Divergence_LT(discriminator, disc_opt, alpha, epochs, m, discriminator_penalty)

    if mthd=="Renyi-DV":
        div_dense = Renyi_Divergence_DV(discriminator, disc_opt, alpha, epochs, m, discriminator_penalty)
        
    if mthd=="Renyi-CC":
        div_dense = Renyi_Divergence_CC(discriminator, disc_opt, alpha, epochs, m, fl_act_func_CC, discriminator_penalty)

    if mthd=="rescaled-Renyi-CC":
        div_dense = Renyi_Divergence_CC_rescaled(discriminator, disc_opt, alpha, epochs, m, fl_act_func_CC, discriminator_penalty)

    if mthd=="Renyi-WCR":
        div_dense = Renyi_Divergence_WCR(discriminator, disc_opt, epochs, m, fl_act_func_CC, discriminator_penalty)

    def noise_source(batch_size):
        return torch.randn(batch_size, DIM_G, device=device)

    GAN_model = GAN_CIFAR10(div_dense, generator, gen_opt, noise_source, epochs, disc_steps_per_gen_step, mthd, m, reverse_order)
    generator_samples, loss_array, gen_losses, disc_losses = GAN_model.train(train_loader, save_frequency, num_gen_samples_to_save)
    
    if not os.path.exists(f'generated_samples/'):
        os.makedirs(f'generated_samples/')
    
    # Save the loss vs epoch plot
    epoch_ax = np.arange(start=1, stop=epochs+1, step=1)
    _, ax = plt.subplots(1, 2, figsize=(15, 5))

    ax[0].plot(epoch_ax, disc_losses, color='blue')
    ax[0].set_xlim(1, epochs)
    ax[0].set_title("Discriminator Loss vs Epoch")
    ax[0].grid()

    ax[1].plot(epoch_ax, gen_losses, color='red')
    ax[1].set_xlim(1, epochs)
    ax[1].set_title("Generator Loss vs Epoch")
    ax[1].grid()

    plt.tight_layout()
    if use_GP:
        plt.savefig(f'samples_{mthd}_GP/loss_vs_epoch.png')
    else:
        plt.savefig(f'samples_{mthd}/loss_vs_epoch.png')
    plt.show()
    plt.close()    
    

    
if __name__ == '__main__':
    main()