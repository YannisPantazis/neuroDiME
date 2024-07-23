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
DECAY = True # Whether to decay LR over learning
INCEPTION_FREQUENCY = 100 # How frequently to calculate Inception score
j=0
LR = 2e-4 # Initial learning rate

device = 'cuda' if torch.cuda.is_available() else 'cpu'

if device == 'cuda':
    torch.cuda.init()
    torch.cuda.current_device()
    print(f"Current CUDA device: {torch.cuda.current_device()}")


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def main():
    # read input arguments
    parser = argparse.ArgumentParser(description='Neural-based Estimation of Divergences between Gaussians')
    parser.add_argument('--method', default='KLD-DV', type=str, metavar='method',
                        help='values: IPM, KLD-DV, KLD-LT, squared-Hel-LT, chi-squared-LT, JS-LT, alpha-LT, Renyi-DV, Renyi-CC, rescaled-Renyi-CC, Renyi-CC-WCR')
    parser.add_argument('--disc_steps_per_gen_step', default=1, type=int)
    parser.add_argument('--batch_size', default=64, type=int, metavar='m')
    parser.add_argument('--lr', default=2e-4, type=float)
    parser.add_argument('--epochs', default=100, type=int,
                        help='number of total epochs to run')
    parser.add_argument('--alpha', default=2.0, type=float, metavar='alpha')
    parser.add_argument('--Lip_constant', default=1.0, type=float, metavar='Lipschitz constant')
    parser.add_argument('--gp_weight', default=1.0, type=float, metavar='GP weight')
    parser.add_argument('--spectral_norm', choices=('True','False'), default='False')
    parser.add_argument('--bounded', choices=('True','False'), default='False')
    parser.add_argument('--reverse_order', choices=('True','False'), default='False')
    parser.add_argument('--use_GP', choices=('True','False'), default='False')
    parser.add_argument('--dataset', choices=('cifar10', 'mnist'), default='cifar10', type=str, metavar='dataset')
    parser.add_argument('--save_model', choices=('True','False'), default='False', type=str, metavar='save_model')
    parser.add_argument('--save_model_path', default='./trained_models/', type=str, metavar='save_model_path')
    parser.add_argument('--load_model', choices=('True','False'), default='False', type=str, metavar='load_model')  
    parser.add_argument('--load_model_path', default='trained_models/', type=str, metavar='load_model_path')
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
    dataset = opt_dict['dataset']
    save_model = opt_dict['save_model']=='True'
    save_model_path = opt_dict['save_model_path']
    load_model = opt_dict['load_model']=='True'
    load_model_path = opt_dict['load_model_path']
    
    print("Spectral_norm: "+str(spec_norm))
    print("Bounded: "+str(bounded))
    print("Reversed: "+str(reverse_order))
    print("Use Gradient Penalty: "+str(use_GP))
        
    if dataset=="cifar10":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        train_dataset = datasets.CIFAR10(root='data', train=True, transform=transform, download=True)
        input_dim=3
        OUTPUT_DIM = 3072 # Number of pixels in cifar10 (32*32*3)
    elif dataset=="mnist":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])        
        train_dataset = datasets.MNIST(root='data', train=True, transform=transform, download=True)
        input_dim=1
        OUTPUT_DIM = 784 # Number of pixels in mnist (28*28)
        
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, drop_last=True)

    generator = Generator(dim_g=DIM_G, output_dim=OUTPUT_DIM).to(device)
    discriminator = Discriminator(input_dim=input_dim, dim_d=DIM_D).to(device)

    generator.apply(weights_init)
    discriminator.apply(weights_init)
    
    summary(generator)
    print()
    summary(discriminator)
    
    print('Using device:', device)
    
    if optimizer == 'RMS':
        gen_opt = optim.RMSprop(generator.parameters(), lr=lr)
        disc_opt = optim.RMSprop(discriminator.parameters(), lr=lr)
    elif optimizer == 'Adam':
        gen_opt = optim.Adam(generator.parameters(), lr=lr, betas=(0., 0.9))
        disc_opt = optim.Adam(discriminator.parameters(), lr=lr, betas=(0., 0.9))
    
    # construct gradient penalty
    if use_GP:
        discriminator_penalty=Gradient_Penalty_2Sided(gp_weight, L)
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
        return torch.randn(batch_size, DIM_G).to(device)

    GAN_model = GAN_CIFAR10(div_dense, generator, gen_opt, noise_source, epochs, disc_steps_per_gen_step, mthd, dataset, m, reverse_order)
    
    # Load the model if specified
    if load_model:
        load_model_path = os.path.join(load_model_path, f'{mthd}_{dataset}.pt')
        GAN_model.load(load_model_path)
        print(f"Model loaded from {load_model_path}")
    else:
        generator_samples, loss_array, gen_losses, disc_losses, mean_scores, std_scores = GAN_model.train(train_loader, save_frequency, num_gen_samples_to_save)
    
        if not os.path.exists(f'generated_samples/'):
            os.makedirs(f'generated_samples/')
        
        # Save the loss vs epoch plot
        epoch_ax = np.arange(start=1, stop=epochs+1, step=1)
        _, ax = plt.subplots(1, 3, figsize=(15, 5))

        ax[0].plot(epoch_ax, disc_losses, color='blue')
        ax[0].set_xlim(1, epochs)
        ax[0].set_title("Discriminator Loss vs Epoch")
        ax[0].grid()

        ax[1].plot(epoch_ax, gen_losses, color='red')
        ax[1].set_xlim(1, epochs)
        ax[1].set_title("Generator Loss vs Epoch")
        ax[1].grid()

        ax[2].plot(epoch_ax, mean_scores, color='green', label='Inception Score Mean')
        ax[2].fill_between(epoch_ax, mean_scores-std_scores, mean_scores+std_scores, color='green', alpha=0.2, label='Inception Score Std')
        ax[2].set_xlim(1, epochs)
        ax[2].set_title("Inception Score Mean and Std vs Epoch")
        ax[2].grid()
        ax[2].legend()

        plt.tight_layout()
        if use_GP:
            plt.savefig(f'samples_{mthd}_GP_{dataset}/loss_vs_epoch.png')
        else:
            plt.savefig(f'samples_{mthd}_{dataset}/loss_vs_epoch.png')
        plt.show()
        plt.close()    
    
    # Save the model
    if save_model:
        save_model_path = os.path.join(save_model_path, f'{mthd}_{dataset}.pt')
        GAN_model.save(save_model_path)
        print(f"Model saved at {save_model_path}")
    
    
if __name__ == '__main__':
    main()
