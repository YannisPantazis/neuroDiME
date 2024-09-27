import torch.optim as optim
from torchvision import utils
import os
from torchvision.utils import save_image, make_grid
import numpy as np
from scipy.stats import entropy
import sys
from tqdm import tqdm
import matplotlib.pyplot as plt
from torchinfo import summary
import argparse
import time
import torch.autograd as autograd
import torchvision
from torch.autograd import Variable
from torchmetrics.image.fid import FrechetInceptionDistance
import json
import csv

current_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from models.GAN_MNIST_torch import *
from models.Divergences_torch import *

start = time.perf_counter()

fl_act_func_CC = 'poly-softplus' # abs, softplus, poly-softplus
optimizer = "Adam" #Adam, RMS
save_frequency = 10 #generator samples are saved every save_frequency epochs
num_gen_samples_to_save = 5000

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class GradientPenalty(Discriminator_Penalty):
    def __init__(self, weight, L):
        Discriminator_Penalty.__init__(self, weight)
        self.L = L

    def get_Lip_constant(self):
        return self.L

    def set_Lip_constant(self, L):
        self.L = L
         
    def evaluate(self, c, images, samples, y):
        assert images.size() == samples.size()
        jump = torch.rand(images.shape[0], 1).cuda()
        jump_ = jump.expand(images.shape[0], images.nelement()//images.shape[0]).contiguous().view(images.shape[0],1,28,28)
        interpolated = Variable(images*jump_ + (1-jump_)*samples, requires_grad = True)
        if y is not None:
            c_ = c(interpolated, y)
        else:
            c_ = c(interpolated)
        gradients = autograd.grad(c_, interpolated, grad_outputs=(torch.ones(c_.size()).cuda()),create_graph = True, retain_graph = True)[0]
        return self.get_penalty_weight()*((self.L-gradients.norm(2,dim=1))**2).mean()


def generate_single_digit(generator, digit):
    generator.eval()
    labels = torch.tensor([digit] * 256, dtype=torch.long).to(device)
    labels = F.one_hot(labels, 10).float()
    print(labels)
    samples = generator(labels, 256)
    idx = torch.randint(0, samples.shape[0], (16,))
    grid = utils.make_grid(samples.cpu()[idx])
    return grid


def generate_labeled_mnist(generator, num_rows=10):
    """
    Generates a grid of labeled MNIST digits, with each column corresponding to a digit 0-9, and each row containing different samples.
    
    Args:
        generator: The generator model used to create MNIST digits.
        gen_params: The parameters of the generator model.
        num_rows: The number of rows of samples for each digit (default is 10).
    
    Returns:
        grid: A grid of images where each column corresponds to a digit 0-9, and each row is a different sample.
    """
    generator.eval()
    # Number of classes (0-9 digits)
    num_classes = 10
        
    # Initialize an empty list to store images for each digit
    digit_images = []
    
    for digit in range(num_classes):
        labels = torch.tensor([digit] * num_rows, dtype=torch.long).to(device)
        labels_one_hot = F.one_hot(labels, 10).float()
        generated_images = generator(labels_one_hot, 10)
        generated_images = generated_images.detach().cpu().numpy()
        digit_images.append(generated_images)
    
    # Stack the generated images vertically
    digit_images = np.vstack(digit_images)
    digit_images = torch.tensor(digit_images)

    # Create a grid of images with num_rows rows and num_classes columns
    grid = make_grid(digit_images, nrow=num_classes, padding=2, normalize=True, scale_each=True)
    return grid


def generate_random_digits(generator, num_rows=10):
    """
    Generates a grid of random MNIST digits, with each column corresponding to a digit 0-9, and each row containing different samples.
    
    Args:
        generator: The generator model used to create MNIST digits.
        gen_params: The parameters of the generator model.
        num_rows: The number of rows of samples for each digit (default is 10).
    
    Returns:
        grid: A grid of images where each column corresponds to a digit 0-9, and each row is a different sample.
    """
    generator.eval()
    # Number of classes (0-9 digits)
    num_classes = 10
    num_samples = num_classes * num_rows
        
    # Generate images
    generated_images = generator(num_samples)
    # generated_images = generated_images
    
    # Create a grid of images with num_rows rows and num_classes columns
    grid = make_grid(generated_images, nrow=num_classes, padding=2, normalize=True, scale_each=True)
    return grid


def main(mthd, run):
    # read input arguments
    parser = argparse.ArgumentParser(description='Neural-based Estimation of Divergences between Gaussians')
    parser.add_argument('--method', default='KLD-DV', type=str, metavar='method',
                        help='values: IPM, KLD-DV, KLD-LT, squared-Hel-LT, chi-squared-LT, JS-LT, alpha-LT, Renyi-DV, Renyi-CC, rescaled-Renyi-CC, Renyi-CC-WCR')
    parser.add_argument('--disc_steps_per_gen_step', default=3, type=int)
    parser.add_argument('--batch_size', default=256, type=int, metavar='m')
    parser.add_argument('--lr', default=2e-4, type=float)
    parser.add_argument('--epochs', default=100, type=int,
                        help='number of total epochs to run')
    parser.add_argument('--alpha', default=2.0, type=float, metavar='alpha')
    parser.add_argument('--Lip_constant', default=1.0, type=float, metavar='Lipschitz constant')
    parser.add_argument('--gp_weight', default=1.0, type=float, metavar='GP weight')
    parser.add_argument('--spectral_norm', choices=('True','False'), default='False')
    parser.add_argument('--bounded', choices=('True','False'), default='False')
    parser.add_argument('--reverse_order', choices=('True','False'), default='False')
    parser.add_argument('--use_GP', choices=('True','False'), default='True')
    parser.add_argument('--save_model', choices=('True','False'), default='False', type=str, metavar='save_model')
    parser.add_argument('--save_model_path', default='./trained_models/', type=str, metavar='save_model_path')
    parser.add_argument('--load_model', choices=('True','False'), default='False', type=str, metavar='load_model')  
    parser.add_argument('--load_model_path', default='trained_models', type=str, metavar='load_model_path')
    parser.add_argument('--run_number', default=1, type=int, metavar='run_num')
    parser.add_argument('--conditional', choices=('True','False'), default='True', type=str, metavar='conditional')

    opt = parser.parse_args()
    opt_dict = vars(opt)
    print('parsed options:', opt_dict)

    # mthd = opt_dict['method']
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
    save_model = opt_dict['save_model']=='True'
    save_model_path = opt_dict['save_model_path']
    load_model = opt_dict['load_model']=='True'
    load_model_path = opt_dict['load_model_path']
    conditional = opt_dict['conditional']=='True'
    
    dataset = 'mnist'
    print("Spectral_norm: "+str(spec_norm))
    print("Bounded: "+str(bounded))
    print("Reversed: "+str(reverse_order))
    print("Use Gradient Penalty: "+str(use_GP))
        
    transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
    trainset = torchvision.datasets.MNIST(root='data', train=True, download=True, transform=transform)
    testset = torchvision.datasets.MNIST(root='data', train=False, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=m, shuffle=True, num_workers=4, drop_last=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=m, shuffle=False, num_workers=4, drop_last=True)

    if conditional:
        generator = Generator_MNIST_cond().to(device)
        discriminator = Discriminator_MNIST_cond().to(device)
    else:
        generator = Generator_MNIST().to(device)
        discriminator = Discriminator_MNIST().to(device)
    
    summary(generator)
    print()
    summary(discriminator)
    
    print('Using device:', device)
    
    if optimizer == 'RMS':
        gen_opt = optim.RMSprop(generator.parameters(), lr=lr)
        disc_opt = optim.RMSprop(discriminator.parameters(), lr=lr)
    elif optimizer == 'Adam':
        gen_opt = optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
        disc_opt = optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))
    
    # construct gradient penalty
    if use_GP:
        discriminator_penalty=GradientPenalty(gp_weight, L)
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

    if mthd=="Renyi-CC-WCR":
        div_dense = Renyi_Divergence_WCR(discriminator, disc_opt, alpha, epochs, m, fl_act_func_CC, discriminator_penalty)

    def noise_source(batch_size):
        return torch.randn(batch_size, 124, 1, 1).to(device)

    # GAN_model = GAN_CIFAR10(div_dense, generator, gen_opt, noise_source, epochs, disc_steps_per_gen_step, mthd, dataset, m, reverse_order)
    
    # Load the model if specified
    # if load_model:
    #     load_model_path = os.path.join(load_model_path, f'{mthd}_{dataset}.pt')
    #     GAN_model.load(load_model_path)
    #     print(f"Model loaded from {load_model_path}")
    # else:
    #     generator_samples, loss_array, gen_losses, disc_losses, mean_scores, std_scores = GAN_model.train(train_loader, save_frequency, num_gen_samples_to_save)
    fid_scores = []
    
    if load_model:
        generated_images_path = f'generated_digits/'
        load_model_path = f'{load_model_path}_MNIST'
        if conditional:
            generated_images_path = f'generated_digits_conditional/'
            load_model_path = f'{load_model_path}_conditional_MNIST'
            
        
        checkpoint = torch.load(f'{load_model_path}/{mthd}_best_fid.pt')
        generator.load_state_dict(checkpoint['generator_state_dict'])
        discriminator.load_state_dict(checkpoint['discriminator_state_dict'])

        if conditional:
            grid = generate_labeled_mnist(generator, num_rows=10)
        else:
            grid = generate_random_digits(generator, num_rows=10)
        
        if not os.path.exists(generated_images_path):
            os.makedirs(generated_images_path)
            
        save_image(torch.tensor(grid), generated_images_path + f'generated_digits_{mthd}.png')
        print(f'Saved generated images to {generated_images_path}generated_digits_{mthd}.png')
    else:
        gen_losses = np.zeros(epochs)
        disc_losses = np.zeros(epochs)
        mean_scores = np.zeros(epochs)
        std_scores = np.zeros(epochs)
        iter = 0
        
        fid = FrechetInceptionDistance().to(device)
        
        sample_images_path = ''
        if use_GP:
            if conditional:
                sample_images_path = f'samples_{mthd}_GP_{dataset}_conditional/'
            else:
                sample_images_path = f'samples_{mthd}_GP_{dataset}/'
                
            if not os.path.exists(sample_images_path):
                os.makedirs(sample_images_path)
        else:
            if conditional:
                sample_images_path = f'samples_{mthd}_{dataset}_conditional/'
            else:
                sample_images_path = f'samples_{mthd}_{dataset}/'
            if not os.path.exists(sample_images_path):
                os.makedirs(sample_images_path)

        for epoch in tqdm(range(epochs), desc='Epochs'):
            d_loss = 0
            g_loss = 0
            
            for images, labels in trainloader:
                y = F.one_hot(labels, 10)
                images, y = images.to(device), y.to(device)
                
                if not conditional:
                    y = None
                    
                # Train the discriminator
                for p in discriminator.parameters():
                    p.requires_grad = True
                for p in generator.parameters():
                    p.requires_grad = False
                
                if conditional:
                    samples = generator(y, m)
                else:
                    samples = generator(m)
                    
                for _ in range(disc_steps_per_gen_step):
                    d_loss = div_dense.train_step(images, samples, y)

                # Train the generator
                for p in discriminator.parameters():
                    p.requires_grad = False
                for p in generator.parameters():
                    p.requires_grad = True
                gen_opt.zero_grad()
                
                if conditional:
                    samples_ = generator(y, m)
                else:
                    samples_ = generator(m)
                    
                g_loss = div_dense.generator_loss(samples_, y)
                g_loss.backward()
                gen_opt.step()
                
                if conditional:
                    d_samples = discriminator(samples_, y)
                else:
                    d_samples = discriminator(samples_)
                    
                iter += 1
                if iter % 150 == 0:
                    idx = torch.randint(0, d_samples.shape[0], (16,))
                    grid = utils.make_grid(samples_.cpu()[idx])
                    utils.save_image(grid, sample_images_path + '/samples'+str(epoch)+'.png')
            print('Epoch:', epoch, 'Discriminator Loss:', d_loss.item(), 'Generator Loss:', g_loss.item()) 
            if math.isnan(d_loss.item()) or math.isnan(g_loss.item()):
                break
            # Calculating the FID for each epoch
            
            # Before updating FID, ensure images are in the correct format
            def preprocess_for_fid(images):
                # If images are in range [0, 1] with dtype=torch.float32
                images = (images * 255).to(torch.uint8)
                return images
            
            # Function to convert grayscale to RGB
            def convert_to_rgb(images):
                return images.repeat(1, 3, 1, 1)

            if conditional:
                fake_images = generator(y, m)
            else:
                fake_images = generator(m)
            real_images = images
            
            # Assuming you have real_images and fake_images
            real_images_preprocessed = preprocess_for_fid(convert_to_rgb(real_images))
            fake_images_preprocessed = preprocess_for_fid(convert_to_rgb(fake_images))

            # Update FID
            fid.update(real_images_preprocessed, real=True)
            fid.update(fake_images_preprocessed, real=False)

            fid_score = fid.compute()
            fid_scores.append(fid_score.item())
            fid.reset()

            print(f'FID Score: {fid_score.item()}')

            if save_model and fid_score.item() == min(fid_scores):
                    if conditional:
                        save_model_path = f'trained_models_conditional_MNIST/'
                    else:
                        save_model_path = f'trained_models_MNIST/'
                        
                    if not os.path.exists(save_model_path):
                        os.makedirs(save_model_path)
                        
                    torch.save({
                        'generator_state_dict': generator.state_dict(),
                        'discriminator_state_dict': discriminator.state_dict(),
                    }, save_model_path + f'{mthd}_best_fid.pt')
                        
                    print(f'Model saved to {save_model_path} + {mthd}.pt')

            
        # # Save the loss vs epoch plot
        # epoch_ax = np.arange(start=1, stop=epochs+1, step=1)
        # _, ax = plt.subplots(1, 3, figsize=(15, 5))

        # ax[0].plot(epoch_ax, disc_losses, color='blue')
        # ax[0].set_xlim(1, epochs)
        # ax[0].set_title("Discriminator Loss vs Epoch")
        # ax[0].grid()

        # ax[1].plot(epoch_ax, gen_losses, color='red')
        # ax[1].set_xlim(1, epochs)
        # ax[1].set_title("Generator Loss vs Epoch")
        # ax[1].grid()

        # ax[2].plot(epoch_ax, mean_scores, color='green', label='Inception Score Mean')
        # ax[2].fill_between(epoch_ax, mean_scores-std_scores, mean_scores+std_scores, color='green', alpha=0.2, label='Inception Score Std')
        # ax[2].set_xlim(1, epochs)
        # ax[2].set_title("Inception Score Mean and Std vs Epoch")
        # ax[2].grid()
        # ax[2].legend()

        # plt.tight_layout()
        
        # if save_model:
        #     if conditional:
        #         save_model_path = f'trained_models_conditional_MNIST/{mthd}'
        #     else:
        #         save_model_path = f'trained_models_MNIST/{mthd}'
        #     if not os.path.exists(save_model_path):
        #         os.makedirs(save_model_path)
        #     torch.save({
        #         'generator_state_dict': generator.state_dict(),
        #         'discriminator_state_dict': discriminator.state_dict(),
        #     }, save_model_path + '.pt')
            
        #     print(f'Model saved to {save_model_path}.pt')
            
        # if use_GP:
        #     if not os.path.exists(f'losses_{mthd}_GP_{dataset}/'):
        #         os.makedirs(f'losses_{mthd}_GP_{dataset}/')

        #     plt.savefig(f'losses_{mthd}_GP_{dataset}/loss_vs_epoch.png')
        # else:
        #     if not os.path.exists(f'losses_{mthd}_{dataset}/'):
        #         os.makedirs(f'losses_{mthd}_{dataset}/')

        #     plt.savefig(f'losses_{mthd}_{dataset}/loss_vs_epoch.png')
        #     plt.show()
        #     plt.close() 
    
    if not os.path.exists(f'fid_scores_with_runs/'):
        os.makedirs(f'fid_scores_with_runs/')
        
    with open(f'fid_scores_with_runs/fid_scores_{mthd}_{run}.csv', 'w') as f:
        writer = csv.writer(f, lineterminator='\n')
        for fid_score in fid_scores:
            writer.writerow([fid_score])
    return fid_scores
    
    
if __name__ == '__main__':
    methods = ['chi-squared-HCR']
    # methods = ['Renyi-CC', 'rescaled-Renyi-CC', 'Renyi-CC-WCR']
    runs = 3

    # for method in methods:
    #     for run in range(1, runs+1):
    #         fid_scores = main(method, run)
    #         print(f'Completed training for {method}')
    fid_scores = main('chi-squared-HCR', 3)
    print(f'COmpleted training for chi-squared-HCR')