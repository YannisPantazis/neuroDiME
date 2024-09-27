import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, utils
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
import torch.autograd as autograd
import torchvision
import torch.nn.functional as F
import torch.nn.init as nninit
from torchmetrics.image.fid import FrechetInceptionDistance
import json

current_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from models.GAN_CIFAR10_torch import *
from models.Divergences_torch import *

start = time.perf_counter()

fl_act_func_CC = 'poly-softplus' # abs, softplus, poly-softplus
optimizer = "Adam" #Adam, RMS
save_frequency = 10 #generator samples are saved every save_frequency epochs
num_gen_samples_to_save = 5000

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def generate_labeled_cifar10(generator, num_rows=10):
    generator.eval()
    num_classes = 10
    
    for i in range(num_classes):
        labels = torch.tensor([i] * num_rows).to(device)
        noise = torch.randn(num_rows, 128).to(device)
        samples = generator(noise, labels)
        samples = samples.view(-1, 3, 32, 32)
        samples = (samples + 1) / 2
        grid = utils.make_grid(samples, nrow=num_rows)
        if i == 0:
            all_samples = grid
        else:
            all_samples = torch.cat((all_samples, grid), dim=1)

    return all_samples


def generate_images_cifar10(generator, num_rows=10):
    generator.eval()
    noise = torch.randn(num_rows * 10, 128).to(device)
    samples = generator(noise)
    samples = samples.view(-1, 3, 32, 32)
    samples = (samples + 1) / 2
    grid = utils.make_grid(samples, nrow=num_rows)
    return grid


class GradientPenalty(Discriminator_Penalty):
    def __init__(self, gp_weight, L):
        Discriminator_Penalty.__init__(self, gp_weight)
        self.L = L
    
    def get_Lip_constant(self):
        return self.L

    def set_Lip_constant(self, L):
        self.L = L
        
    def evaluate(self, discriminator, real_var, fake_var, labels):
        assert real_var.size(0) == fake_var.size(0), \
            'expected real and fake data to have the same batch size'

        batch_size = real_var.size(0)
        alpha = torch.rand(batch_size, 1, 1, 1)
        alpha = alpha.expand_as(real_var)
        alpha = alpha.type_as(real_var)

        interp_data = alpha * real_var + ((1 - alpha) * fake_var)
        interp_data = autograd.Variable(interp_data, requires_grad=True)

        if labels is not None:
            disc_out = discriminator(interp_data, labels)
        else:
            disc_out = discriminator(interp_data, labels)
        grad_outputs = torch.ones(disc_out.size()).type_as(disc_out.data)

        gradients = autograd.grad(
            outputs=disc_out,
            inputs=interp_data,
            grad_outputs=grad_outputs,
            create_graph=True,
            only_inputs=True)[0]

        gradients = gradients.view(batch_size, -1)

        gradient_penalty = self.get_penalty_weight() * ((gradients.norm(2, dim=1) - self.L) ** 2).mean()

        return gradient_penalty


def main(mthd):
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
    parser.add_argument('--use_GP', choices=('True','False'), default='False')
    parser.add_argument('--save_model', choices=('True','False'), default='False', type=str, metavar='save_model')
    parser.add_argument('--save_model_path', default='./trained_models/', type=str, metavar='save_model_path')
    parser.add_argument('--load_model', choices=('True','False'), default='False', type=str, metavar='load_model')  
    parser.add_argument('--load_model_path', default='trained_models/', type=str, metavar='load_model_path')
    parser.add_argument('--run_number', default=1, type=int, metavar='run_num')
    parser.add_argument('--conditional', choices=('True','False'), default='False', type=str, metavar='conditional')
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
    dataset = 'cifar10'
    save_model = opt_dict['save_model']=='True'
    save_model_path = opt_dict['save_model_path'] + f"_{dataset}"
    load_model = opt_dict['load_model']=='True'
    load_model_path = opt_dict['load_model_path']
    conditional = opt_dict['conditional']=='True'
    
    print("Spectral_norm: "+str(spec_norm))
    print("Bounded: "+str(bounded))
    print("Reversed: "+str(reverse_order))
    print("Use Gradient Penalty: "+str(use_GP))
        
    transform = transforms.Compose([
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                    ])
    
    # Create the dataset
    trainset = torchvision.datasets.CIFAR10(root='./data', download=True, transform=transform, train=True)
    testset = torchvision.datasets.CIFAR10(root='./data', download=True, transform=transform, train=False)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=m, shuffle=True, num_workers=4, drop_last=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=m, shuffle=False, num_workers=4, drop_last=True) 
    
    if conditional:
        generator = Generator_cond().to(device)
        discriminator = Discriminator_cond().to(device)
    else:
        generator = Generator().to(device)
        discriminator = Discriminator().to(device)
    
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
        return torch.randn(batch_size, 128).to(device)

    # GAN_model = GAN_CIFAR10(div_dense, generator, gen_opt, noise_source, epochs, disc_steps_per_gen_step, mthd, dataset, m, reverse_order)
    
    # Load the model if specified
    # if load_model:
    #     load_model_path = os.path.join(load_model_path, f'{mthd}_{dataset}.pt')
    #     GAN_model.load(load_model_path)
    #     print(f"Model loaded from {load_model_path}")
    # else:
    #     generator_samples, loss_array, gen_losses, disc_losses, mean_scores, std_scores = GAN_model.train(train_loader, save_frequency, num_gen_samples_to_save)
    
    gen_losses = np.zeros(epochs)
    disc_losses = np.zeros(epochs)
    mean_scores = np.zeros(epochs)
    std_scores = np.zeros(epochs)
    iter = 0
    
    fid = FrechetInceptionDistance().to(device)
    
    sample_images_path = ''
    if conditional:
        if use_GP:
            sample_images_path = f'samples_{mthd}_GP_{dataset}_conditional/'
            if not os.path.exists(sample_images_path):
                os.makedirs(sample_images_path)
        else:
            sample_images_path = f'samples_{mthd}_{dataset}_conditional/'
            if not os.path.exists(sample_images_path):
                os.makedirs(sample_images_path)
    else:
        if use_GP:
            sample_images_path = f'samples_{mthd}_GP_{dataset}/'
            if not os.path.exists(sample_images_path):
                os.makedirs(sample_images_path)
        else:
            sample_images_path = f'samples_{mthd}_{dataset}/'
            if not os.path.exists(sample_images_path):
                os.makedirs(sample_images_path)

    if load_model:
        generated_images_path = f'generated_images_{dataset}/'
        load_model_path = f'trained_models_CIFAR10/'
        if conditional:
            load_model_path = f'trained_models_conditional_CIFAR10/'
            generated_images_path = f'generated_images_{dataset}_conditional/'
        
        if not os.path.exists(f'{load_model_path}/{mthd}_best_fid.pt'):
            checkpoint = torch.load(f'{load_model_path}/{mthd}.pt')
        else:
            checkpoint = torch.load(f'{load_model_path}/{mthd}_best_fid.pt')
        generator.load_state_dict(checkpoint['generator_state_dict'])
        discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        
        if conditional:
            grid = generate_labeled_cifar10(generator, num_rows=10)
        else:
            grid = generate_images_cifar10(generator, num_rows=10)
        
        if not os.path.exists(generated_images_path):
            os.makedirs(generated_images_path)
            
        save_image(torch.tensor(grid), f'{generated_images_path}{mthd}.png', normalize=True)
        print(f'Saved generated images to {generated_images_path}{mthd}.png')
        return []
    else:
        fid_scores = []
        for epoch in tqdm(range(epochs), desc='Epochs'):
            d_loss = 0
            g_loss = 0
            
            for images, labels in trainloader:
                images, labels = images.to(device), labels.to(device)

                # Train the discriminator
                for p in discriminator.parameters():
                    p.requires_grad = True
                for p in generator.parameters():
                    p.requires_grad = False
                    
                noise = noise_source(m)
                samples = generator(noise, labels)
                for _ in range(disc_steps_per_gen_step):
                    d_loss = div_dense.train_step(images, samples, labels)

                # Train the generator
                for p in discriminator.parameters():
                    p.requires_grad = False
                for p in generator.parameters():
                    p.requires_grad = True
                    
                gen_opt.zero_grad()
                samples_ = generator(noise, labels)
                g_loss = div_dense.generator_loss(samples_, labels)
                g_loss.backward()
                gen_opt.step()
                
                gen_losses[epoch] += g_loss.item()
                disc_losses[epoch] += d_loss.item()
                iter += 1
                if iter % 150 == 0:
                    fake = generator(noise, labels)
                    save_image(fake.detach(), f'{sample_images_path}sample_{epoch}.png', normalize=True)
            gen_losses[epoch] = g_loss.item() / len(trainloader)
            disc_losses[epoch] = d_loss.item() / len(trainloader)
            print('Epoch:', epoch, 'Discriminator Loss:', d_loss.item(), 'Generator Loss:', g_loss.item())  
            if math.isnan(d_loss.item()) or math.isnan(g_loss.item()):
                break
            
            # Calculating the FID for each epoch
                
            # Before updating FID, ensure images are in the correct format
            def preprocess_for_fid(images):
                # If images are in range [0, 1] with dtype=torch.float32
                images = (images * 255).to(torch.uint8)
                return images

            if conditional:
                fake_images = generator(noise, labels)
            else:
                fake_images = generator(noise)
            real_images = images
                
            # Assuming you have real_images and fake_images
            real_images_preprocessed = preprocess_for_fid(real_images)
            fake_images_preprocessed = preprocess_for_fid(fake_images)

            # Update FID
            fid.update(real_images_preprocessed, real=True)
            fid.update(fake_images_preprocessed, real=False)

            fid_score = fid.compute()
            fid_scores.append(fid_score.item())
            fid.reset()

            print(f'FID Score: {fid_score.item()}')
            
            if save_model and fid_score.item() == min(fid_scores):
                    if conditional:
                        save_model_path = f'trained_models_conditional_CIFAR10/'
                    else:
                        save_model_path = f'trained_models_CIFAR10/'
                        
                    if not os.path.exists(save_model_path):
                        os.makedirs(save_model_path)
                        
                    torch.save({
                        'generator_state_dict': generator.state_dict(),
                        'discriminator_state_dict': discriminator.state_dict(),
                    }, save_model_path + f'{mthd}_best_fid.pt')
                        
                    print(f'Model saved to {save_model_path} + {mthd}.pt')
            
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
        
        # if save_model:
        #     if conditional:
        #         save_model_path = f'trained_models_conditional_CIFAR10/'
        #     else:
        #         save_model_path = f'trained_models_CIFAR10/'
                
        #     if not os.path.exists(save_model_path):
        #         os.makedirs(save_model_path)
                
        #     torch.save({
        #         'generator_state_dict': generator.state_dict(),
        #         'discriminator_state_dict': discriminator.state_dict(),
        #     }, save_model_path + f'{mthd}.pt')
                
        #     print(f'Model saved to {save_model_path} + {mthd}.pt')

        if use_GP:
            if not os.path.exists(f'losses_{mthd}_GP_{dataset}/'):
                os.makedirs(f'losses_{mthd}_GP_{dataset}/')

            plt.savefig(f'losses_{mthd}_GP_{dataset}/loss_vs_epoch.png')
        else:
            if not os.path.exists(f'losses_{mthd}_{dataset}/'):
                os.makedirs(f'losses_{mthd}_{dataset}/')

            plt.savefig(f'losses_{mthd}_{dataset}/loss_vs_epoch.png')
            plt.show()
            plt.close()    
        
        return fid_scores
    
if __name__ == '__main__':
    methods = ["IPM", "KLD-DV", "KLD-LT", "squared-Hel-LT", "chi-squared-LT", "JS-LT", "alpha-LT", "Renyi-DV", "Renyi-CC", "rescaled-Renyi-CC", "Renyi-CC-WCR"]
    for mthd in methods:
        fid_scores = []
        fid_scores = main(mthd)
        
        with open(f"fid_scores_{mthd}.json", "w") as file:
            json.dump(fid_scores, file)
