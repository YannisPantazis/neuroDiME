import torch
import torch.nn as nn
from tqdm import tqdm

class GAN():
    '''
    Class for training a GAN using one of the provided divergences
    If reverse_order=False the GAN works to minimize min_theta D(P||g_theta(Z)) where P is the distribution to be leared, Z is the noise source and g_theta is the generator (with parameters theta).
    If reverse_order=True the GAN works to minimize min_theta D(g_theta(Z)||P) where P is the distribution to be leared, Z is the noise source and g_theta is the generator (with parameters theta).
    '''
    # initialize
    def __init__(self, divergence, generator, gen_optimizer, noise_source, epochs, disc_steps_per_gen_step, batch_size=None, reverse_order=False, include_penalty_in_gen_loss=False):
        self.divergence = divergence # Variational divergence
        self.generator = generator
        self.epochs = epochs
        self.disc_steps_per_gen_step = disc_steps_per_gen_step
        self.gen_optimizer = gen_optimizer
        self.reverse_order = reverse_order
        self.include_penalty_in_gen_loss = include_penalty_in_gen_loss
        self.noise_source = noise_source

        if batch_size is None:
            self.batch_size = self.divergence.batch_size
        else:
            self.batch_size = batch_size
        
    def estimate_loss(self, x, z):
        ''' Estimating the loss '''
        # z = torch.from_numpy(z).float()
        if self.reverse_order:
            data1 = self.generator(z)
            data2 = x
        else:
            data1 = x
            data2 = self.generator(z)

        return self.divergence.estimate(data1, data2)
    
    def gen_train_step(self, x, z):
        ''' generator's parameters update '''
        self.gen_optimizer.zero_grad()
        # x.requires_grad_(True)
        # z.requires_grad_(True)

        # z = torch.from_numpy(z).float()
        if self.reverse_order:
            data1 = self.generator(z)
            data2 = x
        else:
            data1 = x
            data2 = self.generator(z)

        loss = self.divergence.discriminator_loss(data1, data2)
        if self.include_penalty_in_gen_loss and self.divergence.discriminator_penalty is not None:
            loss = loss - self.divergence.discriminator_penalty.evaluate(self.divergence.discriminator, data1, data2)
        
        loss.backward()
        self.gen_optimizer.step()

    def disc_train_step(self, x, z):
        ''' discriminator's parameters update '''
        # z = torch.from_numpy(z).float()
        if self.reverse_order:
            data1 = self.generator(z)
            data2 = x
        else:
            data1 = x
            data2 = self.generator(z)
        
        self.divergence.train_step(data1, data2)

    def train(self, data_P, save_frequency=None, num_gen_samples_to_save=None, save_loss_estimates=False, device = 'cuda' if torch.cuda.is_available() else 'cpu'):
        ''' training function of our GAN '''
        # dataset slicing into minibatches
        P_dataset = torch.utils.data.DataLoader(data_P, batch_size=self.batch_size, shuffle=True)

        generator_samples = []
        loss_estimates = []
        for epoch in tqdm(range(self.epochs), desc='Epochs'):
            for P_batch in P_dataset:
                Z_batch = torch.from_numpy(self.noise_source(self.batch_size)).float()
                P_batch = P_batch.to(device)
                Z_batch = Z_batch.to(device)
                
                for disc_step in range(self.disc_steps_per_gen_step):
                    self.disc_train_step(P_batch, Z_batch)
                
                self.gen_train_step(P_batch, Z_batch)
            
            if save_frequency is not None and (epoch+1) % save_frequency == 0:
                if num_gen_samples_to_save is not None:
                    generator_samples.append(self.generate_samples(num_gen_samples_to_save))
                
                if save_loss_estimates:
                    loss_estimates.append(float(self.estimate_loss(P_batch, Z_batch)))

        return generator_samples, loss_estimates
    
    def generate_samples(self, N_samples, device = 'cuda' if torch.cuda.is_available() else 'cpu'):
        samples = torch.from_numpy(self.noise_source(N_samples)).float()
        samples = samples.to(device)
        generator_samples = self.generator(samples)
        return generator_samples.cpu().detach().numpy()
    