import jax
import jax.numpy as jnp
import time
import math
from optax import apply_updates
from functools import partial
from jax import jit
from tqdm import tqdm

class GAN():
    '''
    Class for training a GAN using one of the provided divergences
    If reverse_order=False the GAN works to minimize min_theta D(P||g_theta(Z)) where P is the distribution to be leared, Z is the noise source and g_theta is the generator (with parameters theta).
    If reverse_order=True the GAN works to minimize min_theta D(g_theta(Z)||P) where P is the distribution to be leared, Z is the noise source and g_theta is the generator (with parameters theta).
    '''   
    # initialize
    def __init__(self, divergence, generator, gen_optimizer, noise_source, epochs, disc_steps_per_gen_step, batch_size=None, reverse_order=False, include_penalty_in_gen_loss=False, cnn=False):
        self.divergence = divergence # Variational divergence
        self.generator = generator
        self.epochs = epochs
        self.disc_steps_per_gen_step = disc_steps_per_gen_step
        self.gen_optimizer = gen_optimizer
        self.reverse_order = reverse_order
        self.include_penalty_in_gen_loss = include_penalty_in_gen_loss
        self.noise_source = noise_source
        self.cnn = cnn
        
        if batch_size is None:
            self.batch_size = self.divergence.batch_size
        else:
            self.batch_size = batch_size
            
    def estimate_loss(self, x, z, gen_params):
        ''' Estimating the loss '''
        z = jnp.array(z)
        if self.cnn:
            if self.reverse_order:
                data1 = self.generator.apply(gen_params, z, train=False, rngs={'dropout': jax.random.PRNGKey(0)})
                data2 = x
            else:
                data1 = x
                data2 = self.generator.apply(gen_params, z, train=False, rngs={'dropout': jax.random.PRNGKey(0)})
        else:
            if self.reverse_order:
                data1 = self.generator.apply(gen_params, z)
                data2 = x
            else:
                data1 = x
                data2 = self.generator.apply(gen_params, z)
            
        return self.divergence.estimate(data1, data2)
    
    @partial(jax.jit, static_argnums=(0,))
    def gen_train_step(self, x, z, gen_params, gen_opt_state):
        ''' generator's parameters update '''
        
        z = jnp.array(z)
        if self.reverse_order:
            data1 = self.generator.apply(gen_params, z, train=False, rngs={'dropout': jax.random.PRNGKey(0)})
            data2 = x
        else:
            data1 = x
            data2 = self.generator.apply(gen_params, z, train=False, rngs={'dropout': jax.random.PRNGKey(0)})
        
        def loss_fn(gen_params, data1, data2):
            loss = self.divergence.discriminator_loss(data1, data2, gen_params)

            if self.discriminator_penalty is not None:
                loss -= self.discriminator_penalty.evaluate(self.discriminator, data1, data2, gen_params)
                
            return loss        
        
        grads = jax.grad(loss_fn, allow_int=True)(gen_params, data1, data2)
        updates, gen_opt_state = self.gen_optimizer.update(grads, gen_opt_state)
        gen_params = apply_updates(gen_params, updates)
        
        return gen_opt_state, gen_params
    
    @partial(jax.jit, static_argnums=(0,))
    def disc_train_step(self, x, z, disc_params, disc_opt_state):
        ''' discriminator's parameters update '''
        
        z = jnp.array(z)
        if self.reverse_order:
            data1 = self.generator.apply(disc_params, z, rngs={'dropout': jax.random.PRNGKey(0)})
            data2 = x
        else:
            data1 = x
            data2 = self.generator.apply(disc_params, z, rngs={'dropout': jax.random.PRNGKey(0)})
        
        disc_params, disc_opt_state = self.divergence.train_step(data1, data2, disc_params, disc_opt_state)
        return disc_opt_state, disc_params
        
    def train(self, data_P, disc_params, gen_params, disc_opt_state, gen_opt_state, save_frequency=None,  num_gen_samples_to_save=None, save_loss_estimates=False):
        ''' training function of our GAN '''
        # dataset slicing into minibatches
        P_dataset = DataLoader(data_P, batch_size=self.batch_size, shuffle=True)
        
        generator_samples = []
        loss_estimates = []
        for epoch in tqdm(range(self.epochs), desc='Epochs'):
            for P_batch in P_dataset:
                Z_batch = self.noise_source(self.batch_size)
                
                for _ in range(self.disc_steps_per_gen_step):
                    disc_opt_state, disc_params = self.disc_train_step(P_batch, Z_batch, disc_params, disc_opt_state)
                    
                gen_opt_state, gen_params = self.gen_train_step(P_batch, Z_batch, gen_params, gen_opt_state)
                
                if save_frequency is not None and (epoch+1) % save_frequency == 0:
                    if save_loss_estimates:
                        loss_estimates.append(float(self.estimate_loss(P_batch, Z_batch), gen_params))
                    
                    if num_gen_samples_to_save is not None:
                        generator_samples.append(self.generate_samples(num_gen_samples_to_save), gen_params)
        
        return generator_samples, loss_estimates, disc_params, gen_params, disc_opt_state, gen_opt_state

    def generate_samples(self, num_samples, gen_params):
        generator_samples = self.generator.apply(gen_params, float(self.noise_source(num_samples)))
        return generator_samples
    

class DataLoader:
    def __init__(self, data, batch_size, shuffle=True):
        self.data = data
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_samples = len(data)
        self.index = jnp.arange(self.num_samples)
        if shuffle:
            self.index = jax.random.permutation(jax.random.PRNGKey(0), self.index)

    def __iter__(self):
        self.current_idx = 0
        return self

    def __next__(self):
        if self.current_idx >= self.num_samples:
            raise StopIteration
        batch_idx = self.index[self.current_idx:self.current_idx+self.batch_size]
        batch = jnp.take(self.data, batch_idx, axis=0)
        self.current_idx += self.batch_size
        return batch