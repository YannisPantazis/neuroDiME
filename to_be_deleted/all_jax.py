from typing import Any
import jax
import flax
from flax.training import train_state
from flax import linen as nn
import jax.numpy as jnp
import numpy as np
import argparse
import optax
from functools import partial
import time
from jax import jit 


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


class Discriminator(nn.Module):

    input_dim: int
    spec_norm: bool
    bounded: bool
    layers_list: list

    def bounded_activation(x):
        M = 100.0
        return M * jnp.tanh(x / M)

    @nn.compact
    def __call__(self, x):

        if self.spec_norm:
            for h_dim in self.layers_list:
                x = nn.SpectralNorm(nn.Dense(h_dim))(x)
                x = nn.relu(x)
            x = nn.SpectralNorm(nn.Dense(1)(x))
        else:
            for h_dim in self.layers_list:
                x = nn.Dense(h_dim)(x)
                x = nn.relu(x)
            x = nn.Dense(1)(x)
    
        if self.bounded:
            x = self.bounded_activation(x)
        
        return x
    

class KLD_DV:

    def __init__(self, discriminator, disc_optimizer, epochs, batch_size):
        self.batch_size = batch_size
        self.epochs = epochs
        self.discriminator = discriminator
        self.disc_optimizer = disc_optimizer


    def discriminate(self, x, params):
        y = self.discriminator.apply(params, x)
        return y


    def eval_var_formula(self, x, y, params):
        ''' Evaluation of variational formula of KL divergence (based on the Donsker-Varahdan variational formula), KL(P||Q), x~P, y~Q.'''
        D_P = self.discriminate(x, params)
        D_Q = self.discriminate(y, params)
        
        D_loss_P = jnp.mean(D_P)
        
        max_val = jnp.max(D_Q)
        D_loss_Q = jnp.log(jnp.mean(jnp.exp(D_Q - max_val))) + max_val

        D_loss = D_loss_P - D_loss_Q
        return D_loss
   

    def estimate(self, x, y, params):
        divergence_loss = self.eval_var_formula(x, y, params)
        return divergence_loss
    

    def discriminator_loss(self, x, y, params):
        divergnece_loss = self.eval_var_formula(x, y, params)
        return divergnece_loss
    

    @partial(jit, static_argnums=(0,))
    def train_step(self, x, y, params, opt_state):

        def loss_fn(params, x, y):
            loss = -self.discriminator_loss(x, y, params)

            if self.discriminator_penalty is not None:
                loss += self.discriminator_penalty.evaluate(self.discriminator, x, y)
                
            return loss
        
        grads = jax.grad(loss_fn, allow_int=True)(params, x, y)
        updates, opt_state = disc_optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)

        return opt_state, params

        
    def train(self, data_P, data_Q, params, opt_state):
        P_dataset = DataLoader(data_P, batch_size=self.batch_size, shuffle=True)
        Q_dataset = DataLoader(data_Q, batch_size=self.batch_size, shuffle=True)

        estimates = []

        for i in range(self.epochs):
            for P_batch, Q_batch in zip(P_dataset, Q_dataset):
                opt_state, params = self.train_step(P_batch, Q_batch, params, opt_state)

            estimates.append(float(self.estimate(P_batch, Q_batch, params)))    

        return estimates, params


start = time.perf_counter()

# Read input arguments
parser = argparse.ArgumentParser(description='Neural-based Estimation of Divergences between Gaussians')
parser.add_argument('--dimension', default=1, type=int, metavar='d')
parser.add_argument('--method', default='KLD-DV', type=str, metavar='method',
                    help='values: KLD-DV, KLD-DV-GP, KLD-LT, squared-Hel-LT, chi-squared-LT, JS-LT, alpha-LT, Renyi-DV, Renyi-CC, rescaled-Renyi-CC, Renyi-CC-WCR')
parser.add_argument('--sample_size', default=10000, type=int, metavar='N')
parser.add_argument('--batch_size', default=1000, type=int, metavar='m')
parser.add_argument('--lr', default=0.001, type=float)
parser.add_argument('--epochs', default=100, type=int,
                    help='number of total epochs to run')
parser.add_argument('--alpha', default=2.0, type=float, metavar='alpha')
parser.add_argument('--Lip_constant', default=1.0, type=float, metavar='Lipschitz constant')
parser.add_argument('--gp_weight', default=1.0, type=float, metavar='GP weight')
parser.add_argument('--use_GP', choices=('True', 'False'), default='False')
parser.add_argument('--delta_mu', default=1.0, type=float)
parser.add_argument('--run_number', default=1, type=int, metavar='run_num')
opt = parser.parse_args()
opt_dict = vars(opt)
print('parsed options:', opt_dict)

# Set up variables from arguments
mthd = opt_dict['method']
N = opt_dict['sample_size']
d = opt_dict['dimension']
m = opt_dict['batch_size']
lr = opt_dict['lr']
epochs = opt_dict['epochs']
alpha = opt_dict['alpha']
L = opt_dict['Lip_constant']
gp_weight = opt_dict['gp_weight']
delta_mu = opt_dict['delta_mu']
run_num = opt_dict['run_number']
use_GP = opt_dict['use_GP'] == 'True'
bounded=False
spec_norm=False

# Create data sets
mu_p = np.zeros((d, 1))
mu_q = np.zeros((d, 1))
mu_q[0] = delta_mu
Sigma_p = np.identity(d)
Sigma_q = np.identity(d)
Sigma_alpha = alpha * Sigma_q + (1.0 - alpha) * Sigma_p
Mp = np.linalg.cholesky(Sigma_p)
Mq = np.linalg.cholesky(Sigma_q)

def sample_P(N_samp):
    return jnp.transpose((mu_p + jnp.matmul(Mp, np.random.normal(0., 1.0, size=[d, N_samp]))))

def sample_Q(N_samp):
    return jnp.transpose((mu_q + jnp.matmul(Mq, np.random.normal(0., 1.0, size=[d, N_samp]))))

data_P = sample_P(N).astype('f')
data_Q = sample_Q(N).astype('f')

layers_list = [64]
act_func = 'relu'
print(f'Predicting the {mthd} divergence using JaX\n')
discriminator = Discriminator(input_dim=d, spec_norm=False, bounded=False, layers_list=layers_list)

x = jnp.ones((m, d))
test = nn.tabulate(discriminator, jax.random.key(0))
print(test(x))

# Initialize the model's parameters with a dummy input
rng = jax.random.PRNGKey(0)
params = discriminator.init(rng, x)
print(jax.tree_map(lambda x: x.shape, params)) # Check the parameters
optimizer = "RMS"  # Adam, RMS

# Construct optimizers
if optimizer == 'Adam':
    disc_optimizer = optax.adam(lr)

if optimizer == 'RMS':
    disc_optimizer = optax.rmsprop(lr)

opt_state = disc_optimizer.init(params)

# print('Before training')
# print(params)

div_dense = KLD_DV(discriminator, disc_optimizer, epochs, m)
estimates, params = div_dense.train(data_P, data_Q, params, opt_state)

# print()
# print('After training')
# print(params)

div_value_true = float(1.0 / 2.0 * (np.log(np.abs(np.linalg.det(Sigma_q) / np.linalg.det(Sigma_p))) + np.matmul(np.transpose(mu_q - mu_p), np.matmul(np.linalg.inv(Sigma_q), mu_q - mu_p)) - d + np.trace(np.matmul(Sigma_p, np.linalg.inv(Sigma_q)))))
print('KLD (true):\t\t {:.4}'.format(div_value_true))
print('KLD-DV (estimated):\t {:.4}'.format(estimates[-1]))
print()

end = time.perf_counter()
print(end - start)