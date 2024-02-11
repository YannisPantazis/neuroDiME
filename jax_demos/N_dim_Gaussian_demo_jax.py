import jax.numpy as jnp
import numpy as np
import os
import sys
import csv
import argparse
import optax

# Add parent directory to sys.path
current_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from models.jax_model import *
from models.Divergences_jax import *

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

# Construct the discriminator neural network
layers_list = [64]  # UNCECOMP's NN: [16, 16, 8]
# layers_list = [16, 16, 8]
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

state = train_state.TrainState.create(
            apply_fn=discriminator.apply, 
            params=params["params"], 
            tx=disc_optimizer)

# opt_state = disc_optimizer.init(params)

# disc_optimizer_state = disc_optimizer.init(params)

# Construct gradient penalty
if use_GP:
    discriminator_penalty = Gradient_Penalty_1Sided(gp_weight, L)
else:
    discriminator_penalty = None

# Construct divergence, train optimizer and estimate the divergence
if mthd == "KLD-DV":
    div_dense = KLD_DV(discriminator, disc_optimizer, epochs, m, state, discriminator_penalty)
    divergence_estimates = div_dense.train(data_P, data_Q)
    div_value_true = float(1.0 / 2.0 * (np.log(np.abs(np.linalg.det(Sigma_q) / np.linalg.det(Sigma_p))) + np.matmul(np.transpose(mu_q - mu_p), np.matmul(np.linalg.inv(Sigma_q), mu_q - mu_p)) - d + np.trace(np.matmul(Sigma_p, np.linalg.inv(Sigma_q)))))
    print('KLD (true):\t\t {:.4}'.format(div_value_true))
    print('KLD-DV (estimated):\t {:.4}'.format(divergence_estimates[-1]))
    print()

# Handle other methods similarly

test_name = "N_dim_Gaussian_demo"
if not os.path.exists(test_name):
    os.makedirs(test_name)

with open(test_name + '/estimated_' + mthd + '_dim_' + str(d) + '_delta_mu_{:.2f}'.format(delta_mu) + '_N_' + str(N) + '_m_' + str(m) + '_Lrate_{:.1e}'.format(lr) + '_epochs_' + str(epochs) + '_alpha_{:.1f}'.format(alpha) + '_L_{:.1f}'.format(L) + '_gp_weight_{:.1f}'.format(gp_weight) + '_spec_norm_' + str(spec_norm) + '_bounded_' + str(bounded) + '_run_num_' + str(run_num) + '.csv', "w") as output:
    writer = csv.writer(output, lineterminator='\n')
    for div_value_est in divergence_estimates:
        writer.writerow([div_value_est])

with open(test_name + '/true_' + mthd + '_dim_' + str(d) + '_delta_mu_{:.2f}'.format(delta_mu) + '_N_' + str(N) + '_m_' + str(m) + '_Lrate_{:.1e}'.format(lr) + '_epochs_' + str(epochs) + '_alpha_{:.1f}'.format(alpha) + '_L_{:.1f}'.format(L) + '_gp_weight_{:.1f}'.format(gp_weight) + '_spec_norm_' + str(spec_norm) + '_bounded_' + str(bounded) + '_run_num_' + str(run_num) + '.csv', "w") as output:
    writer = csv.writer(output, lineterminator='\n')
    writer.writerow([div_value_true])
