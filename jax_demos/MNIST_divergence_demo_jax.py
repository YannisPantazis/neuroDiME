import jax.numpy as jnp
import numpy as np
import os
import sys
import csv
import argparse
import optax
import time
import torchvision.datasets as datasets
import flax.linen as nn
from flax.training import checkpoints

# Add parent directory to sys.path
current_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from models.jax_model import *
from models.Divergences_jax import *

start = time.perf_counter()

# read input arguments
parser = argparse.ArgumentParser(description='Neural-based Estimation of Divergences between MNIST Digit Distributions')
parser.add_argument('--P_digit', type=int)          
parser.add_argument('--Q_digit', type=int)
parser.add_argument('--method', default='KLD-DV', type=str, metavar='method',
                    help='values: IPM, KLD-DV, KLD-LT, squared-Hel-LT, chi-squared-LT, JS-LT, alpha-LT, Renyi-DV, Renyi-CC, rescaled-Renyi-CC, Renyi-CC-WCR')
                        
parser.add_argument('--sample_size', default=10000, type=int, metavar='N')
parser.add_argument('--batch_size', default=1000, type=int, metavar='m')
parser.add_argument('--lr', default=0.001, type=float)
parser.add_argument('--epochs', default=100, type=int,
                    help='number of total epochs to run')
parser.add_argument('--alpha', default=2.0, type=float, metavar='alpha')
parser.add_argument('--Lip_constant', default=1.0, type=float, metavar='Lipschitz constant')
parser.add_argument('--gp_weight', default=10.0, type=float, metavar='GP weight')
parser.add_argument('--use_GP', choices=('True','False'), default='False')
parser.add_argument('--run_number', default=1, type=int, metavar='run_num')

opt = parser.parse_args()
opt_dict = vars(opt)
print('parsed options:', opt_dict)

P_digit = opt_dict['P_digit']
Q_digit = opt_dict['Q_digit']
mthd = opt_dict['method']
N = opt_dict['sample_size']
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
use_GP = opt_dict['use_GP']=='True'

run_num = opt_dict['run_number']


print("Use Gradient Penalty: "+str(use_GP))

optimizer = "Adam" #Adam, RMS
fl_act_func_CC = 'poly-softplus' # abs, softplus, poly-softplus

class CNN_discriminator(nn.Module):
    @nn.compact
    def __call__(self, x, train):
        x = nn.Conv(features=64, kernel_size=(3,3), strides=(2,2), padding='SAME')(x)
        x = nn.activation.leaky_relu(x, negative_slope=0.2)
        x = nn.Dropout(rate=0.4, deterministic=not train)(x)
        x = nn.Conv(features=64, kernel_size=(3,3), strides=(2,2), padding='SAME')(x)
        x = nn.Dropout(rate=0.4, deterministic=not train)(x)
        x = x.reshape((x.shape[0], -1))
        x = nn.Dense(features=1)(x)
        return x


mnist_trainset = datasets.MNIST(root='./data', train=True, download=True, transform=None)

#data distribution
trainX = mnist_trainset.train_data.numpy()
trainy = mnist_trainset.train_labels.numpy()

X = np.expand_dims(trainX, axis=-1)
# convert from unsigned ints to floats
X = X.astype('float32')
# scale from [0,255] to [0,1]
X = X / 255.0

P_digits=X[trainy==P_digit].astype('f')
Q_digits=X[trainy==Q_digit].astype('f')

P_idx = np.random.randint(0,len(P_digits),size=N)
Q_idx = np.random.randint(0,len(Q_digits),size=N)

data_P = P_digits[P_idx, :]
data_Q = Q_digits[Q_idx, :]

print('P-Data Shape')
print(data_P.shape)
print('Q-Data Shape')
print(data_Q.shape)

#Saved models folder    
saved_name ='MNIST_saved_models'
if not os.path.exists(saved_name):
	os.makedirs(saved_name)

model_file = f'{saved_name}/{mthd}_{P_digit}_{Q_digit}_{N}_{m}_{lr}_{epochs}_{alpha}_{L}_{gp_weight}_{use_GP}_{run_num}.pth'


discriminator = CNN_discriminator()

x = jnp.ones((1, 28, 28, 1))
test = nn.tabulate(discriminator, jax.random.key(0))
print(test(x, train=False))

# Initialize the model's parameters with a dummy input
rng = jax.random.PRNGKey(0)
params = discriminator.init(rng, x, train=False)
print(jax.tree_map(lambda x: x.shape, params)) # Check the parameters
optimizer = "RMS"  # Adam, RMS

#construct optimizers
if optimizer == 'Adam':
    disc_optimizer = optax.adam(lr)

if optimizer == 'RMS':
    disc_optimizer = optax.rmsprop(lr)


# construct gradient penalty
if use_GP:
    discriminator_penalty=Gradient_Penalty_1Sided(gp_weight, L)
else:
    discriminator_penalty=None

# if os.path.exists(model_file):
#     # load the model
#     # discriminator = pickle.load(open(model_file, 'rb'))
#     print()
#     print('Loading Model...')
#     print()
#     discriminator = checkpoints.restore_checkpoint(ckpt_dir='my_checkpoints/',
#                                                    target=params,
#                                                    prefix='my_model')


# construct divergence
if mthd=="IPM":
    divergence_CNN = IPM(discriminator, disc_optimizer, epochs, m, discriminator_penalty, cnn=True)

if mthd=="KLD-LT":
    divergence_CNN = KLD_LT(discriminator, disc_optimizer, epochs, m, discriminator_penalty, cnn=True)
    
if mthd=="KLD-DV":
    divergence_CNN = KLD_DV(discriminator, disc_optimizer, epochs, m, discriminator_penalty, cnn=True)

if mthd=="squared-Hel-LT":
    divergence_CNN = squared_Hellinger_LT(discriminator, disc_optimizer, epochs, m, discriminator_penalty, cnn=True)

if mthd=="chi-squared-LT":
    divergence_CNN = Pearson_chi_squared_LT(discriminator, disc_optimizer, epochs, m, discriminator_penalty, cnn=True)

if mthd=="chi-squared-HCR":
    divergence_CNN = Pearson_chi_squared_HCR(discriminator, disc_optimizer, epochs, m, discriminator_penalty, cnn=True)

if mthd=="JS-LT":
    divergence_CNN = Jensen_Shannon_LT(discriminator, disc_optimizer, epochs, m, discriminator_penalty, cnn=True)    

if mthd=="alpha-LT":
    divergence_CNN = alpha_Divergence_LT(discriminator, disc_optimizer, alpha, epochs, m, discriminator_penalty, cnn=True)

if mthd=="Renyi-DV":
    divergence_CNN = Renyi_Divergence_DV(discriminator, disc_optimizer, alpha, epochs, m, discriminator_penalty, cnn=True)
    
if mthd=="Renyi-CC":
    divergence_CNN = Renyi_Divergence_CC(discriminator, disc_optimizer, alpha, epochs, m, fl_act_func_CC, discriminator_penalty, cnn=True)

if mthd=="rescaled-Renyi-CC":
    divergence_CNN = Renyi_Divergence_CC_rescaled(discriminator, disc_optimizer, alpha, epochs, m, fl_act_func_CC, discriminator_penalty, cnn=True)

if mthd=="Renyi-WCR":
    divergence_CNN = Renyi_Divergence_WCR(discriminator, disc_optimizer, epochs, m, fl_act_func_CC, discriminator_penalty, cnn=True)


if not os.path.exists(model_file):
    #train Discriminator
    print('Training the model...')
    opt_state = disc_optimizer.init(params)
    divergence_estimates, params = divergence_CNN.train(data_P, data_Q, params, opt_state)
    print()
    print("Training Complete")

    # Save the model
    # print('Saving Model...')
    # checkpoints.save_checkpoint(ckpt_dir='my_checkpoints/',
    #                             target=params,
    #                             step=100,
    #                             prefix='my_model',
    #                             overwrite=True)

#Save results    
test_name = f'MNIST_{mthd}_divergence_demo'
if not os.path.exists(test_name):
	os.makedirs(test_name)
	
estimate = divergence_CNN.estimate(data_P, data_Q, params)
print(f'{mthd} estimate between digits {P_digit} and {Q_digit}: {estimate}')


with open(test_name+'/'+mthd+'_div_estimate_P_digit_' +str(P_digit)+'_Q_digit_' +str(Q_digit)+'_N_'+str(N)+'_m_'+str(m)+'_Lrate_{:.1e}'.format(lr)+'_epochs_'+str(epochs)+'_alpha_{:.1f}'.format(alpha)+'_L_{:.1f}'.format(L)+'_gp_weight_{:.1f}'.format(gp_weight)+'_GP_'+str(use_GP)+'_run_num_'+str(run_num)+'.csv', "w") as output:
    writer = csv.writer(output, lineterminator='\n')
    writer.writerow([estimate])


print(f'--- {time.perf_counter() - start} seconds ---')