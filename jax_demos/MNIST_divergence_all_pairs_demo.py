import jax.numpy as jnp
import numpy as np
import os
import sys
import csv
import argparse
from optax import rmsprop, adam
import time
import torchvision.datasets as datasets
import flax.linen as nn
from flax.training import checkpoints, train_state
import glob 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Add parent directory to sys.path
current_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from models.model_jax import *
from models.Divergences_jax import *

start = time.perf_counter()

class GradientPenalty(Discriminator_Penalty):
    def __init__(self, weight, L):
        super().__init__(weight)
        self.L = L
        
    def get_Lip_const(self):
        return self.L

    def set_Lip_const(self, L):
        self.L = L
            
    def evaluate(self, c, images, samples, params, labels=None):
        rng = jax.random.PRNGKey(0)
        assert images.shape == samples.shape
            
        batch_size = images.shape[0]
        shape = images.shape[1:]  # Assuming images have shape (batch_size, height, width, channels)
        
        # Generate random interpolation factor
        rng, subkey = jax.random.split(rng)
        jump = jax.random.uniform(subkey, shape=(batch_size, 1, 1, 1), dtype=images.dtype)
        jump_ = jnp.broadcast_to(jump, images.shape)
        interpolated = images * jump_ + (1 - jump_) * samples
            
        # Define a function for computing the discriminator output
        def discriminator_fn(interpolated):
            if labels is not None:
                return c.apply({'params': params}, interpolated, labels)
            else:
                return c.apply({'params': params}, interpolated)
        
        # Compute gradients with respect to the interpolated images
        gradients = grad(lambda interpolated: jnp.sum(discriminator_fn(interpolated)))(interpolated)
        
        # Compute the gradient penalty
        gradients_norm = jnp.linalg.norm(gradients, axis=-1)  # L2 norm over the spatial dimensions
        penalty = self.get_penalty_weight() * jnp.mean((self.L - gradients_norm) ** 2)
        
        return penalty

def main(mthd):
    # read input arguments
    parser = argparse.ArgumentParser(description='Neural-based Estimation of Divergences between MNIST Digit Distributions')
    parser.add_argument('--method', default='KLD-DV', type=str, metavar='method',
                        help='values: IPM, KLD-DV, KLD-LT, squared-Hel-LT, chi-squared-LT, JS-LT, alpha-LT, Renyi-DV, Renyi-CC, rescaled-Renyi-CC, Renyi-CC-WCR')
                            
    parser.add_argument('--sample_size', default=10000, type=int, metavar='N')
    parser.add_argument('--batch_size', default=1024, type=int, metavar='m')
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

    # mthd = opt_dict['method']
    
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

    mnist_trainset = datasets.MNIST(root='./data', train=True, download=True, transform=None)

    #data distribution
    trainX = mnist_trainset.train_data.numpy()
    trainy = mnist_trainset.train_labels.numpy()

    X = np.expand_dims(trainX, axis=-1)
    # convert from unsigned ints to floats
    X = X.astype('float32')
    # scale from [0,255] to [0,1]
    X = X / 255.0

    if use_GP:
        discriminator_penalty=GradientPenalty(gp_weight, L)
        test_name=f'MNIST_{mthd}_GP_divergence_demo_jax'
        title = f'Colormap of {mthd} estimates between all pair of digits using JaX with Gradient Penalty'
        save_path = f"colormap_jax_{mthd}_GP.png"
    else:
        discriminator_penalty=None
        discriminator_penalty=None
        test_name=f'MNIST_{mthd}_divergence_demo_jax'
        title = f'Colormap of {mthd} estimates between all pair of digits using JaX'
        save_path = f"colormap_jax_{mthd}.png"


    P_digit_arr = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    Q_digit_arr = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

    for P_digit in P_digit_arr:
        for Q_digit in Q_digit_arr:
            discriminator = DiscriminatorMNIST()
            x = jnp.ones((1, 28, 28, 1))
            rng = jax.random.PRNGKey(0)

            params = discriminator.init(rng, x)['params']

            #construct optimizers
            if optimizer == 'Adam':
                disc_optimizer = adam(lr)

            if optimizer == 'RMS':
                disc_optimizer = rmsprop(lr)
                
            training_state  = train_state.TrainState.create(apply_fn=discriminator.apply, params=params, tx=disc_optimizer)

            # construct divergence
            if mthd=="IPM":
                divergence_CNN = IPM(discriminator, disc_optimizer, epochs, m, discriminator_penalty)

            if mthd=="KLD-LT":
                divergence_CNN = KLD_LT(discriminator, disc_optimizer, epochs, m, discriminator_penalty)
                
            if mthd=="KLD-DV":
                divergence_CNN = KLD_DV(discriminator, disc_optimizer, epochs, m, discriminator_penalty)

            if mthd=="squared-Hel-LT":
                divergence_CNN = squared_Hellinger_LT(discriminator, disc_optimizer, epochs, m, discriminator_penalty)

            if mthd=="chi-squared-LT":
                divergence_CNN = Pearson_chi_squared_LT(discriminator, disc_optimizer, epochs, m, discriminator_penalty)

            if mthd=="chi-squared-HCR":
                divergence_CNN = Pearson_chi_squared_HCR(discriminator, disc_optimizer, epochs, m, discriminator_penalty)

            if mthd=="JS-LT":
                divergence_CNN = Jensen_Shannon_LT(discriminator, disc_optimizer, epochs, m, discriminator_penalty)    

            if mthd=="alpha-LT":
                divergence_CNN = alpha_Divergence_LT(discriminator, disc_optimizer, alpha, epochs, m, discriminator_penalty)

            if mthd=="Renyi-DV":
                divergence_CNN = Renyi_Divergence_DV(discriminator, disc_optimizer, alpha, epochs, m, discriminator_penalty)
                
            if mthd=="Renyi-CC":
                divergence_CNN = Renyi_Divergence_CC(discriminator, disc_optimizer, alpha, epochs, m, fl_act_func_CC, discriminator_penalty)

            if mthd=="rescaled-Renyi-CC":
                divergence_CNN = Renyi_Divergence_CC_rescaled(discriminator, disc_optimizer, alpha, epochs, m, fl_act_func_CC, discriminator_penalty)

            if mthd=="Renyi-CC-WCR":
                divergence_CNN = Renyi_Divergence_WCR(discriminator, disc_optimizer, math.inf, epochs, m, fl_act_func_CC, discriminator_penalty)

            P_digits=X[trainy==P_digit].astype('f')
            Q_digits=X[trainy==Q_digit].astype('f')

            P_idx = np.random.randint(0,len(P_digits),size=N)
            Q_idx = np.random.randint(0,len(Q_digits),size=N)

            data_P = P_digits[P_idx, :]
            data_Q = Q_digits[Q_idx, :]

            print('Training digits', P_digit, 'and', Q_digit)
            divergence_estimates, losses, state = divergence_CNN.train(data_P, data_Q, training_state)

            #Save results    
            test_name = f'MNIST_{mthd}_divergence_demo_jax'
            if not os.path.exists(test_name):
                os.makedirs(test_name)
                
            estimate = divergence_CNN.estimate(data_P, data_Q, state.params)
            print(f'{mthd} estimate between digits {P_digit} and {Q_digit}: {estimate}')

            with open(test_name+'/'+mthd+'_div_estimate_P_digit_' +str(P_digit)+'_Q_digit_' +str(Q_digit)+'_N_'+str(N)+'_m_'+str(m)+'_Lrate_{:.1e}'.format(lr)+'_epochs_'+str(epochs)+'_alpha_{:.1f}'.format(alpha)+'_L_{:.1f}'.format(L)+'_gp_weight_{:.1f}'.format(gp_weight)+'_GP_'+str(use_GP)+'_run_num_'+str(run_num)+'.csv', "w") as output:
                writer = csv.writer(output, lineterminator='\n')
                writer.writerow([estimate])

            print(f'--- {time.perf_counter() - start} seconds ---')
        
    # Reading and storing all the estimates from all the csv files
    csv_files = sorted(glob.glob(test_name + '/' + "*.csv"))
    numbers = []
        
    for file in csv_files:
        divergence = pd.read_csv(file, header=None).iloc[0,0]
        if math.isnan(divergence):
            numbers.append(math.inf)
        else:
            numbers.append(np.log10((np.abs(divergence) + 1e-10)))

    numbers = np.array(numbers)
    numbers = numbers.reshape(10,10)

    plt.figure(figsize=(10, 10))
    sns.heatmap(numbers, annot=True, fmt='.2f', linewidth=2, cmap='coolwarm')
    plt.title(title)
    plt.savefig(save_path)
    plt.show()
    plt.close()
    
if __name__ == '__main__':
    for mthd in ['Renyi-CC-WCR']:
        print(f'Running {mthd}...')
        main(mthd)