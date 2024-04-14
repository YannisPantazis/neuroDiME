import numpy as np
import pandas as pd
import csv
import os
import sys
import argparse
#import json
import matplotlib.pyplot as plt
from scipy.stats import norm
from bisect import bisect_left, bisect_right
import torchvision.datasets as datasets
from torch.nn import Sequential, Conv2d, LeakyReLU, Linear, Flatten, Dropout
from torchsummary import summary
import time

current_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from models.torch_model import *
from models.Divergences_torch import *
from models.GAN_torch import *

start = time.perf_counter()

# read input arguments
parser = argparse.ArgumentParser(description='Neural-based Estimation of Divergences between MNIST Digit Distributions')
parser.add_argument('--method', default='KLD-DV', type=str, metavar='method',
                    help='values: IPM, KLD-DV, KLD-LT, squared-Hel-LT, chi-squared-LT, JS-LT, alpha-LT, Renyi-DV, Renyi-CC, rescaled-Renyi-CC, Renyi-CC-WCR')
                        
parser.add_argument('--sample_size', default=10000, type=int, metavar='N')
parser.add_argument('--batch_size', default=100, type=int, metavar='m')
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


mnist_trainset = datasets.MNIST(root='./data', train=True, download=True, transform=None)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

#data distribution
trainX = mnist_trainset.train_data
trainy = mnist_trainset.train_labels

X = np.expand_dims(trainX, axis=1)
# convert from unsigned ints to floats
X = X.astype('float32')
# scale from [0,255] to [0,1]
X = X / 255.0

# construct gradient penalty
if use_GP:
    discriminator_penalty=Gradient_Penalty_1Sided(gp_weight, L)
else:
    discriminator_penalty=None

P_digit_arr = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
Q_digit_arr = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

#Saved models folder    
# saved_name = f'MNIST_{mthd}_saved_models'
# if not os.path.exists(saved_name):
# 	os.makedirs(saved_name)

for P_digit in P_digit_arr:
    for Q_digit in Q_digit_arr:
        # model_file = f'{saved_name}/{mthd}_{P_digit}_{Q_digit}_{N}_{m}_{lr}_{epochs}_{alpha}_{L}_{gp_weight}_{use_GP}_{run_num}.pth'

        # if os.path.exists(model_file):
        #     # load the model
        #     # discriminator = pickle.load(open(model_file, 'rb'))
        #     print()
        #     print('Loading Model...')
        #     print()
        #     discriminator = torch.load(model_file)
        # else:
            # construct the discriminator neural network
        discriminator = Sequential(
                            Conv2d(in_channels=1, out_channels=64, kernel_size=(3,3), stride=(2, 2), padding=1),
                            LeakyReLU(negative_slope=0.2),
                            Dropout(p=0.4),
                            Conv2d(in_channels=64, out_channels=64, kernel_size=(3,3), stride=(2, 2), padding=1),
                            LeakyReLU(negative_slope=0.2),
                            Dropout(p=0.4),
                            Flatten(),
                            Linear(3136, 1)
                            )

        # print()
        # print('Discriminator Summary:')
        # summary(discriminator, (1, 28, 28))
        #construct optimizers
        if optimizer == 'Adam':
            disc_optimizer = torch.optim.Adam(discriminator.parameters(), lr=lr)

        if optimizer == 'RMS':
            disc_optimizer = torch.optim.RMSprop(discriminator.parameters(), lr=lr)

        discriminator.to(device)

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
            divergence_CNN = Renyi_Divergence_WCR(discriminator, disc_optimizer, alpha, epochs, m, fl_act_func_CC, discriminator_penalty)

        #Save results    
        test_name=f'MNIST_{mthd}_divergence_demo_torch'
        if not os.path.exists(test_name):
            os.makedirs(test_name)

        P_digits=X[trainy==P_digit].astype('f')
        Q_digits=X[trainy==Q_digit].astype('f')

        P_idx = np.random.randint(0,len(P_digits),size=N)
        Q_idx = np.random.randint(0,len(Q_digits),size=N)

        data_P = P_digits[P_idx, :]
        data_Q = Q_digits[Q_idx, :]

        # if not os.path.exists(model_file)
        #train Discriminator
        divergence_estimates=divergence_CNN.train(data_P, data_Q, device)
            # save the model
            # print('Saving Model...')
            # torch.save(discriminator, model_file)

        estimate = divergence_CNN.estimate(data_P, data_Q).detach().cpu().numpy()
        print(f'{mthd} estimate between digits {P_digit} and {Q_digit}: {estimate}')


        with open(test_name+'/'+mthd+'_div_estimate_P_digit_' +str(P_digit)+'_Q_digit_' +str(Q_digit)+'_N_'+str(N)+'_m_'+str(m)+'_Lrate_{:.1e}'.format(lr)+'_epochs_'+str(epochs)+'_alpha_{:.1f}'.format(alpha)+'_L_{:.1f}'.format(L)+'_gp_weight_{:.1f}'.format(gp_weight)+'_GP_'+str(use_GP)+'_run_num_'+str(run_num)+'.csv', "w") as output:
            writer = csv.writer(output, lineterminator='\n')
            writer.writerow([estimate])


print(f'--- {time.perf_counter() - start} seconds ---')