import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_addons as tfa
from keras import backend as K
from keras.layers import Dense, Conv2D, Flatten, Dropout, LeakyReLU
import csv
import os
import argparse
#import json
import matplotlib.pyplot as plt
from scipy.stats import norm
from bisect import bisect_left, bisect_right
from Divergences import *
from GAN import *
from keras.datasets.mnist import load_data

# read input arguments
parser = argparse.ArgumentParser(description='Neural-based Estimation of Divergences between MNIST Digit Distributions')
parser.add_argument('--P_digit', type=int)          
parser.add_argument('--Q_digit', type=int)
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


#data distribution
(trainX, trainy), (_, _) = load_data()

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



# construct the discriminator neural network
discriminator = tf.keras.Sequential()
discriminator.add(Conv2D(64, (3,3), strides=(2, 2), padding='same', input_shape=(28,28,1)))
discriminator.add(LeakyReLU(alpha=0.2))
discriminator.add(Dropout(0.4))
discriminator.add(Conv2D(64, (3,3), strides=(2, 2), padding='same'))
discriminator.add(LeakyReLU(alpha=0.2))
discriminator.add(Dropout(0.4))
discriminator.add(Flatten())
discriminator.add(Dense(1, activation='sigmoid'))

print()
print("Discriminator Summary:")
discriminator.summary()

#construct optimizer
if optimizer == 'Adam':
    disc_optimizer = tf.keras.optimizers.Adam(lr)

if optimizer == 'RMS':
    disc_optimizer = tf.keras.optimizers.RMSprop(lr)


# construct gradient penalty
if use_GP:
    discriminator_penalty=Gradient_Penalty_1Sided(gp_weight, L)
else:
    discriminator_penalty=None


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

if mthd=="Renyi-WCR":
    divergence_CNN = Renyi_Divergence_WCR(discriminator, disc_optimizer, epochs, m, fl_act_func_IC, discriminator_penalty)


#train Discriminator
divergence_estimates=divergence_CNN.train(data_P, data_Q)

#Save results    
test_name='MNIST_divergence_demo'
if not os.path.exists(test_name):
	os.makedirs(test_name)
	
	    
with open(test_name+'/'+mthd+'_div_estimate_P_digit_' +str(P_digit)+'_Q_digit_' +str(Q_digit)+'_N_'+str(N)+'_m_'+str(m)+'_Lrate_{:.1e}'.format(lr)+'_epochs_'+str(epochs)+'_alpha_{:.1f}'.format(alpha)+'_L_{:.1f}'.format(L)+'_gp_weight_{:.1f}'.format(gp_weight)+'_GP_'+str(use_GP)+'_run_num_'+str(run_num)+'.csv', "w") as output:
    writer = csv.writer(output, lineterminator='\n')
    for div_est in divergence_estimates:
        writer.writerow([div_est]) 


print()
print("Training Complete")


