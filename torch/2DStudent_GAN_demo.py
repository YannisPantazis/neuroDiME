import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import csv
import os
import argparse
#import json
import matplotlib.pyplot as plt
from scipy.stats import norm
from bisect import bisect_left, bisect_right
from Divergences_torch import *
from torch_model import *
from GAN_torch import *


# read input arguments
parser = argparse.ArgumentParser(description='Neural-based Estimation of Divergences between Gaussians')
parser.add_argument('--method', default='KLD-DV', type=str, metavar='method',
                    help='values: IPM, KLD-DV, KLD-LT, squared-Hel-LT, chi-squared-LT, JS-LT, alpha-LT, Renyi-DV, Renyi-CC, rescaled-Renyi-CC, Renyi-CC-WCR')
parser.add_argument('--disc_steps_per_gen_step', default=5, type=int)
parser.add_argument('--sample_size', default=10000, type=int, metavar='N')
parser.add_argument('--batch_size', default=1000, type=int, metavar='m')
parser.add_argument('--lr', default=0.001, type=float)
parser.add_argument('--epochs', default=100, type=int,
                    help='number of total epochs to run')
parser.add_argument('--alpha', default=2.0, type=float, metavar='alpha')
parser.add_argument('--Lip_constant', default=1.0, type=float, metavar='Lipschitz constant')
parser.add_argument('--gp_weight', default=1.0, type=float, metavar='GP weight')
parser.add_argument('--spectral_norm', choices=('True','False'), default='False')
parser.add_argument('--bounded', choices=('True','False'), default='False')
parser.add_argument('--reverse_order', choices=('True','False'), default='False')
parser.add_argument('--use_GP', choices=('True','False'), default='False')



parser.add_argument('--run_number', default=1, type=int, metavar='run_num')

opt = parser.parse_args()
opt_dict = vars(opt)
print('parsed options:', opt_dict)

mthd = opt_dict['method']
disc_steps_per_gen_step = opt_dict['disc_steps_per_gen_step']
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
run_num=opt_dict['run_number']

spec_norm = opt_dict['spectral_norm']=='True'
bounded=opt_dict['bounded']=='True'
reverse_order = opt_dict['reverse_order']=='True'
use_GP = opt_dict['use_GP']=='True'

print("Spectral_norm: "+str(spec_norm))
print("Bounded: "+str(bounded))
print("Reversed: "+str(reverse_order))
print("Use Gradient Penalty: "+str(use_GP))


fl_act_func_CC = 'poly-softplus' # abs, softplus, poly-softplus
optimizer = "RMS" #Adam, RMS
save_frequency = 100 #generator samples are saved every save_frequency epochs
num_gen_samples_to_save = 5000


#data distribution
d=2
nu=0.5 #degrees of freedom for  d-dim student (use Sigma=I)
#centers
D=10
n_centers=4
Delta_array=np.zeros([n_centers,d])
Delta_array[0,:]=[D,D]
Delta_array[1,:]=[D,-D]
Delta_array[2,:]=[-D,D]
Delta_array[3,:]=[-D,-D]


#embed data in higher dimensional manifold
di=5
df=5
X_dim = d+di+df #dimension of the real data
offset=1.0





def embed_data(x):
    z=np.concatenate((offset*np.ones([x.shape[0],di]),x),axis=1)
    z=np.concatenate((z,offset*np.ones([x.shape[0],df])),axis=1)
    
    return z

#data distribution
def sample_P(N_samp):
    u=np.random.chisquare(nu,size=[N_samp,1])
    x=np.divide(np.random.normal(0.0,1.0,size=[N_samp,d]),np.sqrt(u/nu))
    
    idx=np.random.randint(0,4,size=[N_samp])
    x=x+Delta_array[idx,:]
    
    return embed_data(x)




data_P = sample_P(N)
data_P = data_P.astype('f')



# construct the discriminator neural network
act_func = 'relu'

D_hidden_layers=[64,32,16] #sizes of hidden layers for the discriminator

discriminator = Discriminator(input_dim=X_dim, batch_size=m, spec_norm=spec_norm, bounded=bounded, layers_list=D_hidden_layers)


#construct the generator neural network
G_hidden_layers=[64,32,16] #sizes of hidden layers for the generator
Z_dim=10 #dimension of the noise source for the generator

generator = Generator(X_dim=X_dim, Z_dim=Z_dim, batch_size=m, spec_norm=spec_norm, layers_list=G_hidden_layers)


#Function for sampling from the noise source

def noise_source(N_samp):
    return np.random.normal(0., 1.0, size=[N_samp, Z_dim])

#construct optimizers
if optimizer == 'Adam':
    disc_optimizer = torch.optim.Adam(discriminator.parameters(), lr=lr)
    gen_optimizer = torch.optim.Adam(generator.parameters(), lr=lr)

if optimizer == 'RMS':
    disc_optimizer = torch.optim.RMSprop(discriminator.parameters(), lr=lr)
    gen_optimizer = torch.optim.RMSprop(generator.parameters(), lr=lr)


# construct gradient penalty
if use_GP:
    discriminator_penalty=Gradient_Penalty_1Sided(gp_weight, L)
else:
    discriminator_penalty=None


# construct divergence
if mthd=="IPM":
    div_dense = IPM(discriminator, disc_optimizer, epochs, m, discriminator_penalty)

if mthd=="KLD-LT":
    div_dense = KLD_LT(discriminator, disc_optimizer, epochs, m, discriminator_penalty)
    
if mthd=="KLD-DV":
    div_dense = KLD_DV(discriminator, disc_optimizer, epochs, m, discriminator_penalty)

if mthd=="squared-Hel-LT":
    div_dense = squared_Hellinger_LT(discriminator, disc_optimizer, epochs, m, discriminator_penalty)

if mthd=="chi-squared-LT":
    div_dense = Pearson_chi_squared_LT(discriminator, disc_optimizer, epochs, m, discriminator_penalty)

if mthd=="chi-squared-HCR":
    div_dense = Pearson_chi_squared_HCR(discriminator, disc_optimizer, epochs, m, discriminator_penalty)

if mthd=="JS-LT":
    div_dense = Jensen_Shannon_LT(discriminator, disc_optimizer, epochs, m, discriminator_penalty)    

if mthd=="alpha-LT":
    div_dense = alpha_Divergence_LT(discriminator, disc_optimizer, alpha, epochs, m, discriminator_penalty)

if mthd=="Renyi-DV":
    div_dense = Renyi_Divergence_DV(discriminator, disc_optimizer, alpha, epochs, m, discriminator_penalty)
    
if mthd=="Renyi-CC":
    div_dense = Renyi_Divergence_CC(discriminator, disc_optimizer, alpha, epochs, m, fl_act_func_CC, discriminator_penalty)

if mthd=="rescaled-Renyi-CC":
    div_dense = Renyi_Divergence_CC_rescaled(discriminator, disc_optimizer, alpha, epochs, m, fl_act_func_CC, discriminator_penalty)

if mthd=="Renyi-WCR":
    div_dense = Renyi_Divergence_WCR(discriminator, disc_optimizer, epochs, m, fl_act_func_CC, discriminator_penalty)


#train GAN
GAN_dense = GAN(div_dense, generator, gen_optimizer, noise_source, epochs, disc_steps_per_gen_step, m, reverse_order)
generator_samples, loss_array = GAN_dense.train(data_P, save_frequency, num_gen_samples_to_save, save_loss_estimates=True)



#Save results    
test_name='2Dstudent_submanifold_GAN_demo'
if not os.path.exists(test_name):
	os.makedirs(test_name)
	
	    
with open(test_name+'/loss_'+mthd+'_N_'+str(N)+'_m_'+str(m)+'_Lrate_{:.1e}'.format(lr)+'_epochs_'+str(epochs)+'_alpha_{:.1f}'.format(alpha)+'_L_{:.1f}'.format(L)+'_gp_weight_{:.1f}'.format(gp_weight)+'_spec_norm_'+str(spec_norm)+'_bounded_'+str(bounded)+'_reverse_'+str(reverse_order)+'_GP_'+str(use_GP)+'_optimizer_'+optimizer+'_run_num_'+str(run_num)+'.csv', "w") as output:
    writer = csv.writer(output, lineterminator='\n')
    for loss in loss_array:
        writer.writerow([loss]) 

for j in range(len(generator_samples)):    
    with open(test_name+'/generator_samples_'+mthd+'_N_'+str(N)+'_m_'+str(m)+'_Lrate_{:.1e}'.format(lr)+'_epoch_'+str((j+1)*save_frequency)+'_alpha_{:.1f}'.format(alpha)+'_L_{:.1f}'.format(L)+'_gp_weight_{:.1f}'.format(gp_weight)+'_spec_norm_'+str(spec_norm)+'_bounded_'+str(bounded)+'_reverse_'+str(reverse_order)+'_GP_'+str(use_GP)+'_optimizer_'+optimizer+'_run_num_'+str(run_num)+'.csv', "w") as output:
        writer = csv.writer(output, lineterminator='\n')
        for sample in generator_samples[j]:
            writer.writerow(sample) 

print()
print("Training Complete")


