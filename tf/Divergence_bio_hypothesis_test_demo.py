import numpy as np
import argparse
import csv
import os
import json
#from aux_funcs import *
from numpy import genfromtxt
from tf_model import *


# read input arguments
parser = argparse.ArgumentParser(description='AUC for Sick Cell Detection using Neural-based Variational Divergences ')
parser.add_argument('--prob_sick', type=float, metavar='p')
parser.add_argument('--method', default='all', type=str, metavar='mthd',
                   help='values: IPM, Wasserstein, KLD-LT, KLD-DV, KLD-DV-GP, squared-Hel-LT, chi-squared-HCR, chi-squared-LT, JS-LT, alpha-LT, Renyi-DV, Renyi-CC, rescaled-Renyi-CC, Renyi-WCR')
parser.add_argument('--sample_size', default=10000, type=int, metavar='N')
parser.add_argument('--batch_size', default=1000, type=int, metavar='m')
parser.add_argument('--lr', default=0.001, type=float)
parser.add_argument('--epochs', default=100, type=int,
                    help='number of total epochs to run')
parser.add_argument('--alpha', default=2.0, type=float, metavar='alpha')
parser.add_argument('--run_number', default=1, type=int, metavar='run_num')
parser.add_argument('--Lip_constant', default=1.0, type=float, metavar='Lipschitz constant')
parser.add_argument('--gp_weight', default=10.0, type=float, metavar='GP weight')

parser.add_argument('--spectral_norm', choices=('True','False'), default='False')
parser.add_argument('--bounded', choices=('True','False'), default='False')
parser.add_argument('--reverse', choices=('True','False'), default='False')
parser.add_argument('--use_GP', choices=('True','False'), default='False')

opt = parser.parse_args()
opt_dict = vars(opt)
print('parsed options:', opt_dict)

p=opt_dict['prob_sick']
mthd = opt_dict['method']
N = opt_dict['sample_size']
m = opt_dict['batch_size']
lr = opt_dict['lr']
epochs = opt_dict['epochs']
run_num = opt_dict['run_number']
L = opt_dict['Lip_constant']
gp_weight = opt_dict['gp_weight']

spec_norm = opt_dict['spectral_norm']=='True'
bounded=opt_dict['bounded']=='True'
reverse_order = opt_dict['reverse']=='True'
use_GP = opt_dict['use_GP']=='True'

print("Spectral_norm: "+str(spec_norm))
print("Bounded: "+str(bounded))
print("Reversed: "+str(reverse_order))
print("Use Gradient Penalty: "+str(use_GP))

fl_act_func_IC = 'poly-softplus' # abs, softplus, poly-softplus

if mthd=="squared-Hel-LT":
    alpha=1./2.
elif mthd=="chi-squared-LT":
    alpha=2.
elif mthd=="chi_squared_HCR":
    alpha=2.
else:
    alpha = opt_dict['alpha']

layers_list = [32, 32]
act_func = 'relu'

# load data
data_h = np.genfromtxt("../bio_csv/healthy.csv", delimiter=",").astype('float32')
data_s = np.genfromtxt("../bio_csv/sick.csv", delimiter=",").astype('float32')

d = data_h.shape[1]
layers_list.insert(0, d)


print(f'Predicting the {mthd} divergence using TensorFlow\n')
discriminator = Discriminator(input_dim=d, spec_norm=spec_norm, bounded=bounded, layers_list=layers_list)

# number of sample per class
N2 = int(np.ceil(N*p))
N1 = N - N2
print(N1,N2,p)

# select which samples via indexing
idx_pure = np.random.randint(data_h.shape[0], size=N)

idx_mix1 = np.random.randint(data_h.shape[0], size=N1)
idx_mix2 = np.random.randint(data_s.shape[0], size=N2)

# create datasets
dataset_pure = data_h[idx_pure]
dataset_cntmd = np.concatenate((data_h[idx_mix1], data_s[idx_mix2]), axis=0)

# shuffle contaminated dataset
idx = np.random.randint(N, size=N)
dataset_cntmd = dataset_cntmd[idx]


if reverse_order:
    data_P = dataset_pure
    data_Q = dataset_cntmd
else:
    data_P = dataset_cntmd
    data_Q = dataset_pure


print('Data shapes:')
print(data_P.shape)
print(data_Q.shape)

optimizer = "RMS" #Adam, RMS

#construct optimizers
if optimizer == 'Adam':
    disc_optimizer = tf.keras.optimizers.Adam(lr)
    gen_optimizer = tf.keras.optimizers.Adam(lr)

if optimizer == 'RMS':
    disc_optimizer = tf.keras.optimizers.RMSprop(lr)
    gen_optimizer = tf.keras.optimizers.RMSprop(lr)

# construct gradient penalty
if use_GP:
    discriminator_penalty=Gradient_Penalty_1Sided(gp_weight, L)
else:
    discriminator_penalty=None


#Construct divergence
       
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
    div_dense = Renyi_Divergence_CC(discriminator, disc_optimizer, alpha, epochs, m, fl_act_func_IC, discriminator_penalty)
       	       
if mthd=="rescaled-Renyi-CC":
    div_dense = Renyi_Divergence_CC_rescaled(discriminator, disc_optimizer, alpha, epochs, m, fl_act_func_IC, discriminator_penalty)
       	       
if mthd=="Renyi-WCR":
    div_dense = Renyi_Divergence_WCR(discriminator, disc_optimizer, 'Inf', epochs, m, fl_act_func_IC, discriminator_penalty)
       	              
    	    
# Estimate divergence    
divergence_estimates = div_dense.train(data_P, data_Q)

print('prob sick: '+str(p))      
print(mthd+':\t\t {:.4}'.format(divergence_estimates[-1]))
print()
        
#save result       
test_name="Bio_hypothesis_test_tf"

if not os.path.exists(test_name):
	os.makedirs(test_name)
	
	    
with open(test_name+'/estimated_'+mthd+'_p_{:.1e}'.format(p)+'_N_'+str(N)+'_m_'+str(m)+'_Lrate_{:.1e}'.format(lr)+'_epochs_'+str(epochs)+'_alpha_{:.1f}'.format(alpha)+'_L_{:.1f}'.format(L)+'_gp_weight_{:.1f}'.format(gp_weight)+'_spec_norm_'+str(spec_norm)+'_bounded_'+str(bounded)+'_reverse_'+str(reverse_order)+'_run_num_'+str(run_num)+'.csv', "w") as output:
    writer = csv.writer(output, lineterminator='\n')
    for divergence_estimate in divergence_estimates:
        writer.writerow([divergence_estimate]) 
    




     
        
