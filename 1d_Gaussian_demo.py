import numpy as np
import pandas as pd

import argparse
#import json
import matplotlib.pyplot as plt
from scipy.stats import norm
from bisect import bisect_left, bisect_right

# read input arguments
parser = argparse.ArgumentParser(description='Neural-based Estimation of Divergences')

parser.add_argument('--method', default='KLD-DV', type=str, metavar='method',
                    help='values: KLD-DV, KLD-DV-GP, KLD-LT, squared-Hel-LT, chi-squared-LT, JS-LT, alpha-LT, Renyi-DV, Renyi-CC, rescaled-Renyi-CC, Renyi-CC-WCR, IPM')
parser.add_argument('--sample_size', default=10000, type=int, metavar='N')
parser.add_argument('--batch_size', default=1000, type=int, metavar='m')
parser.add_argument('--lr', default=0.001, type=float)
parser.add_argument('--epochs', default=100, type=int,
                    help='number of total epochs to run')
parser.add_argument('--alpha', default=2.0, type=float, metavar='alpha')
parser.add_argument('--Lip_constant', default=1.0, type=float, metavar='Lipschitz constant')
parser.add_argument('--gp_weight', default=1.0, type=float, metavar='GP weight')

parser.add_argument('--spectral_norm', action='store_true')
parser.add_argument('--no-spectral_norm', dest='spectral_norm', action='store_false')
parser.set_defaults(spectral_norm=False)

parser.add_argument('--bounded', action='store_true')
parser.add_argument('--no-bounded', dest='bounded', action='store_false')
parser.set_defaults(bounded=False)

parser.add_argument('--framework', choices=['tf', 'torch', 'jax'], required=True)

opt = parser.parse_args()
opt_dict = vars(opt)
print('parsed options:', opt_dict)

mthd = opt_dict['method']
N = opt_dict['sample_size']
m = opt_dict['batch_size']
lr = opt_dict['lr']
epochs = opt_dict['epochs']
alpha = opt_dict['alpha']
L = opt_dict['Lip_constant']
gp_weight = opt_dict['gp_weight']
spec_norm = opt_dict['spectral_norm']
bounded = opt_dict['bounded']
fl_act_func_CC = 'abs' # abs, softplus, poly-softplus

fw = opt_dict['framework']

# create data sets
d = 1
mu_P, mu_Q = 0.0, 1.0
sgm_P, sgm_Q = 1.0, 1.0

data_P = np.random.normal(loc=mu_P, scale=sgm_P, size=(N,d))
data_P = data_P.astype('f')
data_Q = np.random.normal(loc=mu_Q, scale=sgm_Q, size=(N,d))
data_Q = data_Q.astype('f')

NoP = 10000
min_x = min(mu_P-4.0*sgm_P, mu_Q-4.0*sgm_Q)
max_x = max(mu_P+4.0*sgm_P, mu_Q+4.0*sgm_Q)
dx = (max_x-min_x) / NoP
x = np.linspace(min_x, max_x, NoP)

pdf_P = norm.pdf(x, mu_P, sgm_P)
pdf_Q = norm.pdf(x, mu_Q, sgm_Q)
eff_vals_P = norm.ppf([0.01, 0.99], loc=mu_P, scale=sgm_P)

# construct the discriminator neural network
layers_list = [32, 32] # UNCECOMP's NN: [16, 16, 8]
act_func = 'relu'

if fw == 'tf':
    print(f'Predicting the {mthd} divergence using TensorFlow\n')
    from models.tensorflow import *
    discriminator = Discriminator(input_dim=d, spec_norm=spec_norm, bounded=bounded, layers_list=layers_list)

elif fw == 'torch':
    print(f'Predicting the {mthd} divergence using PyTorch\n')
    from models.torch import *
    discriminator = Discriminator(input_dim=d, batch_size=m, spec_norm=spec_norm, bounded=bounded, layers_list=layers_list)

else:
    print(f'Predicting the {mthd} divergence using JAX\n')



# construct divergence, train optimizer and estimate the divergence
if mthd=="IPM":
    div_dense = IPM(discriminator, epochs, lr, m)
    div_dense.train(data_P, data_Q)
    div_value_est = float(div_dense.estimate(data_P, data_Q))

    if fw == 'torch':
        x = x.reshape(-1, 1)

    g_est = div_dense.discriminate(x)
    
    g_star = pdf_P - pdf_Q # witness function in MMD (RKHS)
    
    print('IPM:\t\t {:.4}'.format(div_value_est))
    print()

if mthd=="IPM-GP":
    div_dense = Wasserstein_GP(discriminator, epochs, lr, m, L, gp_weight)
    div_dense.train(data_P, data_Q)
    div_value_est = float(div_dense.estimate(data_P, data_Q))

    if fw == 'torch':
        x = x.reshape(-1, 1)

    g_est = div_dense.discriminate(x)
    
    g_star = pdf_P - pdf_Q # witness function in MMD (RKHS)
    
    print('Wasserstein distance:\t\t {:.4}'.format(div_value_est))
    print()

if mthd=="KLD-DV":
    div_dense = KLD_DV(discriminator, epochs, lr, m)
    div_dense.train(data_P, data_Q)
    div_value_est = float(div_dense.estimate(data_P, data_Q))

    if fw == 'torch':
        x = x.reshape(-1, 1)

    g_est = div_dense.discriminate(x)
    
    div_value_true = np.sum(pdf_P*np.log(pdf_P/pdf_Q)) * dx
    g_star = np.log(pdf_P/pdf_Q)
    
    print('KLD (true):\t\t {:.4}'.format(div_value_true))
    print('KLD-DV (estimated):\t {:.4}'.format(div_value_est))
    print()

if mthd=="KLD-DV-GP": # create gp and add it to constructor
    div_dense = KLD_DV_GP(discriminator, epochs, lr, m, L, gp_weight)
    div_dense.train(data_P, data_Q)
    div_value_est = float(div_dense.estimate(data_P, data_Q))

    if fw == 'torch':
        x = x.reshape(-1, 1)

    g_est = div_dense.discriminate(x)
    
    div_value_true = np.sum(pdf_P*np.log(pdf_P/pdf_Q)) * dx
    g_star = np.log(pdf_P/pdf_Q)
    
    print('KLD (true):\t\t {:.4}'.format(div_value_true))
    print('KLD-DV-GP (estimated):\t {:.4}'.format(div_value_est))
    print()

if mthd=="KLD-LT":
    div_dense = KLD_LT(discriminator, epochs, lr, m)
    div_dense.train(data_P, data_Q)
    div_value_est = float(div_dense.estimate(data_P, data_Q))

    if fw == 'torch':
        x = x.reshape(-1, 1)

    g_est = div_dense.discriminate(x)
    
    div_value_true = np.sum(pdf_P*np.log(pdf_P/pdf_Q)) * dx
    g_star = np.log(pdf_P/pdf_Q)
    
    print('KLD (true):\t\t {:.4}'.format(div_value_true))
    print('KLD-LT (estimated):\t {:.4}'.format(div_value_est))
    print()

if mthd=="squared-Hel-LT":
    div_dense = squared_Hellinger_LT(discriminator, epochs, lr, m)
    div_dense.train(data_P, data_Q)
    div_value_est = float(div_dense.estimate(data_P, data_Q))

    if fw == 'torch':
        x = x.reshape(-1, 1)

    g_est = div_dense.discriminate(x)
    
    div_value_true = np.sum((np.sqrt(pdf_P)-(np.sqrt(pdf_Q)))**2.0) * dx
    g_star = (np.sqrt(pdf_P/pdf_Q)-1.0) * np.sqrt(pdf_Q/pdf_P)
    
    print('squared-Hellinger (true):\t\t {:.4}'.format(div_value_true))
    print('squared-Hellinger-LT (estimated):\t\t {:.4}'.format(div_value_est))
    print()

if mthd=="chi-squared-LT":
    div_dense = Pearson_chi_squared_LT(discriminator, epochs, lr, m)
    div_dense.train(data_P, data_Q)
    div_value_est = float(div_dense.estimate(data_P, data_Q))

    if fw == 'torch':
        x = x.reshape(-1, 1)

    g_est = div_dense.discriminate(x)
        
    div_value_true = np.sum((pdf_P-pdf_Q)**2.0 / pdf_Q) * dx
    g_star = 2.0*(pdf_P/pdf_Q-1.0)
    
    print('chi-squared (true):\t\t {:.4}'.format(div_value_true))
    print('chi-squared-LT (estimated):\t\t {:.4}'.format(div_value_est))
    print()

if mthd=="JS-LT":
    div_dense = Jensen_Shannon_LT(discriminator, epochs, lr, m)
    div_dense.train(data_P, data_Q)
    div_value_est = float(div_dense.estimate(data_P, data_Q))

    if fw == 'torch':
        x = x.reshape(-1, 1)

    g_est = div_dense.discriminate(x)
        
    div_value_true = np.sum(pdf_P*np.log(2.0*pdf_P/(pdf_P+pdf_Q))) * dx
    div_value_true = div_value_true + np.sum(pdf_Q*np.log(2.0*pdf_Q/(pdf_P+pdf_Q))) * dx
    g_star = np.log(2.0*pdf_P/(pdf_P+pdf_Q))
    
    print('Jensen-Shannon (true):\t\t {:.4}'.format(div_value_true))
    print('Jensen-Shannon-LT (estimated):\t\t {:.4}'.format(div_value_est))
    print()

if mthd=="alpha-LT":
    div_dense = alpha_Divergence_LT(discriminator, alpha, epochs, lr, m)
    div_dense.train(data_P, data_Q)
    div_value_est = float(div_dense.estimate(data_P, data_Q))

    if fw == 'torch':
        x = x.reshape(-1, 1)

    g_est = div_dense.discriminate(x)

    div_value_true = np.sum(pdf_Q*(np.power(pdf_P/pdf_Q, alpha)-1.0)) * dx/(alpha*(alpha-1.0))
    g_star = np.power(pdf_P/pdf_Q, alpha-1.0) / (alpha-1.0)
    
    print('alpha-divergence (true):\t\t {:.4}'.format(div_value_true))
    print('alpha-divergence-LT (estimated):\t\t {:.4}'.format(div_value_est))
    print()

if mthd=="Renyi-DV":
    div_dense = Renyi_Divergence_DV(discriminator, alpha, epochs, lr, m)
    div_dense.train(data_P, data_Q)
    div_value_est = float(div_dense.estimate(data_P, data_Q))
    g_est = div_dense.discriminate(x)
    
    div_value_true = np.log(np.sum(np.power(pdf_P, alpha)*np.power(pdf_Q, 1.0-alpha)) * dx) / (alpha*(alpha-1.0))
    g_star = np.log(pdf_P/pdf_Q)
    
    print('Renyi-divergence (true):\t\t {:.4}'.format(div_value_true))
    print('Renyi-divergence-DV (estimated):\t\t {:.4}'.format(div_value_est))
    print()

if mthd=="Renyi-CC":
    div_dense = Renyi_Divergence_CC(discriminator, alpha, epochs, lr, m, fl_act_func_CC)
    div_dense.train(data_P, data_Q)
    div_value_est = float(div_dense.estimate(data_P, data_Q))

    if fw == 'torch':
        x = x.reshape(-1, 1)

    g_est = div_dense.discriminate(x)
    
    div_value_true = np.log(np.sum(np.power(pdf_P, alpha)*np.power(pdf_Q, 1.0-alpha)) * dx) / (alpha*(alpha-1.0))
    g_star = np.log(pdf_P/pdf_Q)
    
    print('Renyi-divergence (true):\t\t {:.4}'.format(div_value_true))
    print('Renyi-divergence-CC (estimated):\t\t {:.4}'.format(div_value_est))
    print()

if mthd=="rescaled-Renyi-CC":
    div_dense = Renyi_Divergence_CC_rescaled(discriminator, alpha, epochs, lr, m, fl_act_func_CC)
    div_dense.train(data_P, data_Q)
    div_value_est = float(div_dense.estimate(data_P, data_Q))

    if fw == 'torch':
        x = x.reshape(-1, 1)

    g_est = div_dense.discriminate(x)
    
    div_value_true = np.log(np.sum(np.power(pdf_P, alpha)*np.power(pdf_Q, 1.0-alpha)) * dx) / (alpha-1.0)
    g_star = np.log(pdf_P/pdf_Q)
    
    print('rescaled-Renyi-divergence (true):\t\t {:.4}'.format(div_value_true))
    print('rescaled-Renyi-divergence-CC (estimated):\t\t {:.4}'.format(div_value_est))
    print()

if mthd=="Renyi-WCR":
    div_dense = Renyi_Divergence_WCR(discriminator, 'Inf', epochs, lr, m, fl_act_func_CC)
    div_dense.train(data_P, data_Q)
    div_value_est = float(div_dense.estimate(data_P, data_Q))

    if fw == 'torch':
        x = x.reshape(-1, 1)

    g_est = div_dense.discriminate(x)
    
    div_value_true = np.log(max(pdf_P/pdf_Q)) / (alpha-1.0)
    g_star = np.log(pdf_P/pdf_Q)
    
    print('worst-case-regret (true):\t\t {:.4}'.format(div_value_true))
    print('worst-case-regret-CC (estimated):\t\t {:.4}'.format(div_value_est))
    print()


# plot optimizer
if fw=='torch':
    g_est = g_est.detach().numpy()
else:
    g_est = g_est.numpy()

g_est = np.reshape(g_est, (NoP,))

start = bisect_right(x,eff_vals_P[0])
end = bisect_left(x,eff_vals_P[1]) - 1

fig, (ax0, ax1) = plt.subplots(1, 2)
ax0.plot(x, g_star, 'b', label='Ground truth')
ax0.plot(x, g_est-max(g_est)+max(g_star), 'r--', label='Estimated')
ax0.set_title('Optimizer')
ax0.legend(frameon=False)

ax1.plot(x[start:end], g_star[start:end], 'b', label='Ground truth')
ax1.plot(x[start:end], g_est[start:end]-max(g_est)+max(g_star), 'r--', label='Estimated')
ax1.set_title('Effective Optimizer')
ax1.legend(frameon=False)
fig.suptitle(mthd)
plt.show()
