import numpy as np
import pandas as pd
from Divergences import *
import argparse
#import os
import json
#from aux_funcs import *
from numpy import genfromtxt

# read input arguments
parser = argparse.ArgumentParser(description='Gamma Renyi Divergence')

parser.add_argument('--method', default='all', type=str, metavar='method',
                    help='values: all, DV, DV-log, IC, IC-inf, log-Dfalpha, Dfalpha')
parser.add_argument('--sample_size', default=10000, type=int, metavar='N')
parser.add_argument('--batch_size', default=1000, type=int, metavar='m')
parser.add_argument('--lr', default=0.001, type=float)
parser.add_argument('--epochs', default=100, type=int,
                    help='number of total epochs to run')
parser.add_argument('--alpha', default=2.0, type=float, metavar='alpha')
parser.add_argument('--no_repeats', default=10, type=int)

parser.add_argument('--spectral_norm', action='store_true')
parser.add_argument('--no-spectral_norm', dest='spectral_norm', action='store_false')
parser.set_defaults(spectral_norm=True)

parser.add_argument('--reverse', action='store_true')
parser.set_defaults(spectral_norm=False)

opt = parser.parse_args()
opt_dict = vars(opt)
print('parsed options:', opt_dict)

mthd = opt_dict['method']
N = opt_dict['sample_size']
m = opt_dict['batch_size']
lr = opt_dict['lr']
epochs = opt_dict['epochs']
alpha = opt_dict['alpha']
NoRep = opt_dict['no_repeats']
spec_norm = opt_dict['spectral_norm']
reverse_order = opt_dict['reverse']
fl_act_func_IC = 'poly-softplus' # abs, softplus, poly-softplus
fl_act_func_DV = 'abs'

layers_list = [32, 32] # Tasos NN: [16, 16, 8]
act_func = 'relu'

#pr_val = [0.0, 0.001, 0.003, 0.005, 0.01]
pr_val = [0.0, 0.001, 0.002, 0.003, 0.004, 0.005, 0.0075, 0.01, 0.015, 0.02, 0.05, 0.1]
NoPr = len(pr_val)
print(pr_val)


# load data
data_h = pd.read_csv("csv/healthy.csv")
data_s = pd.read_csv("csv/sick.csv")

d = data_h.shape[1]
layers_list.insert(0, d)

# run
WD_est = np.zeros((NoPr, NoRep))
KLD_DV_est = np.zeros((NoPr, NoRep))
RD_DV_est = np.zeros((NoPr, NoRep))
RD_DV_log_est = np.zeros((NoPr, NoRep))
RD_IC_est = np.zeros((NoPr, NoRep))
RD_IC_inf_est = np.zeros((NoPr, NoRep))
RD_log_est = np.zeros((NoPr, NoRep))
Df_alpha_est = np.zeros((NoPr, NoRep))

for i1 in range(NoPr):
    p = pr_val[i1]
    for i2 in range(NoRep):
        # make the class instance for each estimator
        tf.keras.backend.clear_session()

        # number of sample per class
        N2 = int(np.ceil(N*p))
        N1 = N - N2
        print(N1,N2,p)

        # select which samples via indexing
        idx0 = np.random.randint(data_h.shape[0], size=N)
        idx1 = np.random.randint(data_h.shape[0], size=N1)
        idx2 = np.random.randint(data_s.shape[0], size=N2)

        # create datasets
        dataset_pure = data_h.loc[idx0]
        dataset_cntmd = pd.concat([data_h.loc[idx1], data_s.loc[idx2]], ignore_index=True)

        # shuffle
        idx = np.random.randint(N, size=N)
        dataset_cntmd = dataset_cntmd.loc[idx]
        
        if reverse_order:
            P_dataset = tf.data.Dataset.from_tensor_slices(dataset_cntmd).batch(m)
            Q_dataset = tf.data.Dataset.from_tensor_slices(dataset_pure).batch(m)
        else:
            P_dataset = tf.data.Dataset.from_tensor_slices(dataset_pure).batch(m)
            Q_dataset = tf.data.Dataset.from_tensor_slices(dataset_cntmd).batch(m)

        # choose estimation method
        if mthd=="all":
            WassD = Wasserstein_Distance_dense(layers_list, act_func, epochs, lr, m, spec_norm)
            WassD.train(Q_dataset, P_dataset)
            WD_est[i1,i2] = float(WassD.estimate(dataset_cntmd.to_numpy(), dataset_pure.to_numpy()))
                    
            KLD_DV = KL_Divergence_DV_dense(layers_list, act_func, epochs, lr, m, spec_norm)
            KLD_DV.train(Q_dataset, P_dataset)
            KLD_DV_est[i1,i2] = float(KLD_DV.estimate(dataset_cntmd.to_numpy(), dataset_pure.to_numpy()))
                    
            RD_DV = Renyi_Divergence_DV_dense(layers_list, act_func, alpha, epochs, lr, m, spec_norm)
            RD_DV.train(Q_dataset, P_dataset)
            RD_DV_est[i1,i2] = float(RD_DV.estimate(dataset_cntmd.to_numpy(), dataset_pure.to_numpy()))
            
            RD_DV_log = Renyi_Divergence_DV_log_dense(layers_list, act_func, alpha, epochs, lr, m, spec_norm, fl_act_func_DV)
            RD_DV_log.train(Q_dataset, P_dataset)
            RD_DV_log_est[i1,i2] = float(RD_DV_log.estimate(dataset_cntmd.to_numpy(), dataset_pure.to_numpy()))

            RD_inf_conv = Renyi_Divergence_inf_conv_dense(layers_list, act_func, alpha, epochs, lr, m, spec_norm, fl_act_func_IC, False)
            RD_inf_conv.train(Q_dataset, P_dataset)
            RD_IC_est[i1,i2] = float(RD_inf_conv.estimate(dataset_cntmd.to_numpy(), dataset_pure.to_numpy()))

            RD_log_Df_alpha = Renyi_Divergence_log_Df_alpha_dense(layers_list, act_func, alpha, epochs, lr, m, spec_norm)
            RD_log_Df_alpha.train(Q_dataset, P_dataset)
            RD_log_est[i1,i2] = float(RD_log_Df_alpha.estimate(dataset_cntmd.to_numpy(), dataset_pure.to_numpy()))
            
            Df_alpha = alpha_Divergence_dense(layers_list, act_func, alpha, epochs, lr, m, spec_norm)
            Df_alpha.train(Q_dataset, P_dataset)
            Df_alpha_est[i1,i2] = float(Df_alpha.estimate(dataset_cntmd.to_numpy(), dataset_pure.to_numpy()))

            print('Wasserstein:\t\t {:.4}'.format(WD_est[i1,i2]))
            print('KLD-DV:\t\t {:.4}'.format(KLD_DV_est[i1,i2]))
            print('RD-DV:\t\t {:.4}'.format(RD_DV_est[i1,i2]))
            print('RD-DV-log:\t {:.4}'.format(RD_DV_log_est[i1,i2]))
            print('RD-IC:\t\t {:.4}'.format(RD_IC_est[i1,i2]))
            print('RD-log-Df_alpha: {:.4}'.format(RD_log_est[i1,i2]))
            print('Df_alpha:\t {:.4}'.format(Df_alpha_est[i1,i2]))
            print()
        
        if mthd=="WD":
            WassD = Wasserstein_Distance_dense(layers_list, act_func, epochs, lr, m, spec_norm)
            WassD.train(Q_dataset, P_dataset)
            WD_est[i1,i2] = float(WassD.estimate(dataset_cntmd.to_numpy(), dataset_pure.to_numpy()))
            
            print('Wasserstein:\t\t {:.4}'.format(WD_est[i1,i2]))
            print()
            
        if mthd=="KLD-DV":
            KLD_DV = KL_Divergence_DV_dense(layers_list, act_func, epochs, lr, m, spec_norm)
            KLD_DV.train(Q_dataset, P_dataset)
            KLD_DV_est[i1,i2] = float(KLD_DV.estimate(dataset_cntmd.to_numpy(), dataset_pure.to_numpy()))
                        
            print('KLD-DV:\t\t {:.4}'.format(KLD_DV_est[i1,i2]))
            print()
        
        if mthd=="DV":
            RD_DV = Renyi_Divergence_DV_dense(layers_list, act_func, alpha, epochs, lr, m, spec_norm)
            RD_DV.train(Q_dataset, P_dataset)
            RD_DV_est[i1,i2] = float(RD_DV.estimate(dataset_cntmd.to_numpy(), dataset_pure.to_numpy()))
            
            print('RD-DV: {:.4}'.format(RD_DV_est[i1,i2]))
            print()
                                    
        if mthd=="DV-log":
            RD_DV_log = Renyi_Divergence_DV_log_dense(layers_list, act_func, alpha, epochs, lr, m, spec_norm, fl_act_func_DV)
            RD_DV_log.train(Q_dataset, P_dataset)
            RD_DV_log_est[i1,i2] = float(RD_DV_log.estimate(dataset_cntmd.to_numpy(), dataset_pure.to_numpy()))
            
            print('RD-DV-log: {:.4}'.format(RD_DV_log_est[i1,i2]))
            print()
                    
        if mthd=="IC":
            RD_inf_conv = Renyi_Divergence_inf_conv_dense(layers_list, act_func, alpha, epochs, lr, m, spec_norm, fl_act_func_IC, False)
            RD_inf_conv.train(Q_dataset, P_dataset)
            RD_IC_est[i1,i2] = float(RD_inf_conv.estimate(dataset_cntmd.to_numpy(), dataset_pure.to_numpy()))
            
            print('RD-IC: {:.4}'.format(RD_IC_est[i1,i2]))
            print()
                    
        if mthd=="IC-rescaled":
            RD_inf_conv = Renyi_Divergence_inf_conv_dense(layers_list, act_func, alpha, epochs, lr, m, spec_norm, fl_act_func_IC, True)
            RD_inf_conv.train(Q_dataset, P_dataset)
            RD_IC_est[i1,i2] = float(RD_inf_conv.estimate(dataset_cntmd.to_numpy(), dataset_pure.to_numpy()))
            
            print('RD-IC-rescaled: {:.4}'.format(RD_IC_est[i1,i2]))
            print()
        
        if mthd=="IC-inf":
            RD_inf_conv_inf = Renyi_Divergence_inf_conv_inf_dense(layers_list, act_func, None, epochs, lr, m, spec_norm, fl_act_func_IC)
            RD_inf_conv_inf.train(Q_dataset, P_dataset)
            RD_IC_inf_est[i1,i2] = float(RD_inf_conv_inf.estimate(dataset_cntmd.to_numpy(), dataset_pure.to_numpy()))
            
            print('RD-IC-inf: {:.4}'.format(RD_IC_inf_est[i1,i2]))
            print()
            
        if mthd=="log-Dfalpha":
            RD_log_Df_alpha = Renyi_Divergence_log_Df_alpha_dense(layers_list, act_func, alpha, epochs, lr, m, spec_norm)
            RD_log_Df_alpha.train(Q_dataset, P_dataset)
            RD_log_est[i1,i2] = float(RD_log_Df_alpha.estimate(dataset_cntmd.to_numpy(), dataset_pure.to_numpy()))
                                    
            print('RD-log-Df_alpha: {:.4}'.format(RD_log_est[i1,i2]))
            print()
            
        if mthd=="Dfalpha":
            Df_alpha = alpha_Divergence_dense(layers_list, act_func, alpha, epochs, lr, m, spec_norm)
            Df_alpha.train(Q_dataset, P_dataset)
            Df_alpha_est[i1,i2] = float(Df_alpha.estimate(dataset_cntmd.to_numpy(), dataset_pure.to_numpy()))
            
            print('Df_alpha: {:.4}'.format(Df_alpha_est[i1,i2]))
            print()
            

prefix = ''
if reverse_order:
    prefix = 'r_'

# save the RD values in file(s)
if mthd=="all":
    np.save('results/bio/'+prefix+'WD_est_sample_size_'+str(N)+'_batch_size_'+str(m)+'.npy', WD_est)
    np.save('results/bio/'+prefix+'KLD_DV_est_sample_size_'+str(N)+'_batch_size_'+str(m)+'.npy', KLD_DV_est)
    np.save('results/bio/'+prefix+'RD_DV_est_sample_size_'+str(N)+'_batch_size_'+str(m)+'_alpha_'+str(alpha)+'.npy', RD_DV_est)
    np.save('results/bio/'+prefix+'RD_DV_log_est_sample_size_'+str(N)+'_batch_size_'+str(m)+'_alpha_'+str(alpha)+'.npy', RD_DV_log_est)
    np.save('results/bio/'+prefix+'RD_IC_est_sample_size_'+str(N)+'_batch_size_'+str(m)+'_alpha_'+str(alpha)+'.npy', RD_IC_est)
    np.save('results/bio/'+prefix+'RD_log_est_sample_size_'+str(N)+'_batch_size_'+str(m)+'_alpha_'+str(alpha)+'.npy', RD_log_est)
    np.save('results/bio/'+prefix+'Df_alpha_est_sample_size_'+str(N)+'_batch_size_'+str(m)+'_alpha_'+str(alpha)+'.npy', Df_alpha_est)
        
if mthd=="WD":
    np.save('results/bio/'+prefix+'WD_est_sample_size_'+str(N)+'_batch_size_'+str(m)+'.npy', WD_est)
        
if mthd=="KLD-DV":
    np.save('results/bio/'+prefix+'KLD_DV_est_sample_size_'+str(N)+'_batch_size_'+str(m)+'.npy', KLD_DV_est)

if mthd=="DV":
    np.save('results/bio/'+prefix+'RD_DV_est_sample_size_'+str(N)+'_batch_size_'+str(m)+'_alpha_'+str(alpha)+'.npy', RD_DV_est)

if mthd=="DV-log":
    np.save('results/bio/'+prefix+'RD_DV_log_est_sample_size_'+str(N)+'_batch_size_'+str(m)+'_alpha_'+str(alpha)+'.npy', RD_DV_log_est)

if mthd=="IC-rescaled":
    np.save('results/bio/'+prefix+'RD_IC_est_rescaled_sample_size_'+str(N)+'_batch_size_'+str(m)+'_alpha_'+str(alpha)+'.npy', RD_IC_est)

if mthd=="IC":
    np.save('results/bio/'+prefix+'RD_IC_est_sample_size_'+str(N)+'_batch_size_'+str(m)+'_alpha_'+str(alpha)+'.npy', RD_IC_est)

if mthd=="IC-inf":
    np.save('results/bio/'+prefix+'RD_IC_inf_est_sample_size_'+str(N)+'_batch_size_'+str(m)+'_alpha_inf.npy', RD_IC_inf_est)

if mthd=="log-Dfalpha":
    np.save('results/bio/'+prefix+'RD_log_est_sample_size_'+str(N)+'_batch_size_'+str(m)+'_alpha_'+str(alpha)+'.npy', RD_log_est)
    
if mthd=="Dfalpha":
    np.save('results/bio/'+prefix+'Df_alpha_est_sample_size_'+str(N)+'_batch_size_'+str(m)+'_alpha_'+str(alpha)+'.npy', Df_alpha_est)
