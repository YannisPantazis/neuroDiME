import numpy as np
import pandas as pd
import csv
import os
import sys
import argparse
#import json
import matplotlib.pyplot as plt
import torchvision.datasets as datasets
import time
import torch.autograd as autograd
from torch.autograd import Variable
import seaborn as sns
import glob

current_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from models.torch_model import *
from models.Divergences_torch import *

start = time.perf_counter()

# read input arguments
parser = argparse.ArgumentParser(description='Neural-based Estimation of Divergences between MNIST Digit Distributions')
parser.add_argument('--method', default='KLD-DV', type=str, metavar='method',
                    help='values: IPM, KLD-DV, KLD-LT, squared-Hel-LT, chi-squared-LT, JS-LT, alpha-LT, Renyi-DV, Renyi-CC, rescaled-Renyi-CC, Renyi-CC-WCR')
                        
parser.add_argument('--sample_size', default=10000, type=int, metavar='N')
parser.add_argument('--batch_size', default=128, type=int, metavar='m')
parser.add_argument('--lr', default=0.001, type=float)
parser.add_argument('--epochs', default=250, type=int,
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

class Gradient_Penalty(Discriminator_Penalty):
    def __init__(self, weight, L):
        Discriminator_Penalty.__init__(self, weight)
        self.L = L
    
    def get_Lip_const(self):
        return self.L

    def set_Lip_const(self, L):
        self.L = L
        
    def evaluate(self, c, images, samples, labels=None):
        assert images.size() == samples.size()
        jump = torch.rand(images.shape[0], 1).cuda()
        jump_ = jump.expand(images.shape[0], images.nelement()//images.shape[0]).contiguous().view(images.shape[0],1,28,28)
        interpolated = Variable(images*jump_ + (1-jump_)*samples, requires_grad = True)
        if labels is not None:
            c_ = c(interpolated, labels)
        else:
            c_ = c(interpolated)
        gradients = autograd.grad(c_, interpolated, grad_outputs=(torch.ones(c_.size()).cuda()),create_graph = True, retain_graph = True)[0]
        return self.get_penalty_weight()*((self.L-gradients.norm(2,dim=1))**2).mean()

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
    discriminator_penalty=Gradient_Penalty(gp_weight, L)
    test_name=f'MNIST_{mthd}_GP_divergence_demo_torch'
    title = f'Colormap of {mthd} estimates between all pair of digits using Torch with Gradient Penalty'
    save_path = f"colormap_torch_{mthd}_GP.png"
else:
    discriminator_penalty=None
    test_name=f'MNIST_{mthd}_divergence_demo_torch'
    title = f'Colormap of {mthd} estimates between all pair of digits using Torch'
    save_path = f"colormap_torch_{mthd}.png"

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
        discriminator = Discriminator_MNIST().to(device)
        
        # print()
        # print('Discriminator Summary:')
        # summary(discriminator, (1, 28, 28))
        #construct optimizers
        if optimizer == 'Adam':
            disc_optimizer = torch.optim.Adam(discriminator.parameters(), lr=lr)

        if optimizer == 'RMS':
            disc_optimizer = torch.optim.RMSprop(discriminator.parameters(), lr=lr)

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

        #Save result
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
        divergence_estimates=divergence_CNN.train(data_P, data_Q)
            # save the model
            # print('Saving Model...')
            # torch.save(discriminator, model_file)
        estimate = divergence_CNN.estimate(data_P, data_Q).detach().cpu().numpy()
        tries = 0
        while np.isnan(estimate):
            print('Nan estimate, retraining...')
            divergence_estimates=divergence_CNN.train(data_P, data_Q, device)
            estimate = divergence_CNN.estimate(data_P, data_Q).detach().cpu().numpy()
            if tries > 10:
                print('Too many tries, stopping the training...')
                break
            
        print(f'{mthd} estimate between digits {P_digit} and {Q_digit}: {estimate}')


        with open(test_name+'/'+mthd+'_div_estimate_P_digit_' +str(P_digit)+'_Q_digit_' +str(Q_digit)+'_N_'+str(N)+'_m_'+str(m)+'_Lrate_{:.1e}'.format(lr)+'_epochs_'+str(epochs)+'_alpha_{:.1f}'.format(alpha)+'_L_{:.1f}'.format(L)+'_gp_weight_{:.1f}'.format(gp_weight)+'_GP_'+str(use_GP)+'_run_num_'+str(run_num)+'.csv', "w") as output:
            writer = csv.writer(output, lineterminator='\n')
            writer.writerow([estimate])

# Reading and storing all the estimates from all the csv files
csv_files = sorted(glob.glob(test_name + '/' + "*.csv"))
numbers = []
    
for file in csv_files:
    divergence = pd.read_csv(file, header=None).iloc[0,0]
    if math.isnan(divergence):
        numbers.append(math.inf)
    else:
        numbers.append(np.log10((np.abs(divergence))))

numbers = np.array(numbers)
numbers = numbers.reshape(10,10)

plt.figure(figsize=(10, 10))
sns.heatmap(numbers, annot=True, fmt='.2f', linewidth=2, cmap='coolwarm')
plt.title(title)
plt.savefig(save_path)
plt.show()
plt.close()


print(f'--- {time.perf_counter() - start} seconds ---')