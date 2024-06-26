#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import tensorflow as tf
import numpy as np
import os
import sys
import csv

#N_samples=100000 #total number of samples to use
N_samples=int(sys.argv[1]) 


epochs = 200000 #number of SGD epochs
mb_size = 1000 #minibatch size for SGD
SF = 10000 #save test samples from generator every SF iterations
k=5 #number of discriminator iterations per generator iteration
NoT = 10000 #number of test samples to save 


#names of GAN methods we want to test: used to name saved sample files
methods=[ 'W','W',2.0,5.0,10.0,100.0] 
optimizer=['RMS','RMS','RMS','RMS','RMS','RMS'] 
GP_type=[2,1,1,1,1,1] #1 or 2 sided grad penalty, 0 means no GP
learning_rates=[2e-4,2e-4,2e-4,2e-4,2e-4,2e-4]
reverse=[0,0,1,1,1,1] #set to 1 to use reverse GAN, 0 for forward GAN
L=1.0
beta=10.0

D_hidden_layers=[64,32,16] #sizes of hidden layers for the discriminator
G_hidden_layers=[64,32,16] #sizes of hidden layers for the generator
Z_dim=10 #dimension of the noise source for the generator





#data distribution
test_name='2Dstudent_submanifold_gan'
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



#construct variables for the neural networks:

D_layers=[X_dim]+D_hidden_layers+[1]  #dimensions of layers of disciminator
G_layers=[Z_dim]+G_hidden_layers+[X_dim] #dimensions of layers of generator

#define neural network network structure/parameters
def xavier_init(size):
    in_dim = size[0]
    xavier_stddev = 1.0 / tf.sqrt(in_dim / 2.)
    return tf.random_normal(shape=size, stddev=xavier_stddev)

def initialize_W(layers):
    W_init=[]
    num_layers = len(layers)
    for l in range(0,num_layers-1):
        W_init.append(xavier_init(size=[layers[l], layers[l+1]]))
    return W_init

def initialize_NN(layers,W_init):
    NN_W = []
    NN_b = []
    num_layers = len(layers)
    for l in range(0,num_layers-1):
        W = tf.Variable(W_init[l])
        b = tf.Variable(tf.zeros([1,layers[l+1]], dtype=tf.float32), dtype=tf.float32)
        NN_W.append(W)
        NN_b.append(b)
    return NN_W, NN_b

#discriminator variable
X = tf.placeholder(tf.float32, shape=[None, X_dim])
#generator variable
Z = tf.placeholder(tf.float32, shape=[None, Z_dim])
#variable for the mixture of Q and P
W=tf.placeholder(tf.float32, shape=[None, X_dim])

N_methods=len(methods)

D_init=initialize_W(D_layers)
D_W_array=[]
D_b_array=[]
theta_D_array=[]
for j in range(N_methods):
    [D_W,D_b]=initialize_NN(D_layers,D_init)
    D_W_array.append(D_W)
    D_b_array.append(D_b)
    theta_D_array.append([D_W,D_b])


G_init=initialize_W(G_layers)
G_W_array=[]
G_b_array=[]
theta_G_array=[]
for j in range(N_methods):
    [G_W,G_b]=initialize_NN(G_layers,G_init)
    G_W_array.append(G_W)
    G_b_array.append(G_b)
    theta_G_array.append([G_W,G_b])




def discriminator(x, D_W, D_b):
        num_layers = len(D_W) + 1
        
        h = x
        for l in range(0,num_layers-2):
            W = D_W[l]
            b = D_b[l]
            h = tf.nn.relu(tf.add(tf.matmul(h, W), b))

        W = D_W[-1]
        b = D_b[-1]
        out=tf.matmul(h, W) + b
        
        return out  


    
def generator(x, G_W, G_b):
        num_layers = len(G_W) + 1
        
        h = x
        for l in range(0,num_layers-2):
            W = G_W[l]
            b = G_b[l]
            h = tf.nn.relu(tf.add(tf.matmul(h, W), b))
        
        W = G_W[-1]
        b = G_b[-1]
        out =  tf.matmul(h, W) + b

        
        return out  
    
 
def f_alpha_star(y,alpha):
    return tf.math.pow(tf.nn.relu(y),alpha/(alpha-1.0))*tf.math.pow((alpha-1.0),alpha/(alpha-1.0))/alpha+1/(alpha*(alpha-1.0))
   


def sample_Z(m, n):
    return np.random.normal(0., 1.0, size=[m, n])



def embed_data(x):
    z=np.concatenate((offset*np.ones([x.shape[0],di]),x),axis=1)
    z=np.concatenate((z,offset*np.ones([x.shape[0],df])),axis=1)
    
    return z

def sample_Q(m):
    u=np.random.chisquare(nu,size=[m,1])
    x=np.divide(np.random.normal(0.0,1.0,size=[m,d]),np.sqrt(u/nu))
    
    idx=np.random.randint(0,4,size=[m])
    x=x+Delta_array[idx,:]
    
    return embed_data(x)


G_sample_array=[]
D_real_array=[]
D_fake_array=[]
grad_D_array=[]
for j in range(N_methods):
    G_sample_array.append(generator(Z,G_W_array[j],G_b_array[j]))
    D_real_array.append(discriminator(X,D_W_array[j],D_b_array[j]))
    D_fake_array.append(discriminator(G_sample_array[j],D_W_array[j],D_b_array[j]))
    grad_D_array.append(tf.gradients( discriminator(W,D_W_array[j],D_b_array[j]),W))



    
gen_loss_array=[]
loss_array=[]
# Losses:
# -------------------

for k in range(N_methods):  
    if GP_type[k]==1:
        GP=beta*tf.reduce_mean(tf.nn.relu(tf.math.reduce_sum(tf.math.square(grad_D_array[k]),2)/L**2-1.0))
        #GP=beta*tf.reduce_mean(tf.math.square(tf.nn.relu(tf.math.sqrt(tf.math.reduce_sum(tf.math.square(grad_D_array[k]),2))/L-1.0)))
    elif GP_type[k]==2:
        GP=beta*tf.reduce_mean(tf.math.square(tf.math.sqrt(tf.math.reduce_sum(tf.math.square(grad_D_array[k]),2))/L-1.0))
    elif GP_type[k]==0:
        GP=0

        
    if methods[k]=='W': #Wasserstein GAN
                
        if reverse[k]==1:
            D_Q=D_fake_array[k]
            D_P=D_real_array[k]
            
            G_loss= tf.reduce_mean(D_Q)-GP
            
        else:
            D_Q=D_real_array[k]
            D_P=D_fake_array[k]
            
            G_loss=-tf.reduce_mean(D_P)-GP
               
        loss= tf.reduce_mean(D_Q)-tf.reduce_mean(D_P)-GP  
        
        
    elif methods[k]=='W_original': #Grad-penaly Wasserstein GAN method from the original paper (GP not consistent)
                
        if reverse[k]==1:
            D_Q=D_fake_array[k]
            D_P=D_real_array[k]
            
            G_loss= tf.reduce_mean(D_Q)
            
        else:
            D_Q=D_real_array[k]
            D_P=D_fake_array[k]
            
            G_loss=-tf.reduce_mean(D_P)
               
        loss= tf.reduce_mean(D_Q)-tf.reduce_mean(D_P)-GP     

        
    elif methods[k]=='KL': #Lipschitz KL GAN
        D_P_max=  tf.reduce_mean(D_P)
        if reverse[k]==1:
            D_Q=D_fake_array[k]
            D_P=D_real_array[k]
            G_loss=tf.reduce_mean(D_Q)-GP
            
        else:
            D_Q=D_real_array[k]
            D_P=D_fake_array[k]

            G_loss=-D_P_max-tf.math.log(tf.reduce_mean(tf.math.exp(D_P-D_P_max)))-GP
        
        loss=tf.reduce_mean(D_Q)-D_P_max-tf.math.log(tf.reduce_mean(tf.math.exp(D_P-D_P_max)))-GP

    elif methods[k]=='infty': #alpha->\infty limit of objective function

        if reverse[k]==1:
            D_Q=D_fake_array[k]
            D_P=D_real_array[k]
            G_loss=tf.reduce_mean(D_Q)-GP

            
        else:
            D_Q=D_real_array[k]
            D_P=D_fake_array[k]
            G_loss=-tf.reduce_mean(tf.nn.relu(D_P))-GP
                        
        loss=tf.reduce_mean(D_Q)-tf.reduce_mean(tf.nn.relu(D_P))-GP

    else: #Lipschitz alpha-div GAN

        if reverse[k]==1:
            D_Q=D_fake_array[k]
            D_P=D_real_array[k]
            G_loss=tf.reduce_mean(D_Q)-GP

            
        else:
            D_Q=D_real_array[k]
            D_P=D_fake_array[k]
            G_loss=-tf.reduce_mean(f_alpha_star(D_P,methods[k]))-GP
            
        loss=tf.reduce_mean(D_Q)-tf.reduce_mean(f_alpha_star(D_P,methods[k]))-GP


    loss_array.append(loss)  
    gen_loss_array.append(G_loss)



D_solver_array=[]
G_solver_array=[]
for j in range(N_methods):
    if optimizer[j]=='RMS':
        D_solver_array.append( tf.train.RMSPropOptimizer(learning_rates[j]).minimize(-loss_array[j], var_list=theta_D_array[j]))
        G_solver_array.append( tf.train.RMSPropOptimizer(learning_rates[j]).minimize(gen_loss_array[j], var_list=theta_G_array[j]))
    elif optimizer[j]=='Adam':
        D_solver_array.append( tf.train.AdamOptimizer(learning_rate=learning_rates[j]).minimize(-loss_array[j], var_list=theta_D_array[j]))
        G_solver_array.append( tf.train.AdamOptimizer(learning_rate=learning_rates[j]).minimize(gen_loss_array[j], var_list=theta_G_array[j]))






config = tf.ConfigProto(device_count={'GPU': 0})
sess = tf.Session(config=config)
sess.run(tf.global_variables_initializer())

if not os.path.exists(test_name+'/csv/'):
    os.makedirs(test_name+'/csv/')

      

samples=sample_Q(N_samples)


for it in range(epochs+1):
    if it % SF == 0 or it==100 or it == 500 or (it<=30000 and it % 1000==0):
        Z_new = sample_Z(NoT, Z_dim)
        for j in range(N_methods):
            G_samples = sess.run(G_sample_array[j], feed_dict={Z: Z_new})
            
            method_name=''
            if reverse[j]==1:
                method_name='r_'
                
            method_name=method_name+'GP'+str(GP_type[j])+'_'
                
                     
            if methods[j]=='W':
                method_name=method_name+'W'            
            elif methods[j]=='W_original':
                method_name=method_name+'W_orig'
            elif methods[j]=='KL':
                method_name=method_name+'Lip_KL'
            elif methods[j]=='infty':
                method_name=method_name+'Lip_alpha_infty'
            else:
                method_name=method_name+'Lip_alpha_{:.1f}'.format(methods[j])
           
            with open(test_name+'/csv/'+method_name+'_samples_'+str(N_samples)+'_iter_'+str(it)+'_'+optimizer[j]+'Lrate_{:.1e}'.format(learning_rates[j])+'.csv', "w") as output:
                writer = csv.writer(output, lineterminator='\n')
                for val in G_samples:
                    writer.writerow(val)
        
        print('Iter: {}'.format(it)) 
        print()

        
    
    for ell in range(k):
        idx=np.random.randint(0,N_samples,size=[mb_size])
        X_mb = samples[idx,:]
        
        Z_new=sample_Z(mb_size, Z_dim)
        T=np.random.uniform(0,1,[mb_size,1])
        for j in range(N_methods):
            G_samples = sess.run(G_sample_array[j], feed_dict={Z: Z_new})
            W_new=np.multiply(T,X_mb)+np.multiply(1-T,G_samples)
            sess.run(D_solver_array[j], feed_dict={X: X_mb, Z: Z_new, W: W_new})

    idx=np.random.randint(0,N_samples,size=[mb_size])
    X_mb = samples[idx,:]
    Z_new=sample_Z(mb_size, Z_dim)
    T=np.random.uniform(0,1,[mb_size,1])
    for j in range(N_methods):
        G_samples = sess.run(G_sample_array[j], feed_dict={Z: Z_new})
        W_new=np.multiply(T,X_mb)+np.multiply(1-T,G_samples)
        sess.run(G_solver_array[j], feed_dict={X: X_mb, Z: Z_new, W: W_new})



