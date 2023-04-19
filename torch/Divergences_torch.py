import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import grad as torch_grad


'''
Divergence D(P||Q) between random variables x~P, y~Q.
Parent class where the common parameters and the common functions are defined.
'''
class Divergence(nn.Module):

    # initialize
    def __init__(self, discriminator, disc_optimizer, epochs, batch_size, discriminator_penalty=None):
        super(Divergence, self).__init__()
        self.batch_size = batch_size
        self.epochs = epochs
        self.discriminator_penalty = discriminator_penalty
        self.discriminator = discriminator
        self.disc_optimizer = disc_optimizer

    def __repr__(self):
        return 'discriminator: {}'.format(self.discriminator)

    def discriminate(self, x): # g(x)
        if not torch.is_tensor(x):
            x = torch.from_numpy(x)
            
        y = self.discriminator(x.float())
        return y
    
    def eval_var_formula(self, x, y): # depends on the variational formula to be used
        return None
    
    def estimate(self, x, y): # same as self.eval_val_formula()
        divergece_loss = self.eval_var_formula(x,y)
        return divergece_loss

    def discriminator_loss(self, x, y): # same as self.estimate() (in principle)
        divergence_loss = self.eval_var_formula(x, y)
        return divergence_loss
    
    def train_step(self, x, y):
        # discriminator's parameters update
        self.disc_optimizer.zero_grad()
        x.requires_grad_(True)
        y.requires_grad_(True)
        loss = -self.discriminator_loss(x, y) # with minus because we maximize the discriminator loss
        if self.discriminator_penalty is not None:
            loss = loss + self.discriminator_penalty.evaluate(self.discriminator, x, y)
        
        loss.backward()
        self.disc_optimizer.step()

    def train(self, data_P, data_Q, save_estimates=True):
        # dataset slicing into minibatches
        P_dataset = torch.utils.data.DataLoader(data_P, batch_size=self.batch_size, shuffle=True)
        Q_dataset = torch.utils.data.DataLoader(data_Q, batch_size=self.batch_size, shuffle=True)

        estimates = []
        for epoch in range(self.epochs):
            for P_batch, Q_batch in zip(P_dataset, Q_dataset):
                self.train_step(P_batch, Q_batch)
            
            if save_estimates:
                estimates.append(float(self.estimate(P_batch, Q_batch)))

        return estimates

    def get_discriminator(self):
        return self.discriminator

    def set_discriminator(self, discriminator):
        self.discriminator = discriminator

    def get_no_epochs(self):
        return self.epochs

    def set_no_epochs(self, epochs):
        self.epochs = epochs

    def get_batch_size(self):
        return self.batch_size

    def set_batch_size(self, BATCH_SIZE):
        self.batch_size = BATCH_SIZE

    def get_learning_rate(self):
        return self.learning_rate

    def set_learning_rate(self, lr):
        self.learning_rate = lr


'''
IPM class
'''
class IPM(Divergence):

    def eval_var_formula(self, x, y):
        D_P = self.discriminate(x)
        D_Q = self.discriminate(y)

        D_loss_P = torch.mean(D_P)
        D_loss_Q = torch.mean(D_Q)

        D_loss = D_loss_P - D_loss_Q
        return D_loss


'''
f-divergence class (parent class)
D_f(P||Q), x~P, y~Q.
'''
class f_Divergence(Divergence):
 
    # Legendre transform of f
    def f_star(self, y):
        return None
    
    def final_layer_activation(self, y):
        return y

    def eval_var_formula(self, x, y):
        D_P = self.discriminate(x)
        D_P = self.final_layer_activation(D_P)
        D_Q = self.discriminate(y)
        D_Q = self.final_layer_activation(D_Q)
        
        D_loss_P = torch.mean(D_P)
        D_loss_Q = torch.mean(self.f_star(D_Q))
        
        D_loss = D_loss_P - D_loss_Q
        return D_loss


'''
Kullback-Leibler (KL) divergence class (based on Legendre transform)
KL(P||Q), x~P, y~Q.
'''
class KLD_LT(f_Divergence):
 
    # Legendre transform of f(y)=y*log(y)
    def f_star(self, y):
        f_star_y = torch.exp(y-1)
        return f_star_y


'''
Pearson chi^2-divergence class (based on Legendre transform)
chi^2(P||Q), x~P, y~Q.
'''
class Pearson_chi_squared_LT(f_Divergence):
 
    # Legendre transform of f(y)=(y-1)^2
    def f_star(self, y):
        f_star_y = 0.25*torch.pow(y,2.0) + y
        return f_star_y


'''
squared Hellinger distance class (based on Legendre transform)
H(P||Q), x~P, y~Q.
'''
class squared_Hellinger_LT(f_Divergence):
 
    # Legendre transform of f(y)=(sqrt(y)-1)^2
    def f_star(self, y):
        f_star_y = y / (1-y)
        return f_star_y

    def final_layer_activation(self, y):
        out = 1.0 - torch.exp(-y)
        return out


'''
Jensen-Shannon divergence class (based on Legendre transform)
JS(P||Q), x~P, y~Q.
'''
class Jensen_Shannon_LT(f_Divergence):
 
    # Legendre transform of f(y)=y*log(y)-(y+1)*log((y+1)/2)
    def f_star(self, y):
#        max_val = tf.reduce_max(y)
        f_star_y = -torch.log(2.0 - torch.exp(y))
        return f_star_y

    def final_layer_activation(self, y):
        out = - torch.log(0.5 + 0.5*torch.exp(-y))
        return out


'''
alpha-divergence class (based on Legendre transform)
D_{f_alpha}(P||Q), x~P, y~Q.
'''
class alpha_Divergence_LT(f_Divergence):
 
    # initialize
    def __init__(self, discriminator, alpha, epochs, learning_rate, batch_size, discriminator_penalty=None):        
        Divergence.__init__(self, discriminator, epochs, learning_rate, batch_size, discriminator_penalty)
        self.alpha = alpha # order
    
    def get_order(self):
        return self.alpha

    def set_order(self, alpha):
        self.alpha = alpha
    
    # Legendre transform of f_alpha
    def f_star(self, y):
        if self.alpha>1.0:
            f_star_y = ((self.alpha-1.0)*F.relu(y))**(self.alpha/(self.alpha-1.0)) / self.alpha + 1.0/(self.alpha*(self.alpha-1.0))
        elif (self.alpha<1.0) & (self.alpha>0.0):
            f_star_y = torch.pow((1.0-self.alpha)*F.relu(y), self.alpha/(self.alpha-1.0)) / self.alpha - 1.0/(self.alpha*(self.alpha-1.0))
        
        return f_star_y


'''
Pearson chi^2-divergence class (based on Hammersley-Chapman-Robbins bound)
chi^2(P||Q), x~P, y~Q.
'''
class Pearson_chi_squared_HCR(Divergence):

    def eval_var_formula(self, x, y):
        D_P = self.discriminate(x)
        D_Q = self.discriminate(y)

        D_loss_P = torch.mean(D_P)
        D_loss_Q = torch.mean(D_Q)

        D_loss = (D_loss_P - D_loss_Q)**2 / torch.var(D_Q)
        return D_loss


'''
KL divergence class (based on the Donsker-Varahdan variational formula)
KL(P||Q), x~P, y~Q.
'''
class KLD_DV(Divergence):

    def eval_var_formula(self, x, y):
        D_P = self.discriminate(x)
        D_Q = self.discriminate(y)

        D_loss_P = torch.mean(D_P)
        
        max_val = torch.max(D_Q)
        D_loss_Q = torch.log(torch.mean(torch.exp(D_Q - max_val))) + max_val
        D_loss = D_loss_P - D_loss_Q
        return D_loss


'''
Renyi divergence class
R_alpha(P||Q), x~P, y~Q.
'''
class Renyi_Divergence(Divergence):
 
    # initialize
    def __init__(self, discriminator, alpha, epochs, learning_rate, batch_size, discriminator_penalty=None):        
        Divergence.__init__(self, discriminator, epochs, learning_rate, batch_size, discriminator_penalty)
        self.alpha = alpha # Renyi Divergence order

    def get_order(self):
        return self.alpha

    def set_order(self, alpha):
        self.alpha = alpha


'''
Renyi divergence class (based on the Renyi-Donsker-Varahdan variational formula)
R_alpha(P||Q), x~P, y~Q.
'''
class Renyi_Divergence_DV(Renyi_Divergence):

    def eval_var_formula(self, x, y):
        gamma = self.alpha
        beta = 1.0 - self.alpha

        D_P = self.discriminate(x)
        D_Q = self.discriminate(y)

        if beta == 0.0:
            D_loss_P = torch.mean(D_P)
        else:
            max_val = torch.max((-beta) * D_P)
            D_loss_P = -(1.0/beta) * (torch.log(torch.mean(torch.exp((-beta) * D_P - max_val))) + max_val)

        if gamma == 0.0:
            D_loss_Q = torch.mean(D_Q)
        else:
            max_val = torch.max((gamma) * D_Q)
            D_loss_Q = (1.0/gamma) * (torch.log(torch.mean(torch.exp(gamma * D_Q - max_val))) + max_val)

        D_loss = D_loss_P - D_loss_Q
        return D_loss


'''
Renyi divergence class (based on the convex-conjugate variational formula)
R_alpha(P||Q), x~P, y~Q.
'''
class Renyi_Divergence_CC(Renyi_Divergence):
    
    # initialize
    def __init__(self, discriminator, alpha, epochs, learning_rate, batch_size, fl_act_func, discriminator_penalty=None):        
        Renyi_Divergence.__init__(self, discriminator, alpha, epochs, learning_rate, batch_size, discriminator_penalty)
        self.act_func = fl_act_func
    
    def final_layer_activation(self, y): # enforce positive values
        if self.act_func=='abs':
            out = torch.abs(y)
        elif self.act_func=='softplus':
            out = F.softplus(y)
        elif self.act_func=='poly-softplus':
            out = 1.0+(1.0/(1.0+F.relu(-y))-1.0)*(1.0-torch.sign(y))/2.0 +y*(torch.sign(y)+1.0)/2.0
        
        return out
    
    def eval_var_formula(self, x, y):
        D_P = self.discriminate(x)
        D_P = self.final_layer_activation(D_P)
        D_Q = self.discriminate(y)
        D_Q = self.final_layer_activation(D_Q)
        
        D_loss_P = -torch.mean(D_P)
        D_loss_Q = torch.log(torch.mean(D_Q**((self.alpha-1.0)/self.alpha))) / (self.alpha-1.0)
        
        D_loss = D_loss_P + D_loss_Q + (np.log(self.alpha)+1.0)/self.alpha
        return D_loss


'''
Rescaled Renyi divergence class (based on the rescaled convex-conjugate variational formula)
alpha*R_alpha(P||Q), x~P, y~Q.
'''
class Renyi_Divergence_CC_rescaled(Renyi_Divergence_CC):
    
    def final_layer_activation(self, y): # enforce positive values
#        super(Renyi_Divergence_CC, self).final_layer_activation(self, y)
        out = Renyi_Divergence_CC.final_layer_activation(self, y)
        out = out/self.alpha
        
        return out

    def eval_var_formula(self, x, y):
#        super(Renyi_Divergence_CC, self).eval_var_formula(self, x, y)
        D_loss = Renyi_Divergence_CC.eval_var_formula(self, x, y)
        D_loss = D_loss * self.alpha
        
        return D_loss


'''
"Rescaled Renyi divergence class as alpha --> infinity (aka worst-case regret divergence)
D_\infty(P||Q), x~P, y~Q.
'''
class Renyi_Divergence_WCR(Renyi_Divergence_CC):
    
    def eval_var_formula(self, x, y):
        D_P = self.discriminate(x)
        D_P = self.final_layer_activation(D_P)
        D_Q = self.discriminate(y)
        D_Q = self.final_layer_activation(D_Q)
        
        D_loss_P = torch.log(torch.mean(D_P))
        D_loss_Q = -torch.mean(D_Q)
        
        D_loss = D_loss_P + D_loss_Q + 1.0
        return D_loss
    

'''
Discriminator penalty class penalizes the divergence objective functional during training.
Allows for the (approximate) implementation of discriminator constraints.
'''
class Discriminator_Penalty():
    # initialize
    def __init__(self, penalty_weight):
        self.penalty_weight = penalty_weight # weighting of penalty term
        
    def get_penalty_weight(self):
        return self.penalty_weight

    def set_penalty_weight(self, weight):
        self.penalty_weight = weight
    
    def evaluate(self, discriminator, x, y): # depends on the choice of penalty 
        return None


'''
One-sided gradient penalty (constraint: Lipschitz constant  <= Lip_const)
'''
class Gradient_Penalty_1Sided(Discriminator_Penalty):
    # initialize
    def __init__(self, penalty_weight, Lip_const):
        
        Discriminator_Penalty.__init__(self, penalty_weight)
        self.Lip_const = Lip_const # Lipschitz constant


    def get_Lip_constant(self):
        return self.Lip_const

    def set_Lip_constant(self, L):
        self.Lip_const = L
    
    def evaluate(self, discriminator, x, y): # compute the gradient penalty
        temp_shape = [x.shape[0]] + [1 for _ in  range(len(x.shape)-1)]
        ratio = torch.rand(temp_shape, dtype=torch.float32, requires_grad=True)
        diff = y - x
        interpltd = x + (ratio * diff) # get the interpolated samples

        # interpltd = torch.autograd.Variable(interpltd, requires_grad=True) 
        D_pred = discriminator(interpltd)

        grads = torch_grad(outputs=D_pred, inputs=interpltd, grad_outputs=torch.ones(D_pred.size()), create_graph=True, retain_graph=True)[0]
        if x.shape[1]==1: # calculate the norm
            norm = torch.sqrt(torch.square(grads))
        else:
            norm = torch.sqrt(torch.sum(torch.square(grads)))

        gp = self.penalty_weight * torch.mean(torch.maximum(norm - self.Lip_const, torch.tensor(0.0))) # one-sided gradient penalty
        return gp


'''
Two-sided gradient penalty (constraint: Lipschitz constant = Lip_const)
'''
class Gradient_Penalty_2Sided(Discriminator_Penalty):
    # initialize
    def __init__(self, penalty_weight, Lip_const):
        
        Discriminator_Penalty.__init__(self, penalty_weight)
        self.Lip_const = Lip_const # Lipschitz constant


    def get_Lip_constant(self):
        return self.Lip_const

    def set_Lip_constant(self, L):
        self.Lip_const = L
    

    def evaluate(self, discriminator, x, y): # compute the gradient penalty
        temp_shape = [x.shape[0]] + [1 for _ in  range(len(x.shape)-1)]
        ratio = torch.rand(temp_shape, dtype=torch.float32)
        diff = y - x
        interpltd = x + (ratio * diff) # get the interpolated samples

        D_pred = discriminator(interpltd)

        grads = torch_grad(outputs=D_pred, inputs=interpltd, grad_outputs=torch.ones(interpltd.size()), create_graph=True, retain_graph=True)[0]
        if x.shape[1]==1: # calculate the norm
            norm = torch.sqrt(torch.square(grads))
        else:
            norm = torch.sqrt(torch.sum(torch.square(grads)))

        gp = self.penalty_weight*torch.mean((norm - self.Lip_const) ** 2) # two-sided gradient penalty
        return gp