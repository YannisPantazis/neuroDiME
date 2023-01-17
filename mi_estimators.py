import numpy as np
import math

import torch 
import torch.nn as nn


class CLUB(nn.Module):  # CLUB: Mutual Information Contrastive Learning Upper Bound
    '''
        This class provides the CLUB estimation to I(X,Y)
        Method:
            mi_est() :      provides the estimation with input samples  
            loglikeli() :   provides the log-likelihood of the approximation q(Y|X) with input samples
        Arguments:
            x_dim, y_dim :         the dimensions of samples from X, Y respectively
            hidden_size :          the dimension of the hidden layer of the approximation network q(Y|X)
            x_samples, y_samples : samples from X and Y, having shape [sample_size, x_dim/y_dim] 
    '''
    def __init__(self, x_dim, y_dim, hidden_size, var_flag):
        super(CLUB, self).__init__()
        # p_mu outputs mean of q(Y|X)
        self.p_mu = nn.Sequential(nn.Linear(x_dim, hidden_size//2),
                                       nn.ReLU(),
                                       nn.Linear(hidden_size//2, y_dim))
        # p_logvar outputs log of variance of q(Y|X)
        self.p_logvar = nn.Sequential(nn.Linear(x_dim, hidden_size//2),
                                       nn.ReLU(),
                                       nn.Linear(hidden_size//2, y_dim),
                                       nn.Tanh())

        self.var_flag = var_flag

    def get_mu_logvar(self, x_samples):
        mu = self.p_mu(x_samples)
        logvar = self.p_logvar(x_samples)
        return mu, logvar
    
    def mi_est(self, x_samples, y_samples, lambda_VR): 
        mu, logvar = self.get_mu_logvar(x_samples)
        
        # log of conditional     probability of positive sample pairs
        positive = - (mu - y_samples)**2 /2./logvar.exp()
        positive = positive.sum(dim = -1)
        
        prediction_1 = mu.unsqueeze(1)          # shape [nsample,1,dim]
        y_samples_1 = y_samples.unsqueeze(0)    # shape [1,nsample,dim]

        # log of conditional probability of negative sample pairs
        negative = - ((y_samples_1 - prediction_1)**2).mean(dim=1)/2./logvar.exp() 
        negative = negative.sum(dim = -1)

        bound = (positive - negative).mean()

        if not self.var_flag:
            loss = self.loglikeli(x_samples, y_samples, 0)
        else:
            loss = self.loglikeli(x_samples, y_samples, lambda_VR)
      
        return bound, loss

    def loglikeli(self, x_samples, y_samples, lambda_VR): # unnormalized loglikelihood 
        mu, logvar = self.get_mu_logvar(x_samples) 
        term1 = (-(mu - y_samples)**2 /logvar.exp()).sum(dim=1)
        term2 = - logvar.sum(dim=1)
        VR_penalty = ((term1 - term1.mean())**2 + (term2 - term2.mean())**2).mean()
        return term1.mean() + term2.mean() - lambda_VR * VR_penalty

    def learning_loss(self, x_samples, y_samples):
        return self.loglikeli(x_samples, y_samples)
    

# class CLUB_ours(nn.Module):  # CLUB: Mutual Information Contrastive Learning Upper Bound
#     '''
#         This class provides the CLUB estimation to I(X,Y)
#         Method:
#             mi_est() :      provides the estimation with input samples  
#             loglikeli() :   provides the log-likelihood of the approximation q(Y|X) with input samples
#         Arguments:
#             x_dim, y_dim :         the dimensions of samples from X, Y respectively
#             hidden_size :          the dimension of the hidden layer of the approximation network q(Y|X)
#             x_samples, y_samples : samples from X and Y, having shape [sample_size, x_dim/y_dim] 
#     '''
#     def __init__(self, x_dim, y_dim, hidden_size, var_flag):
#         super(CLUB_ours, self).__init__()
#         # p_mu outputs mean of q(Y|X)
#         self.p_mu = nn.Sequential(nn.Linear(x_dim, hidden_size),
#                                     nn.Tanh(),
#                                     nn.Linear(hidden_size, hidden_size),
#                                     nn.Tanh(),
#                                     nn.Linear(hidden_size, hidden_size),
#                                     nn.Tanh(),
#                                     nn.Linear(hidden_size, hidden_size//2),
#                                     nn.Tanh(),
#                                     nn.Linear(hidden_size//2, y_dim))
#         # p_logvar outputs log of variance of q(Y|X)
#         self.p_logvar = nn.Sequential(nn.Linear(x_dim, hidden_size),
#                                     nn.Tanh(),
#                                     nn.Linear(hidden_size, hidden_size),
#                                     nn.Tanh(),
#                                     nn.Linear(hidden_size, hidden_size),
#                                     nn.Tanh(),
#                                     nn.Linear(hidden_size, hidden_size//2),
#                                     nn.Tanh(),
#                                     nn.Linear(hidden_size//2, y_dim),
#                                     nn.Tanh())

#         self.var_flag = var_flag

#     def get_mu_logvar(self, x_samples):
#         mu = self.p_mu(x_samples)
#         logvar = self.p_logvar(x_samples)
#         return mu, logvar
    
#     def mi_est(self, x_samples, y_samples, lambda_VR): 
#         mu, logvar = self.get_mu_logvar(x_samples)
        
#         # log of conditional     probability of positive sample pairs
#         positive = - (mu - y_samples)**2 /2./logvar.exp()
#         positive = positive.sum(dim = -1)
        
#         prediction_1 = mu.unsqueeze(1)          # shape [nsample,1,dim]
#         y_samples_1 = y_samples.unsqueeze(0)    # shape [1,nsample,dim]

#         # log of conditional probability of negative sample pairs
#         negative = - ((y_samples_1 - prediction_1)**2).mean(dim=1)/2./logvar.exp() 
#         negative = negative.sum(dim = -1)

#         bound = (positive - negative).mean()

#         if not self.var_flag:
#             loss = self.loglikeli(x_samples, y_samples, 0)
#         else:
#             loss = self.loglikeli(x_samples, y_samples, lambda_VR)
      
#         return bound, loss

#     def loglikeli(self, x_samples, y_samples, lambda_VR): # unnormalized loglikelihood 
#         mu, logvar = self.get_mu_logvar(x_samples) 
#         term1 = (-(mu - y_samples)**2 /logvar.exp()).sum(dim=1)
#         term2 = - logvar.sum(dim=1)
#         VR_penalty = ((term1 - term1.mean())**2 + (term2 - term2.mean())**2).mean()
#         return term1.mean() + term2.mean() - lambda_VR * VR_penalty

#     def learning_loss(self, x_samples, y_samples):
#         return self.loglikeli(x_samples, y_samples)

    
class CLUBSample(nn.Module):  # Sampled version of the CLUB estimator
    def __init__(self, x_dim, y_dim, hidden_size, var_flag):
        super(CLUBSample, self).__init__()
        self.p_mu = nn.Sequential(nn.Linear(x_dim, hidden_size//2),
                                       nn.ReLU(),
                                       nn.Linear(hidden_size//2, y_dim))

        self.p_logvar = nn.Sequential(nn.Linear(x_dim, hidden_size//2),
                                       nn.ReLU(),
                                       nn.Linear(hidden_size//2, y_dim),
                                       nn.Tanh())

        self.var_flag = var_flag

    def get_mu_logvar(self, x_samples):
        mu = self.p_mu(x_samples)
        logvar = self.p_logvar(x_samples)
        return mu, logvar
     
        
    def loglikeli(self, x_samples, y_samples, lambda_VR): # unnormalized loglikelihood 
        mu, logvar = self.get_mu_logvar(x_samples) 
        term1 = (-(mu - y_samples)**2 /logvar.exp()).sum(dim=1)
        term2 = - logvar.sum(dim=1)
        VR_penalty = ((term1 - term1.mean())**2 + (term2 - term2.mean())**2).mean()
        return term1.mean() + term2.mean() - lambda_VR * VR_penalty
    

    def mi_est(self, x_samples, y_samples, lambda_VR):
        mu, logvar = self.get_mu_logvar(x_samples)
        
        sample_size = x_samples.shape[0]
        #random_index = torch.randint(sample_size, (sample_size,)).long()
        random_index = torch.randperm(sample_size).long()
        
        positive = - (mu - y_samples)**2 / logvar.exp()
        positive = positive.sum(dim = -1)

        negative = - (mu - y_samples[random_index])**2 / logvar.exp()
        negative = negative.sum(dim = -1)

        bound = (positive - negative).mean() / 2

        if not self.var_flag:

            loss = self.loglikeli(x_samples, y_samples, 0)
        else:
            loss = self.loglikeli(x_samples, y_samples, lambda_VR)
      
        return bound, loss


# class NWJ(nn.Module):   
#     def __init__(self, x_dim, y_dim, hidden_size, var_flag):
#         super(NWJ, self).__init__()
#         self.F_func = nn.Sequential(nn.Linear(x_dim + y_dim, hidden_size),
#                                     nn.ReLU(),
#                                     nn.Linear(hidden_size, 1))

#         self.var_flag = var_flag
                                    
#     # def mi_est(self, x_samples, y_samples, a): 
#     #     # shuffle and concatenate
#     #     sample_size = y_samples.shape[0]

#     #     x_tile = x_samples.unsqueeze(0).repeat((sample_size, 1, 1))
#     #     y_tile = y_samples.unsqueeze(1).repeat((1, sample_size, 1))

#     #     T0 = self.F_func(torch.cat([x_samples,y_samples], dim = -1))
#     #     T1 = self.F_func(torch.cat([x_tile, y_tile], dim = -1))-1.  #shape [sample_size, sample_size, 1]
#     #     import pdb;pdb.set_trace()

#     #     lower_bound = T0.mean() - (T1.logsumexp(dim = 1) - np.log(sample_size)).exp().mean() 
#     #     return lower_bound


 
#     def mi_est(self, x_samples, y_samples, lambda_VR): 
#         # shuffle and concatenate
#         sample_size = y_samples.shape[0]

#         random_index = torch.randint(sample_size, (sample_size,)).long()

#         y_shuffle = y_samples[random_index]


#         T0 = self.F_func(torch.cat([x_samples,y_samples], dim = -1))   # [64, 40], [64 1]

#         T1 = self.F_func(torch.cat([x_samples,y_shuffle], dim = -1))  #shape [sample_size, sample_size, 1]

#         term1 = T0.mean()
#         term2 = (T1 - 1).exp().mean()

#         lower_bound = term1 - term2

#         if not self.var_flag:
#             loss = lower_bound
#         else:
#             T1_exp = (T1 - 1).exp()
#             m1_term = term2
#             VR_penalty = ((T0 - term1)**2).mean() + (((T1_exp - m1_term)**2).mean())
#             loss = lower_bound - lambda_VR * VR_penalty

#         return lower_bound, loss

    
class NWJ(nn.Module):   
    def __init__(self, x_dim, y_dim, hidden_size, var_flag):
        super(NWJ, self).__init__()
        self.F_func = nn.Sequential(nn.Linear(x_dim + y_dim, hidden_size),
                                    nn.ReLU(),
                                    nn.Linear(hidden_size, 1))

        self.var_flag = var_flag
                                    
    def mi_est(self, x_samples, y_samples, lambda_VR): 
        # shuffle and concatenate
        sample_size = y_samples.shape[0]

        x_tile = x_samples.unsqueeze(0).repeat((sample_size, 1, 1))
        y_tile = y_samples.unsqueeze(1).repeat((1, sample_size, 1))

        T0 = self.F_func(torch.cat([x_samples,y_samples], dim = -1))
        T1 = self.F_func(torch.cat([x_tile, y_tile], dim = -1))-1.  #shape [sample_size, sample_size, 1]


        term1 = T0
        term2 = (T1.logsumexp(dim = 1) - np.log(sample_size)).exp()

        lower_bound = term1.mean() - term2.mean()

        if not self.var_flag:
            loss = lower_bound
        else:       
            VR_penalty = ((term1 - term1.mean())**2 + (term2 - term2.mean())**2).mean()
            loss = lower_bound - lambda_VR * VR_penalty 

        return lower_bound, loss


class InfoNCE(nn.Module):
    def __init__(self, x_dim, y_dim, hidden_size, var_flag):
        super(InfoNCE, self).__init__()
        self.F_func = nn.Sequential(nn.Linear(x_dim + y_dim, hidden_size),
                                    nn.ReLU(),
                                    nn.Linear(hidden_size, 1),
                                    nn.Softplus())
        self.var_flag = var_flag
    
    def mi_est(self, x_samples, y_samples, lambda_VR):  # samples have shape [sample_size, dim]
        # shuffle and concatenate
        sample_size = y_samples.shape[0]

        x_tile = x_samples.unsqueeze(0).repeat((sample_size, 1, 1))
        y_tile = y_samples.unsqueeze(1).repeat((1, sample_size, 1))

        T0 = self.F_func(torch.cat([x_samples,y_samples], dim = -1))
        T1 = self.F_func(torch.cat([x_tile, y_tile], dim = -1))  #[sample_size, sample_size, 1]

        term1 = T0.mean()
        term2 = (T1.logsumexp(dim = 1).mean() - np.log(sample_size))      

        lower_bound = term1 - term2

        if not self.var_flag:
            loss = lower_bound
        else:       
            VR_penalty = ((term1 - term1.mean())**2 + (term2 - term2.mean())**2).mean()
            loss = lower_bound - lambda_VR * VR_penalty 

        return lower_bound, loss


    # def mi_est(self, x_samples, y_samples, lambda_VR): 
    #     # shuffle and concatenate
    #     sample_size = y_samples.shape[0]

    #     random_index = torch.randint(sample_size, (sample_size,)).long()

    #     y_shuffle = y_samples[random_index]


    #     T0 = self.F_func(torch.cat([x_samples,y_samples], dim = -1))   # [64, 40], [64 1]
    #     T1 = self.F_func(torch.cat([x_samples,y_shuffle], dim = -1))  #shape [sample_size, sample_size, 1]

    #     term1 = T0.mean()
    #     term2 = torch.log(T1.exp().mean())

    #     lower_bound = term1 - term2

    #     if not self.var_flag:
    #         loss = lower_bound
    #     else:
    #         T1_exp = T1.exp()
    #         m1_term = T1_exp.mean()
    #         VR_penalty = ((T0 - T0.mean())**2).mean() + (((T1_exp - m1_term)**2).mean())/ m1_term **2
    #         loss = lower_bound - lambda_VR * VR_penalty

    #     return lower_bound, loss

class MINE(nn.Module):
    def __init__(self, x_dim, y_dim, hidden_size):
        super(MINE, self).__init__()
        self.T_func = nn.Sequential(nn.Linear(x_dim + y_dim, hidden_size),
                                    nn.ReLU(),
                                    nn.Linear(hidden_size, 1))
    
    def mi_est(self, x_samples, y_samples):  # samples have shape [sample_size, dim]
        # shuffle and concatenate
        sample_size = y_samples.shape[0]
        random_index = torch.randint(sample_size, (sample_size,)).long()

        y_shuffle = y_samples[random_index]

        T0 = self.T_func(torch.cat([x_samples,y_samples], dim = -1))
        T1 = self.T_func(torch.cat([x_samples,y_shuffle], dim = -1))

        lower_bound = T0.mean() - torch.log(T1.exp().mean())
        
        # compute the negative loss (maximise loss == minimise -loss)
        return lower_bound




def log_sum_exp(value, dim=None, keepdim=False):
    """Numerically stable implementation of the operation
    value.exp().sum(dim, keepdim).log()
    """
    # TODO: torch.max(value, dim=None) threw an error at time of writing
    if dim is not None:
        m, _ = torch.max(value, dim=dim, keepdim=True)
        value0 = value - m
        if keepdim is False:
            m = m.squeeze(dim)
        return m + torch.log(torch.sum(torch.exp(value0),
                                       dim=dim, keepdim=keepdim))
    else:
        m = torch.max(value)
        sum_exp = torch.sum(torch.exp(value - m))
        if isinstance(sum_exp, Number):
            return m + math.log(sum_exp)
        else:
            return m + torch.log(sum_exp)


class L1OutUB(nn.Module):  # naive upper bound
    def __init__(self, x_dim, y_dim, hidden_size, var_flag):
        super(L1OutUB, self).__init__()
        self.p_mu = nn.Sequential(nn.Linear(x_dim, hidden_size//2),
                                       nn.ReLU(),
                                       nn.Linear(hidden_size//2, y_dim))

        self.p_logvar = nn.Sequential(nn.Linear(x_dim, hidden_size//2),
                                       nn.ReLU(),
                                       nn.Linear(hidden_size//2, y_dim),
                                       nn.Tanh())
        self.var_flag = var_flag

    def get_mu_logvar(self, x_samples):
        mu = self.p_mu(x_samples)
        logvar = self.p_logvar(x_samples)
        return mu, logvar

    def mi_est(self, x_samples, y_samples, lambda_VR): 
        batch_size = y_samples.shape[0]
        mu, logvar = self.get_mu_logvar(x_samples)              # torch.Size([nsample, 20])

        positive = (- (mu - y_samples)**2 /2./logvar.exp() - logvar/2.).sum(dim = -1) #[nsample]

        mu_1 = mu.unsqueeze(1)          # [nsample,1,dim]
        logvar_1 = logvar.unsqueeze(1)
        y_samples_1 = y_samples.unsqueeze(0)            # [1,nsample,dim]
        all_probs =  (- (y_samples_1 - mu_1)**2/2./logvar_1.exp()- logvar_1/2.).sum(dim = -1)  #[nsample, nsample]

        diag_mask =  torch.ones([batch_size]).diag().unsqueeze(-1).cuda() * (-20.)
        negative = log_sum_exp(all_probs + diag_mask,dim=0) - np.log(batch_size-1.) #[nsample]

        bound = (positive - negative).mean()

        if not self.var_flag:
            loss = self.loglikeli(x_samples, y_samples, 0)
        else:
            loss = self.loglikeli(x_samples, y_samples, lambda_VR)
      
        return bound, loss
        

    def loglikeli(self, x_samples, y_samples, lambda_VR): # unnormalized loglikelihood 
        mu, logvar = self.get_mu_logvar(x_samples) 
        term1 = (-(mu - y_samples)**2 /logvar.exp()).sum(dim=1)
        term2 = - logvar.sum(dim=1)
        VR_penalty = ((term1 - term1.mean())**2 + (term2 - term2.mean())**2).mean()
        return term1.mean() + term2.mean() - lambda_VR * VR_penalty

class VarUB(nn.Module):  #    variational upper bound
    def __init__(self, x_dim, y_dim, hidden_size):
        super(VarUB, self).__init__()
        self.p_mu = nn.Sequential(nn.Linear(x_dim, hidden_size//2),
                                       nn.ReLU(),
                                       nn.Linear(hidden_size//2, y_dim))

        self.p_logvar = nn.Sequential(nn.Linear(x_dim, hidden_size//2),
                                       nn.ReLU(),
                                       nn.Linear(hidden_size//2, y_dim),
                                       nn.Tanh())

    def get_mu_logvar(self, x_samples):
        mu = self.p_mu(x_samples)
        logvar = self.p_logvar(x_samples)
        return mu, logvar
            
    def mi_est(self, x_samples, y_samples): #[nsample, 1]
        mu, logvar = self.get_mu_logvar(x_samples)
        return 1./2.*(mu**2 + logvar.exp() - 1. - logvar).mean()
        
    def loglikeli(self, x_samples, y_samples):
        mu, logvar = self.get_mu_logvar(x_samples)
        return (-(mu - y_samples)**2 /logvar.exp()-logvar).sum(dim=1).mean(dim=0)



# class MINE_VAR(nn.Module):
#     def __init__(self, x_dim, y_dim, hidden_size):
#         super(MINE_VAR, self).__init__()
#         self.T_func = nn.Sequential(nn.Linear(x_dim + y_dim, hidden_size),
#                                     nn.ReLU(),
#                                     nn.Linear(hidden_size, 1))
#         self.T_func_ext = nn.Sequential(nn.Linear(1, hidden_size),
#                                     nn.ReLU(),
#                                     nn.Linear(hidden_size, 1))

#     def var_cal(self, T):
#         weights = torch.Tensor(size_out, size_in)
#         self.weights = nn.Parameter(weights)
        
    
#     def mi_est(self, x_samples, y_samples):  # samples have shape [sample_size, dim]
#         # shuffle and concatenate
#         sample_size = y_samples.shape[0]
#         random_index = torch.randint(sample_size, (sample_size,)).long()

#         y_shuffle = y_samples[random_index]

#         T0 = self.T_func(torch.cat([x_samples,y_samples], dim = -1))
#         T0_external = self.T_func_ext(T0)
#         T1 = self.T_func(torch.cat([x_samples,y_shuffle], dim = -1))
#         T1_external = self.T_func_ext(T1)
#         lower_bound = T0_external.mean() - torch.log(T1_external.exp().mean())

#         # compute the negative loss (maximise loss == minimise -loss)
#         return lower_bound    



# class R_MINE_VAR(nn.Module):
#     def __init__(self, x_dim, y_dim, hidden_size, beta_values, var_flag):
#         super(R_MINE_VAR, self).__init__()
#         self.T_func = nn.Sequential(nn.Linear(x_dim + y_dim, hidden_size),
#                                     nn.ReLU(),
#                                     nn.Linear(hidden_size, 1))
#         self.ext_T_func = nn.Sequential(nn.Linear(1, hidden_size),
#                                     nn.ReLU(),
#                                     nn.Linear(hidden_size, 1))
#         self.beta = beta_values
#         self.var_flag = var_flag

#     def measure_mi(self, joint, marginal, lambda_VR):
#         loss = 0
#         # Reyni Divengence Disentanglement information

#         for i, beta_val in enumerate(self.beta):
#             term1, term2 = 0.0, 0.0
#             if beta_val == 0:
#                 max_val = torch.max((1 - beta_val) * marginal)

#                 term1 = torch.mean(joint)

#                 term2_exp = torch.mean(torch.exp((1 - beta_val) * marginal - max_val))
#                 term2 = (1 / (1 - beta_val)) * (torch.log(term2_exp) + max_val)

#                 S_term1 = torch.mean(joint**2) - term1**2
#                 m_term1 = term1

#                 m_term2 = term2_exp
#                 S_term2 = torch.mean((torch.exp((1 - beta_val) * marginal - max_val)) ** 2) - (m_term2) ** 2


#             elif beta_val == 1:
#                 max_val = torch.max(- beta_val * joint)

#                 term1_exp = torch.mean(torch.exp(-beta_val * joint - max_val))
#                 term1 = -(1 / beta_val) * (torch.log(term1_exp) + max_val)

#                 term2 = torch.mean(marginal)

#                 m_term1 = term1_exp
#                 S_term1 = torch.mean((torch.exp((-beta_val) * joint - max_val)) ** 2) - (m_term1) ** 2

#                 S_term2 = torch.mean(marginal**2) - term2**2
#                 m_term2 = term2

#             else:
#                 max_val_1 = torch.max(- beta_val * joint)
#                 max_val_2 = torch.max((1 - beta_val) * marginal)

#                 term1_exp = torch.mean(torch.exp(-beta_val * joint - max_val_1))
#                 term1 = -(1 / beta_val) * (torch.log(term1_exp) + max_val_1)

#                 term2_exp = torch.mean(torch.exp((1 - beta_val) * marginal - max_val_2))
#                 term2 = (1 / (1 - beta_val)) * (torch.log(term2_exp) + max_val_2)

#                 m_term1 = term1_exp
#                 S_term1 = torch.mean((torch.exp((-beta_val) * joint - max_val_1)) ** 2) - (m_term1) ** 2

#                 m_term2 = term2_exp
#                 S_term2 = torch.mean((torch.exp((1 - beta_val) * marginal - max_val_2)) ** 2) - (m_term2) ** 2

#             if not self.var_flag:
#                 loss += (term1 - term2)

#             else:

#                 if beta_val == 0:
#                     VR_penalty = S_term1 + (1.0 / (1 - beta_val)**2) * (S_term2/ m_term2**2)
#                 elif beta_val == 1:
#                     VR_penalty = (1.0 / (-beta_val) ** 2) * torch.div(S_term1, m_term1 ** 2) + S_term2
#                 else:
#                     VR_penalty = (1.0 / (-beta_val)**2) * (S_term1 / m_term1 ** 2) + (1 / (1 - beta_val)**2) * (S_term2 / m_term2 ** 2)

#                 loss += (term1 - term2) - lambda_VR * VR_penalty

#         return loss

#     def mi_est(self, x_samples, y_samples, lambda_VR):  # samples have shape [sample_size, dim]
#         # shuffle and concatenate
#         sample_size = y_samples.shape[0]
#         random_index = torch.randint(sample_size, (sample_size,)).long()

#         y_shuffle = y_samples[random_index]

#         T0 = self.T_func(torch.cat([x_samples,y_samples], dim = -1))
#         T0_external = self.ext_T_func(T0) + T0
#         T1 = self.T_func(torch.cat([x_samples,y_shuffle], dim = -1))
#         T1_external = self.ext_T_func(T1) + T1
#         lower_bound = self.measure_mi(T0_external, T1_external, lambda_VR)

#         # compute the negative loss (maximise loss == minimise -loss)
#         return lower_bound


# class R_MINE_VAR_v2(nn.Module):
#     def __init__(self, x_dim, y_dim, hidden_size, beta_values, var_flag):
#         super(R_MINE_VAR_v2, self).__init__()
#         self.mine = nn.Sequential(nn.Linear(x_dim + y_dim, hidden_size),
#                                     nn.ReLU(),
#                                     nn.Linear(hidden_size, 1))
#         self.ext = nn.Sequential(nn.Linear(1, hidden_size),
#                                     nn.ReLU(),
#                                     nn.Linear(hidden_size, 1))
#         self.beta = beta_values
#         self.var_flag = var_flag

#     def measure_mi(self, joint, marginal, lambda_VR):
#         ext_loss, loss = 0, 0
#         # Reyni Divengence Disentanglement information

#         for i, beta_val in enumerate(self.beta):
#             term1, term2 = 0.0, 0.0
#             if beta_val == 0:
#                 max_val = torch.max((1 - beta_val) * marginal)

#                 term1 = torch.mean(joint)

#                 term2_exp = torch.mean(torch.exp((1 - beta_val) * marginal - max_val))
#                 term2 = (1 / (1 - beta_val)) * (torch.log(term2_exp) + max_val)

#                 S_term1 = torch.mean(joint**2) - term1**2
#                 m_term1 = term1

#                 m_term2 = term2_exp
#                 S_term2 = torch.mean((torch.exp((1 - beta_val) * marginal - max_val)) ** 2) - (m_term2) ** 2


#             elif beta_val == 1:
#                 max_val = torch.max(- beta_val * joint)

#                 term1_exp = torch.mean(torch.exp(-beta_val * joint - max_val))
#                 term1 = -(1 / beta_val) * (torch.log(term1_exp) + max_val)

#                 term2 = torch.mean(marginal)

#                 m_term1 = term1_exp
#                 S_term1 = torch.mean((torch.exp((-beta_val) * joint - max_val)) ** 2) - (m_term1) ** 2

#                 S_term2 = torch.mean(marginal**2) - term2**2
#                 m_term2 = term2

#             else:
#                 max_val_1 = torch.max(- beta_val * joint)
#                 max_val_2 = torch.max((1 - beta_val) * marginal)

#                 term1_exp = torch.mean(torch.exp(-beta_val * joint - max_val_1))
#                 term1 = -(1 / beta_val) * (torch.log(term1_exp) + max_val_1)

#                 term2_exp = torch.mean(torch.exp((1 - beta_val) * marginal - max_val_2))
#                 term2 = (1 / (1 - beta_val)) * (torch.log(term2_exp) + max_val_2)

#                 m_term1 = term1_exp
#                 S_term1 = torch.mean((torch.exp((-beta_val) * joint - max_val_1)) ** 2) - (m_term1) ** 2

#                 m_term2 = term2_exp
#                 S_term2 = torch.mean((torch.exp((1 - beta_val) * marginal - max_val_2)) ** 2) - (m_term2) ** 2

#             if not self.var_flag:
#                 loss += (term1 - term2)

#             else:

#                 if beta_val == 0:
#                     VR_penalty = S_term1 + (1.0 / (1 - beta_val)**2) * (S_term2/ m_term2**2)
#                 elif beta_val == 1:
#                     VR_penalty = (1.0 / (-beta_val) ** 2) * torch.div(S_term1, m_term1 ** 2) + S_term2
#                 else:
#                     VR_penalty = (1.0 / (-beta_val)**2) * (S_term1 / m_term1 ** 2) + (1 / (1 - beta_val)**2) * (S_term2 / m_term2 ** 2)

#                 ext_loss += (term1 - term2)

#                 loss += ext_loss - lambda_VR * VR_penalty

#         return ext_loss, loss

#     def mi_est(self, x_samples, y_samples, lambda_VR):  # samples have shape [sample_size, dim]
#         # shuffle and concatenate
#         sample_size = y_samples.shape[0]
#         random_index = torch.randint(sample_size, (sample_size,)).long()

#         y_shuffle = y_samples[random_index]

#         T0 = self.mine(torch.cat([x_samples,y_samples], dim = -1))
#         T0_external = self.ext(T0) + T0
#         T1 = self.mine(torch.cat([x_samples,y_shuffle], dim = -1))
#         T1_external = self.ext(T1) + T1
#         lower_bound, loss = self.measure_mi(T0_external, T1_external, lambda_VR)

#         # compute the negative loss (maximise loss == minimise -loss)
#         return lower_bound, loss

class R_MINE_VAR_v1(nn.Module):
    def __init__(self, x_dim, y_dim, hidden_size, hidden_size_ext,  beta_values, var_flag):
        super(R_MINE_VAR_v1, self).__init__()
        self.mine = nn.Sequential(nn.Linear(x_dim + y_dim, hidden_size),
                                    nn.Tanh(),
                                    nn.Linear(hidden_size, hidden_size),
                                    nn.Tanh(),
                                    nn.Linear(hidden_size, hidden_size),
                                    nn.Tanh(),
                                    nn.Linear(hidden_size, hidden_size//2),
                                    nn.Tanh(),
                                    nn.Linear(hidden_size//2, 1))

        self.beta = beta_values
        self.var_flag = var_flag

    def measure_mi(self, joint, marginal, lambda_VR):
        ext_loss, loss = 0, 0
        # Reyni Divengence Disentanglement information

        for i, beta_val in enumerate(self.beta):
            term1, term2 = 0.0, 0.0
            if beta_val == 0:
                max_val = torch.max((1 - beta_val) * marginal)

                term1 = torch.mean(joint)

                term2_exp = torch.mean(torch.exp((1 - beta_val) * marginal - max_val))
                term2 = (1 / (1 - beta_val)) * (torch.log(term2_exp) + max_val)

                S_term1 = torch.mean(joint**2) - term1**2
                m_term1 = term1

                m_term2 = term2_exp
                S_term2 = torch.mean((torch.exp((1 - beta_val) * marginal - max_val)) ** 2) - (m_term2) ** 2


            elif beta_val == 1:
                max_val = torch.max(- beta_val * joint)

                term1_exp = torch.mean(torch.exp(-beta_val * joint - max_val))
                term1 = -(1 / beta_val) * (torch.log(term1_exp) + max_val)

                term2 = torch.mean(marginal)

                m_term1 = term1_exp
                S_term1 = torch.mean((torch.exp((-beta_val) * joint - max_val)) ** 2) - (m_term1) ** 2

                S_term2 = torch.mean(marginal**2) - term2**2
                m_term2 = term2

            else:
                max_val_1 = torch.max(- beta_val * joint)
                max_val_2 = torch.max((1 - beta_val) * marginal)

                term1_exp = torch.mean(torch.exp(-beta_val * joint - max_val_1))
                term1 = -(1 / beta_val) * (torch.log(term1_exp) + max_val_1)

                term2_exp = torch.mean(torch.exp((1 - beta_val) * marginal - max_val_2))
                term2 = (1 / (1 - beta_val)) * (torch.log(term2_exp) + max_val_2)

                m_term1 = term1_exp
                S_term1 = torch.mean((torch.exp((-beta_val) * joint - max_val_1)) ** 2) - (m_term1) ** 2

                m_term2 = term2_exp
                S_term2 = torch.mean((torch.exp((1 - beta_val) * marginal - max_val_2)) ** 2) - (m_term2) ** 2

            if not self.var_flag:
                loss += (term1 - term2)


        return ext_loss, loss

    def mi_est(self, x_samples, y_samples, lambda_VR):  # samples have shape [sample_size, dim]
        # shuffle and concatenate
        sample_size = y_samples.shape[0]
        random_index = torch.randint(sample_size, (sample_size,)).long()

        y_shuffle = y_samples[random_index]

        T0 = 50. * torch.tanh(self.mine(torch.cat([x_samples,y_samples], dim = -1))/50.)
        T1 = 50. * torch.tanh(self.mine(torch.cat([x_samples,y_shuffle], dim = -1))/50.)
        lower_bound, loss = self.measure_mi(T0, T1, lambda_VR)

        # compute the negative loss (maximise loss == minimise -loss)
        return lower_bound, loss



class R_MINE_VAR_v2(nn.Module):
    def __init__(self, x_dim, y_dim, hidden_size, hidden_size_ext,  beta_values, var_flag):
        super(R_MINE_VAR_v2, self).__init__()
        self.mine = nn.Sequential(nn.Linear(x_dim + y_dim, hidden_size),
                                    nn.Tanh(),
                                    nn.Linear(hidden_size, hidden_size),
                                    nn.Tanh(),
                                    nn.Linear(hidden_size, hidden_size),
                                    nn.Tanh(),
                                    nn.Linear(hidden_size, hidden_size//2),
                                    nn.Tanh(),
                                    nn.Linear(hidden_size//2, 1))

        self.ext = nn.Sequential(nn.Linear(1, hidden_size_ext),
                                    nn.Tanh(),
                                    nn.Linear(hidden_size_ext, 1))
        self.beta = beta_values
        self.var_flag = var_flag

    def measure_mi(self, joint, marginal, lambda_VR):
        ext_loss, loss = 0, 0
        # Reyni Divengence Disentanglement information

        for i, beta_val in enumerate(self.beta):
            term1, term2 = 0.0, 0.0
            if beta_val == 0:
                max_val = torch.max((1 - beta_val) * marginal)

                term1 = torch.mean(joint)

                term2_exp = torch.mean(torch.exp((1 - beta_val) * marginal - max_val))
                term2 = (1 / (1 - beta_val)) * (torch.log(term2_exp) + max_val)

                S_term1 = torch.mean(joint**2) - term1**2
                m_term1 = term1

                m_term2 = term2_exp
                S_term2 = torch.mean((torch.exp((1 - beta_val) * marginal - max_val)) ** 2) - (m_term2) ** 2


            elif beta_val == 1:
                max_val = torch.max(- beta_val * joint)

                term1_exp = torch.mean(torch.exp(-beta_val * joint - max_val))
                term1 = -(1 / beta_val) * (torch.log(term1_exp) + max_val)

                term2 = torch.mean(marginal)

                m_term1 = term1_exp
                S_term1 = torch.mean((torch.exp((-beta_val) * joint - max_val)) ** 2) - (m_term1) ** 2

                S_term2 = torch.mean(marginal**2) - term2**2
                m_term2 = term2

            else:
                max_val_1 = torch.max(- beta_val * joint)
                max_val_2 = torch.max((1 - beta_val) * marginal)

                term1_exp = torch.mean(torch.exp(-beta_val * joint - max_val_1))
                term1 = -(1 / beta_val) * (torch.log(term1_exp) + max_val_1)

                term2_exp = torch.mean(torch.exp((1 - beta_val) * marginal - max_val_2))
                term2 = (1 / (1 - beta_val)) * (torch.log(term2_exp) + max_val_2)

                m_term1 = term1_exp
                S_term1 = torch.mean((torch.exp((-beta_val) * joint - max_val_1)) ** 2) - (m_term1) ** 2

                m_term2 = term2_exp
                S_term2 = torch.mean((torch.exp((1 - beta_val) * marginal - max_val_2)) ** 2) - (m_term2) ** 2

            if not self.var_flag:
                loss += (term1 - term2)

            else:

                if beta_val == 0:
                    VR_penalty = S_term1 + (1.0 / (1 - beta_val)**2) * (S_term2/ m_term2**2)
                elif beta_val == 1:
                    VR_penalty = (1.0 / (-beta_val) ** 2) * torch.div(S_term1, m_term1 ** 2) + S_term2
                else:
                    VR_penalty = (1.0 / (-beta_val)**2) * (S_term1 / m_term1 ** 2) + (1 / (1 - beta_val)**2) * (S_term2 / m_term2 ** 2)

                ext_loss += (term1 - term2) 

                loss += ext_loss - lambda_VR * VR_penalty

        return ext_loss, loss

    def mi_est(self, x_samples, y_samples, lambda_VR):  # samples have shape [sample_size, dim]
        # shuffle and concatenate
        sample_size = y_samples.shape[0]
        random_index = torch.randint(sample_size, (sample_size,)).long()

        y_shuffle = y_samples[random_index]
        T0 = self.mine(torch.cat([x_samples,y_samples], dim = -1))
        T0_external = 50. * torch.tanh((self.ext(T0) + T0)/50.)
        T1 = self.mine(torch.cat([x_samples,y_shuffle], dim = -1))
        T1_external = 50. * torch.tanh((self.ext(T1)+ T1)/50.)       # 20, 10
        lower_bound, loss = self.measure_mi(T0_external, T1_external, lambda_VR)

        # compute the negative loss (maximise loss == minimise -loss)
        return lower_bound, loss

class R_MINE_VAR_v3(nn.Module):
    def __init__(self, x_dim, y_dim, hidden_size, hidden_size_ext,  beta_values, var_flag):
        super(R_MINE_VAR_v3, self).__init__()
        self.fc1 = nn.Linear(x_dim + y_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size//2)
        self.fc4 = nn.Linear(hidden_size//2, 1)

        self.ext = nn.Sequential(nn.Linear(hidden_size//2, hidden_size_ext),
                                    nn.Tanh(),
                                    nn.Linear(hidden_size_ext, 1))
        self.act = nn.Tanh()
        self.beta = beta_values
        self.var_flag = var_flag

    def measure_mi(self, joint, marginal, lambda_VR):
        ext_loss, loss = 0, 0
        # Reyni Divengence Disentanglement information

        for i, beta_val in enumerate(self.beta):
            term1, term2 = 0.0, 0.0
            if beta_val == 0:
                max_val = torch.max((1 - beta_val) * marginal)

                term1 = torch.mean(joint)

                term2_exp = torch.mean(torch.exp((1 - beta_val) * marginal - max_val))
                term2 = (1 / (1 - beta_val)) * (torch.log(term2_exp) + max_val)

                S_term1 = torch.mean(joint**2) - term1**2
                m_term1 = term1

                m_term2 = term2_exp
                S_term2 = torch.mean((torch.exp((1 - beta_val) * marginal - max_val)) ** 2) - (m_term2) ** 2


            elif beta_val == 1:
                max_val = torch.max(- beta_val * joint)

                term1_exp = torch.mean(torch.exp(-beta_val * joint - max_val))
                term1 = -(1 / beta_val) * (torch.log(term1_exp) + max_val)

                term2 = torch.mean(marginal)

                m_term1 = term1_exp
                S_term1 = torch.mean((torch.exp((-beta_val) * joint - max_val)) ** 2) - (m_term1) ** 2

                S_term2 = torch.mean(marginal**2) - term2**2
                m_term2 = term2

            else:
                max_val_1 = torch.max(- beta_val * joint)
                max_val_2 = torch.max((1 - beta_val) * marginal)

                term1_exp = torch.mean(torch.exp(-beta_val * joint - max_val_1))
                term1 = -(1 / beta_val) * (torch.log(term1_exp) + max_val_1)

                term2_exp = torch.mean(torch.exp((1 - beta_val) * marginal - max_val_2))
                term2 = (1 / (1 - beta_val)) * (torch.log(term2_exp) + max_val_2)

                m_term1 = term1_exp
                S_term1 = torch.mean((torch.exp((-beta_val) * joint - max_val_1)) ** 2) - (m_term1) ** 2

                m_term2 = term2_exp
                S_term2 = torch.mean((torch.exp((1 - beta_val) * marginal - max_val_2)) ** 2) - (m_term2) ** 2

            if not self.var_flag:
                loss += (term1 - term2)

            else:

                if beta_val == 0:
                    VR_penalty = S_term1 + (1.0 / (1 - beta_val)**2) * (S_term2/ m_term2**2)
                elif beta_val == 1:
                    VR_penalty = (1.0 / (-beta_val) ** 2) * torch.div(S_term1, m_term1 ** 2) + S_term2
                else:
                    VR_penalty = (1.0 / (-beta_val)**2) * (S_term1 / m_term1 ** 2) + (1 / (1 - beta_val)**2) * (S_term2 / m_term2 ** 2)

                ext_loss += (term1 - term2)

                loss += ext_loss - lambda_VR * VR_penalty

        return ext_loss, loss

    def mi_est(self, x_samples, y_samples, lambda_VR):  # samples have shape [sample_size, dim]
        # shuffle and concatenate
        sample_size = y_samples.shape[0]
        random_index = torch.randint(sample_size, (sample_size,)).long()

        y_shuffle = y_samples[random_index]
        T0 = self.fc3(self.act(self.fc2(self.act(self.fc1(torch.cat([x_samples,y_samples], dim = -1))))))
        T0_external = 50. * torch.tanh((self.ext(T0) + self.fc4(self.act(T0)))/50.)
        T1 = self.fc3(self.act(self.fc2(self.act(self.fc1(torch.cat([x_samples,y_shuffle], dim = -1))))))
        T1_external = 50. * torch.tanh((self.ext(T1)+ self.fc4(self.act(T1)))/50.) 
        lower_bound, loss = self.measure_mi(T0_external, T1_external, lambda_VR)

        # compute the negative loss (maximise loss == minimise -loss)
        return lower_bound, loss


class R_MINE_VAR_adaptive(nn.Module):
    def __init__(self, x_dim, y_dim, hidden_size, hidden_size_ext,  beta_values, var_flag):
        super(R_MINE_VAR_adaptive, self).__init__()
        self.mine = nn.Sequential(nn.Linear(x_dim + y_dim, hidden_size),
                                    nn.Tanh(),
                                    nn.Linear(hidden_size, hidden_size),
                                    nn.Tanh(),
                                    nn.Linear(hidden_size, hidden_size),
                                    nn.Tanh(),
                                    nn.Linear(hidden_size, hidden_size//2),
                                    nn.Tanh(),
                                    nn.Linear(hidden_size//2, 1))

        self.ext = nn.Sequential(nn.Linear(1, hidden_size_ext),
                                    nn.Tanh(),
                                    nn.Linear(hidden_size_ext, 1))
        self.beta = beta_values
        self.var_flag = var_flag
        self.upper_thres = 0.1
        self.lower_thres = 0.001
        self.K = 50

    def measure_mi(self, joint, marginal, lambda_VR):
        ext_loss, loss, VR_penalty = 0, 0, 0
        # Reyni Divengence Disentanglement information

        for i, beta_val in enumerate(self.beta):
            term1, term2 = 0.0, 0.0
            if beta_val == 0:
                max_val = torch.max((1 - beta_val) * marginal)

                term1 = torch.mean(joint)

                term2_exp = torch.mean(torch.exp((1 - beta_val) * marginal - max_val))
                term2 = (1 / (1 - beta_val)) * (torch.log(term2_exp) + max_val)

                # S_term1 = torch.mean(joint**2) - term1**2
                # m_term1 = term1

                # m_term2 = term2_exp
                # S_term2 = torch.mean((torch.exp((1 - beta_val) * marginal - max_val)) ** 2) - (m_term2) ** 2
                vp_loss_term1 = torch.var_mean(joint)
                m_term2 = torch.exp((1 - beta_val) * marginal - max_val)
                vp_loss_term2 = int(1.0 / (1 - beta_val)**2) * torch.var_mean(m_term2) / torch.mean(m_term2)**2

            elif beta_val == 1:
                max_val = torch.max(- beta_val * joint)

                term1_exp = torch.mean(torch.exp(-beta_val * joint - max_val))
                term1 = -(1 / beta_val) * (torch.log(term1_exp) + max_val)

                term2 = torch.mean(marginal)

                # m_term1 = term1_exp
                # S_term1 = torch.mean((torch.exp((-beta_val) * joint - max_val)) ** 2) - (m_term1) ** 2

                # S_term2 = torch.mean(marginal**2) - term2**2
                # m_term2 = term2

                m_term1 = torch.exp((- beta_val) * joint - max_val)
                vp_loss_term1 = (int(1.0 / (beta_val**2)) * torch.var_mean(m_term1) / torch.mean(m_term1)**2)
                vp_loss_term2 = torch.var_mean(marginal)


            else:
                max_val_1 = torch.max(- beta_val * joint)
                max_val_2 = torch.max((1 - beta_val) * marginal)

                term1_exp = torch.mean(torch.exp(-beta_val * joint - max_val_1))
                term1 = -(1 / beta_val) * (torch.log(term1_exp) + max_val_1)

                term2_exp = torch.mean(torch.exp((1 - beta_val) * marginal - max_val_2))
                term2 = (1 / (1 - beta_val)) * (torch.log(term2_exp) + max_val_2)

                # m_term1 = term1_exp
                # S_term1 = torch.mean((torch.exp((-beta_val) * joint - max_val_1)) ** 2) - (m_term1) ** 2

                # m_term2 = term2_exp
                # S_term2 = torch.mean((torch.exp((1 - beta_val) * marginal - max_val_2)) ** 2) - (m_term2) ** 2

                m_term1 = torch.exp((- beta_val) * joint - max_val_1)
                vp_loss_term1 = (1.0 / (beta_val**2) * torch.var_mean(m_term1) / torch.mean(m_term1)**2)

                m_term2 = torch.exp((1 - beta_val) * marginal - max_val_2)
                vp_loss_term2 = (1.0 / (1 - beta_val)**2) * torch.var_mean(m_term2) / torch.mean(m_term2)**2

            if not self.var_flag:
                loss += (term1 - term2)

            else:

                ext_loss += (term1 - term2) 
                VR_penalty += (vp_loss_term1 + vp_loss_term2)

                if VR_penalty/ext_loss**2 > self.K:
                    lambda_VR = 1.01*lambda_VR
                else:
                    lambda_VR = 0.99*lambda_VR

                lambda_VR = max(min(lambda_VR, self.upper_thres), self.lower_thres)

                loss += ext_loss - lambda_VR * VR_penalty

        return ext_loss, loss

    def mi_est(self, x_samples, y_samples, lambda_VR):  # samples have shape [sample_size, dim]
        # shuffle and concatenate
        sample_size = y_samples.shape[0]
        random_index = torch.randint(sample_size, (sample_size,)).long()

        y_shuffle = y_samples[random_index]
        T0 = self.mine(torch.cat([x_samples,y_samples], dim = -1))
        T0_external = 50. * torch.tanh((self.ext(T0) + T0)/50.)
        T1 = self.mine(torch.cat([x_samples,y_shuffle], dim = -1))
        T1_external = 50. * torch.tanh((self.ext(T1)+ T1)/50.)       # 20, 10
        lower_bound, loss = self.measure_mi(T0_external, T1_external, lambda_VR)

        # compute the negative loss (maximise loss == minimise -loss)
        return lower_bound, loss



