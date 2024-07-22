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