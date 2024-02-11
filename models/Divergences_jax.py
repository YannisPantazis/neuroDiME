import jax
import jax.numpy as jnp
import time
import math
from flax.training import train_state
import optax

class Divergence:
    '''
    Divergence D(P||Q) between random variables x~P, y~Q.
    Parent class where the common parameters and the common functions are defined.
    '''
    # initialize
    def __init__(self, discriminator, disc_optimizer, epochs, batch_size, init_state, discriminator_penalty=None):
        self.batch_size = batch_size
        self.epochs = epochs
        self.discriminator = discriminator
        self.disc_optimizer = disc_optimizer
        self.discriminator_penalty = discriminator_penalty
        self.state = init_state

    def __repr__(self):
        return 'discriminator: {}'.format(self.discriminator)

    def discriminate(self, x): 
        ''' g(x) '''
        # y = self.discriminator(x)
        y = self.state.apply_fn({"params": self.state.params}, x)
        return y

    def eval_var_formula(self, x, y):
        ''' depends on the variational formula to be used '''
        return None

    def estimate(self, x, y): 
        ''' same as self.eval_var_formula() '''
        divergence_loss = self.eval_var_formula(x, y)
        return divergence_loss

    def discriminator_loss(self, x, y): 
        ''' same as self.estimate() (in principle) '''
        divergence_loss = self.eval_var_formula(x, y)
        return divergence_loss


    # @jax.jit
    def train_step(self, x, y):
        # grad_loss = jax.grad(loss_fn)(x, y)

        ''' discriminator's parameters update '''
        def loss_fn(_, x, y):
            loss = -self.discriminator_loss(x, y) # with minus because we maximize the discrimination loss

            if self.discriminator_penalty is not None:
                loss += self.discriminator_penalty.evaluate(self.discriminator, x, y)
                
            return loss
        
        grads_fn = jax.grad(loss_fn)
        grads = grads_fn(self.state.params, x, y)
        self.state = self.state.apply_gradients(grads=grads)


    def train(self, data_P, data_Q, save_estimates=True):
        ''' training function for our models '''
        # dataset slicing into minibatches
        P_dataset = DataLoader(data_P, batch_size=self.batch_size, shuffle=True)
        Q_dataset = DataLoader(data_Q, batch_size=self.batch_size, shuffle=True)

        estimates=[]
        
        for i in range(self.epochs):
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


class IPM(Divergence):
    '''
    IPM class
    '''
    def eval_var_formula(self, x, y):
        ''' Evaluation of variational formula of IPM. '''
        D_P = self.discriminate(x)
        D_Q = self.discriminate(y)

        D_loss_P = jnp.mean(D_P)
        D_loss_Q = jnp.mean(D_Q)

        D_loss = D_loss_P - D_loss_Q
        return D_loss


class f_Divergence(Divergence):
    '''
    f-divergence class (parent class)
    D_f(P||Q), x~P, y~Q.
    '''

    def f_star(self, y):
        ''' Legendre transform of f '''
        return None
    
    def final_layer_activation(self, y):
        return y

    def eval_var_formula(self, x, y):
        ''' Evaluation of variational formula of f-divergence, D_f(P||Q), x~P, y~Q. '''
        D_P = self.discriminate(x)
        D_P = self.final_layer_activation(D_P)
        D_Q = self.discriminate(y)
        D_Q = self.final_layer_activation(D_Q)
        
        D_loss_P = jnp.mean(D_P)
        D_loss_Q = jnp.mean(self.f_star(D_Q))
        
        D_loss = D_loss_P - D_loss_Q
        return D_loss


class KLD_DV(Divergence):
    '''
    KL divergence class (based on the Donsker-Varahdan variational formula)
    KL(P||Q), x~P, y~Q.
    '''
    def eval_var_formula(self, x, y):
        ''' Evaluation of variational formula of KL divergence (based on the Donsker-Varahdan variational formula), KL(P||Q), x~P, y~Q.'''
        D_P = self.discriminate(x)
        D_Q = self.discriminate(y)
        
        D_loss_P = jnp.mean(D_P)
        
        max_val = jnp.max(D_Q)
        D_loss_Q = jnp.log(jnp.mean(jnp.exp(D_Q - max_val))) + max_val

        D_loss = D_loss_P - D_loss_Q
        return D_loss
    

class KLD_LT(f_Divergence):
    '''
    Kullback-Leibler (KL) divergence class (based on Legendre transform)
    KL(P||Q), x~P, y~Q.
    '''
    def f_star(self, y):
        ''' Legendre transform of f(y)=y*log(y) '''
        f_star_y = jnp.exp(y-1)
        return f_star_y


class Pearson_chi_squared_LT(f_Divergence):
    '''
    Pearson chi^2-divergence class (based on Legendre transform)
    chi^2(P||Q), x~P, y~Q.
    '''
    def f_star(self, y):
        ''' Legendre transform of f(y)=(y-1)^2 '''
        f_star_y = 0.25*jnp.power(y,2.0) + y
        return f_star_y


class squared_Hellinger_LT(f_Divergence):
    '''
    squared Hellinger distance class (based on Legendre transform)
    H(P||Q), x~P, y~Q.
    ''' 
    def f_star(self, y):
        ''' Legendre transform of f(y)=(sqrt(y)-1)^2 '''
        f_star_y = y / (1-y)
        return f_star_y

    def final_layer_activation(self, y):
        ''' Final layer activation function. '''
        out = 1.0 - jnp.exp(-y)
        return out


class Jensen_Shannon_LT(f_Divergence):
    '''
    Jensen-Shannon divergence class (based on Legendre transform)
    JS(P||Q), x~P, y~Q.
    '''
    def f_star(self, y):
        ''' Legendre transform of f(y)=y*log(y)-(y+1)*log((y+1)/2) '''
#        max_val = jnp.max(y)
        f_star_y = -jnp.log(2.0-jnp.exp(y))
        return f_star_y

    def final_layer_activation(self, y):
        ''' Final layer activation function. '''
        out = - jnp.log(0.5 + 0.5*jnp.exp(-y))
        return out


class alpha_Divergence_LT(f_Divergence):
    '''
    alpha-divergence class (based on Legendre transform)
    D_{f_alpha}(P||Q), x~P, y~Q.
    '''

    # initialize
    def __init__(self, discriminator, disc_optimizer, alpha, epochs, batch_size,discriminator_penalty=None):
        
        Divergence.__init__(self, discriminator, disc_optimizer, epochs, batch_size, discriminator_penalty)
        self.alpha = alpha # order
    
    def get_order(self):
        return self.alpha

    def set_order(self, alpha):
        self.alpha = alpha
    
    def f_star(self, y):
        ''' Legendre transform of f_alpha '''
        if self.alpha>1.0:
            f_star_y = ((self.alpha-1.0)*jnp.maximum(y, 0.0))**(self.alpha/(self.alpha-1.0)) / self.alpha + 1.0/(self.alpha*(self.alpha-1.0))
        elif (self.alpha<1.0) & (self.alpha>0.0):
            f_star_y = jnp.power((1.0-self.alpha)*jnp.maximum(y, 0.0), self.alpha/(self.alpha-1.0)) / self.alpha - 1.0/(self.alpha*(self.alpha-1.0))
        
        return f_star_y
        
        
class Pearson_chi_squared_HCR(Divergence):
    '''
    Pearson chi^2-divergence class (based on Hammersley-Chapman-Robbins bound)
    chi^2(P||Q), x~P, y~Q.
    '''
    def eval_var_formula(self, x, y):
        ''' Evaluation of variational formula of Pearson chi^2-divergence (based on Hammersley-Chapman-Robbins bound), chi^2(P||Q), x~P, y~Q.'''
        D_P = self.discriminate(x)
        D_Q = self.discriminate(y)

        D_loss_P = jnp.mean(D_P)
        D_loss_Q = jnp.mean(D_Q)

        D_loss = (D_loss_P - D_loss_Q)**2 / jnp.var(D_Q)
        return D_loss


class Renyi_Divergence(Divergence):
    '''
    Renyi divergence class
    R_alpha(P||Q), x~P, y~Q.
    '''
    # initialize
    def __init__(self, discriminator, disc_optimizer, alpha, epochs, batch_size, discriminator_penalty=None):
        
        Divergence.__init__(self, discriminator, disc_optimizer, epochs, batch_size, discriminator_penalty)
        self.alpha = alpha # Renyi Divergence order

    def get_order(self):
        return self.alpha

    def set_order(self, alpha):
        self.alpha = alpha


class Renyi_Divergence_DV(Renyi_Divergence):
    '''
    Renyi divergence class (based on the Renyi-Donsker-Varahdan variational formula)
    R_alpha(P||Q), x~P, y~Q.
    '''
    def eval_var_formula(self, x, y):
        ''' Evaluation of variational formula of Renyi divergence (based on the Renyi-Donsker-Varahdan variational formula), R_alpha(P||Q), x~P, y~Q. '''
        gamma = self.alpha
        beta = 1.0 - self.alpha

        D_P = self.discriminate(x)
        D_Q = self.discriminate(y)

        if beta == 0.0:
            D_loss_P = jnp.mean(D_P)
        else:
            max_val = jnp.max((-beta) * D_P)
            D_loss_P = -(1.0/beta) * (jnp.log(jnp.mean(jnp.exp((-beta) * D_P - max_val))) + max_val)

        if gamma == 0.0:
            D_loss_Q = jnp.mean(D_Q)
        else:
            max_val = jnp.max((gamma) * D_Q)
            D_loss_Q = (1.0/gamma) * (jnp.log(jnp.mean(jnp.exp(gamma * D_Q - max_val))) + max_val)

        D_loss = D_loss_P - D_loss_Q
        return D_loss


class Renyi_Divergence_CC(Renyi_Divergence):
    '''
    Renyi divergence class (based on the convex-conjugate variational formula)
    R_alpha(P||Q), x~P, y~Q.
    '''    
    # initialize
    def __init__(self, discriminator, disc_optimizer, alpha, epochs, batch_size, final_act_func, discriminator_penalty=None):
        
        Renyi_Divergence.__init__(self, discriminator, disc_optimizer, alpha, epochs, batch_size, discriminator_penalty)
        self.final_act_func = final_act_func
    
    def final_layer_activation(self, y): 
        ''' Final layer activation function, enforces positive values. '''
        if self.final_act_func=='abs':
            out = jnp.abs(y)
        elif self.final_act_func=='softplus':
            out = jax.nn.softplus(y)
        elif self.final_act_func=='poly-softplus':
            out = 1.0+(1.0/(1.0+jnp.maximum(-y,0.0))-1.0)*(1.0-jnp.sign(y))/2.0 +y*(jnp.sign(y)+1.0)/2.0
        
        return out
    
    def eval_var_formula(self, x, y):
        ''' Evaluation of variational formula of Renyi divergence (based on the convex-conjugate variational formula), R_alpha(P||Q), x~P, y~Q. '''
        D_P = self.discriminate(x)
        D_P = self.final_layer_activation(D_P)
        D_Q = self.discriminate(y)
        D_Q = self.final_layer_activation(D_Q)
        
        D_loss_Q = -jnp.mean(D_Q)
        D_loss_P = jnp.log(jnp.mean(jnp.power(D_P, (self.alpha-1.0)/self.alpha))) / (self.alpha-1.0)
        
        D_loss = D_loss_Q + D_loss_P + (jnp.log(self.alpha)+1.0)/self.alpha
        return D_loss


class Renyi_Divergence_CC_rescaled(Renyi_Divergence_CC):
    '''
    Rescaled Renyi divergence class (based on the rescaled convex-conjugate variational formula)
    alpha*R_alpha(P||Q), x~P, y~Q.
    '''    
    def final_layer_activation(self, y):
        ''' Final layer activation function, enforces positive values. '''
#        super(Renyi_Divergence_CC, self).final_layer_activation(self, y)
        out = Renyi_Divergence_CC.final_layer_activation(self, y)
        out = out/self.alpha
        
        return out

    def eval_var_formula(self, x, y):
        ''' Evaluation of variational formula of Rescaled Renyi divergence class (based on the rescaled convex-conjugate variational formula), alpha*R_alpha(P||Q), x~P, y~Q.'''
#        super(Renyi_Divergence_CC, self).eval_var_formula(self, x, y)
        D_loss = Renyi_Divergence_CC.eval_var_formula(self, x, y)
        D_loss = D_loss * self.alpha
        
        return D_loss


class Renyi_Divergence_WCR(Renyi_Divergence_CC):
    '''
    Rescaled Renyi divergence class as alpha --> infinity (aka worst-case regret divergence)
    D_\infty(P||Q), x~P, y~Q.
    '''
    # initialize
    def __init__(self, discriminator, disc_optimizer, epochs, batch_size, final_act_func, discriminator_penalty=None):

        Renyi_Divergence_CC.__init__(self, discriminator, disc_optimizer, math.inf, epochs, batch_size, final_act_func, discriminator_penalty)
   
    def eval_var_formula(self, x, y):
        ''' Evaluation of variational formula of Rescaled Renyi divergence class as alpha --> infinity (aka worst-case regret divergence), D_\infty(P||Q), x~P, y~Q. '''
        D_P = self.discriminate(x)
        D_P = self.final_layer_activation(D_P)
        D_Q = self.discriminate(y)
        D_Q = self.final_layer_activation(D_Q)
        
        D_loss_P = jnp.log(jnp.mean(D_P))
        D_loss_Q = -jnp.mean(D_Q)
        
        D_loss = D_loss_Q + D_loss_P + 1.0
        return D_loss


class Discriminator_Penalty():
    '''
    Discriminator penalty class penalizes the divergence objective functional during training.
    Allows for the (approximate) implementation of discriminator constraints.
    '''
    # initialize
    def __init__(self, penalty_weight):
        self.penalty_weight = penalty_weight # weighting of penalty term
        
    def get_penalty_weight(self):
        return self.penalty_weight

    def set_penalty_weight(self, weight):
        self.penalty_weight = weight
    
    def evaluate(self, discriminator, x, y): 
        ''' depends on the choice of penalty '''
        return None


class Gradient_Penalty_1Sided(Discriminator_Penalty):
    '''
    One-sided gradient penalty (constraint: Lipschitz constant  <= Lip_const)
    '''
    # initialize
    def __init__(self, penalty_weight, Lip_const):
        
        Discriminator_Penalty.__init__(self, penalty_weight)
        self.Lip_const = Lip_const # Lipschitz constant


    def get_Lip_constant(self):
        return self.Lip_const

    def set_Lip_constant(self, L):
        self.Lip_const = L
    
    def evaluate(self, discriminator, x, y): 
        ''' Computes the one-sided gradient penalty (constraint: Lipschitz constant  <= Lip_const). '''
        temp_shape = [x.shape[0]] + [1 for _ in  range(len(x.shape)-1)]
        ratio = jax.random.uniform(jax.random.PRNGKey(0), temp_shape, 0.0, 1.0, dtype=jax.dtypes.float32)
        diff = y - x
        interpltd = x + (ratio * diff) # get the interpolated samples

        with jax.grad() as gp_tape: # get the discriminator output
            gp_tape.watch(interpltd)
            D_pred = discriminator(interpltd, training=True)

        grads = gp_tape.gradient(D_pred, [interpltd])[0] # calculate the gradients      
        if x.shape[1]==1: # calculate the norm
            norm_squared = jnp.square(grads)
        else:
            norm_squared = jnp.sum(jnp.square(grads))

        gp = self.penalty_weight*jnp.mean(jnp.maximum(norm_squared/self.Lip_const**2-1., 0.0)) # one-sided gradient penalty
        return gp

        
class Gradient_Penalty_2Sided(Discriminator_Penalty):
    '''
    Two-sided gradient penalty (constraint: Lipschitz constant = Lip_const)
    '''
    # initialize
    def __init__(self, penalty_weight, Lip_const):
        
        Discriminator_Penalty.__init__(self, penalty_weight)
        self.Lip_const = Lip_const # Lipschitz constant


    def get_Lip_constant(self):
        return self.Lip_const

    def set_Lip_constant(self, L):
        self.Lip_const = L
    

    def evaluate(self, discriminator, x, y): 
        ''' Computes the two-sided gradient penalty (constraint: Lipschitz constant = Lip_const). '''
        temp_shape = [x.shape[0]] + [1 for _ in  range(len(x.shape)-1)]
        ratio = jax.random.uniform(jax.random.PRNGKey(0), temp_shape, 0.0, 1.0, dtype=jax.dtypes.float32)
        diff = y - x
        interpltd = x + (ratio * diff) # get the interpolated samples

        with jax.grad() as gp_tape: # get the discriminator output
            gp_tape.watch(interpltd)
            D_pred = discriminator(interpltd, training=True)

        grads = gp_tape.gradient(D_pred, [interpltd])[0] # calculate the gradients
        if x.shape[1]==1: # calculate the norm
            norm = jnp.sqrt(jnp.square(grads))
        else:
            norm = jnp.sqrt(jnp.sum(jnp.square(grads)))

        gp = self.penalty_weight*jnp.mean((norm - self.Lip_const) ** 2) # two-sided gradient penalty
        return gp


class DataLoader:
    def __init__(self, data, batch_size, shuffle=True):
        self.data = data
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_samples = len(data)
        self.index = jnp.arange(self.num_samples)
        if shuffle:
            self.index = jax.random.permutation(jax.random.PRNGKey(0), self.index)

    def __iter__(self):
        self.current_idx = 0
        return self

    def __next__(self):
        if self.current_idx >= self.num_samples:
            raise StopIteration
        batch_idx = self.index[self.current_idx:self.current_idx+self.batch_size]
        batch = jnp.take(self.data, batch_idx, axis=0)
        self.current_idx += self.batch_size
        return batch