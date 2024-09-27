import jax
import jax.numpy as jnp
import math
from functools import partial
from jax import jit, grad
from tqdm import tqdm
import numpy as np

class Divergence:
    '''
    Base class for Divergence measures D(P||Q) between random variables x~P, y~Q.
    This parent class defines common parameters and functions for different divergence measures.
    '''
    
    def __init__(self, discriminator, disc_optimizer, epochs, batch_size, discriminator_penalty=None):
        '''
        Initializes the Divergence class.

        Args:
            discriminator: The neural network model used to discriminate between P and Q.
            disc_optimizer: Optimizer for the discriminator.
            epochs: Number of training epochs.
            batch_size: Batch size for training.
            discriminator_penalty: Optional penalty applied to discriminator loss.
        '''
        self.batch_size = batch_size
        self.epochs = epochs
        self.discriminator = discriminator
        self.disc_optimizer = disc_optimizer
        self.discriminator_penalty = discriminator_penalty

    def __repr__(self):
        '''
        Returns the string representation of the discriminator model.
        '''
        return 'discriminator: {}'.format(self.discriminator)

    def discriminate(self, x, params, vars, labels=None, dropout_rng=None):
        '''
        Discriminates between samples from x~P and y~Q using the discriminator model.

        Args:
            x: Input data to be discriminated.
            params: Parameters of the discriminator model.
            vars: Additional variables such as batch statistics.
            labels: Optional labels for the input data.
            dropout_rng: Optional dropout key for stochasticity in dropout layers.

        Returns:
            Tuple of the discriminator output and optional updated batch statistics.
        '''
        vars_d = None
        if 'batch_stats' in vars:
            if labels is not None:
                y = self.discriminator.apply({'params': params, 'batch_stats': vars['batch_stats']}, x, labels, mutable=['batch_stats'])
            else:
                y = self.discriminator.apply({'params': params, 'batch_stats': vars['batch_stats']}, x, mutable=['batch_stats'])
            y, vars_d = y
        elif dropout_rng is not None:
            if labels is not None:
                y = self.discriminator.apply({'params': params}, x, labels, rngs={'dropout': dropout_rng})
            else:
                y = self.discriminator.apply({'params': params}, x, rngs={'dropout': dropout_rng})
        else:
            if labels is not None:
                y = self.discriminator.apply({'params': params}, x, labels)
            else:
                y = self.discriminator.apply({'params': params}, x)
        return y, vars_d

    def eval_var_formula(self, x, y, params, vars, labels=None, dropout_rng=None):
        '''
        Placeholder method for evaluating the variational formula of a specific divergence.
        Should be overridden by subclasses.

        Args:
            x: Samples from P.
            y: Samples from Q.
            params: Discriminator parameters.
            vars: Additional discriminator variables.
            labels: Optional input labels.
            dropout_rng: Optional dropout key for stochasticity.

        Returns:
            None.
        '''
        return None

    def estimate(self, x, y, params, vars, labels=None, dropout_rng=None):
        '''
        Estimates the divergence between P and Q.

        Args:
            x: Samples from P.
            y: Samples from Q.
            params: Discriminator parameters.
            vars: Additional discriminator variables.
            labels: Optional input labels.
            dropout_rng: Optional dropout key for stochasticity.

        Returns:
            Tuple of estimated divergence and updated variables.
        '''
        divergence_loss, vars_d = self.eval_var_formula(x, y, params, vars, labels, dropout_rng)
        return divergence_loss, vars_d

    def generator_loss(self, x, params, vars, labels=None, dropout_rng=None):
        '''
        Computes the loss for the generator model.

        Args:
            x: Generated samples.
            params: Discriminator parameters.
            vars: Additional discriminator variables.
            labels: Optional input labels.
            dropout_rng: Optional dropout key for stochasticity.

        Returns:
            Tuple of generator loss and updated variables.
        '''
        divergence_loss, vars_d = self.eval_var_formula_gen(x, params, vars, labels, dropout_rng)
        return divergence_loss, vars_d

    def discriminator_loss(self, x, y, params, vars, labels=None, dropout_rng=None):
        '''
        Computes the loss for the discriminator model.

        Args:
            x: Samples from P.
            y: Samples from Q.
            params: Discriminator parameters.
            vars: Additional discriminator variables.
            labels: Optional input labels.
            dropout_rng: Optional dropout key for stochasticity.

        Returns:
            Tuple of discriminator loss and updated variables.
        '''
        divergence_loss, vars_d = self.eval_var_formula(x, y, params, vars, labels, dropout_rng)
        return divergence_loss, vars_d

    @partial(jit, static_argnums=(0,))
    def train_step(self, x, y, state, vars, key, labels=None, dropout_rng=None):
        '''
        Performs a single training step for the discriminator.

        Args:
            x: Samples from P.
            y: Samples from Q.
            state: Optimizer state.
            vars: Additional discriminator variables.
            key: Random key for JAX RNG.
            labels: Optional input labels.
            dropout_rng: Optional dropout key for stochasticity.

        Returns:
            Updated state and loss value for the current step.
        '''
        def loss_fn(params):
            loss, vars_d = self.discriminator_loss(x, y, params, vars, labels, dropout_rng)
            loss = -loss  # Maximize the discrimination loss
            if self.discriminator_penalty is not None:
                if 'batch_stats' in vars:
                    loss += self.discriminator_penalty.evaluate(self.discriminator, x, y, params, vars['batch_stats'], key, labels, dropout_rng)
                else:
                    loss += self.discriminator_penalty.evaluate(self.discriminator, x, y, params, None, key, labels, dropout_rng)
            return loss, vars_d

        grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
        (loss, vars_d), grads = grad_fn(state.params)
        state = state.apply_gradients(grads=grads)
        if vars_d is not None:
            state = state.replace(batch_stats=vars_d['batch_stats'])
        return state, loss

    @partial(jit, static_argnums=(0,))
    def gen_train_step(self, gen_state, disc_state, disc_vars, gen_vars, key, z, labels=None, dropout_rng=None):
        '''
        Performs a single training step for the generator.

        Args:
            gen_state: Generator optimizer state.
            disc_state: Discriminator optimizer state.
            disc_vars: Discriminator variables.
            gen_vars: Generator variables.
            key: Random key for JAX RNG.
            z: Latent input to the generator.
            labels: Optional input labels.
            dropout_rng: Optional dropout key for stochasticity.

        Returns:
            Updated generator state and generator loss.
        '''
        def loss_fn(params):
            vars_g = None
            if 'batch_stats' in gen_vars:
                if labels is not None:
                    x, vars_g = gen_state.apply_fn({'params': params, 'batch_stats': gen_vars['batch_stats']}, labels=labels, z=z, mutable=['batch_stats'])
                else:
                    x, vars_g = gen_state.apply_fn({'params': params, 'batch_stats': gen_vars['batch_stats']}, z=z, mutable=['batch_stats'])
            elif dropout_rng is not None:
                if labels is not None:
                    x = gen_state.apply_fn({'params': params}, labels=labels, z=z, rngs={'dropout': dropout_rng})
                else:
                    x = gen_state.apply_fn({'params': params}, z=z, rngs={'dropout': dropout_rng})
            else:
                if labels is not None:
                    x = gen_state.apply_fn({'params': params}, labels=labels, z=z)
                else:
                    x = gen_state.apply_fn({'params': params}, z=z)

            loss, _ = self.generator_loss(x, disc_state.params, disc_vars, labels, dropout_rng)
            return loss, vars_g

        grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
        (g_loss, vars_g), grads = grad_fn(gen_state.params)
        gen_state = gen_state.apply_gradients(grads=grads)
        if vars_g is not None:
            gen_state = gen_state.apply_gradients(grads=grads, batch_stats=vars_g['batch_stats'])
        else:
            gen_state = gen_state.apply_gradients(grads=grads)
        return gen_state, g_loss

    def train(self, data_P, data_Q, state, vars, save_estimates=True, labels=None, dropout_rng=None):
        '''
        Trains the model for a given number of epochs.

        Args:
            data_P: Data samples from distribution P.
            data_Q: Data samples from distribution Q.
            state: Discriminator optimizer state.
            vars: Discriminator variables.
            save_estimates: Whether to save divergence estimates.
            labels: Optional input labels.
            dropout_rng: Optional dropout key for stochasticity.

        Returns:
            Tuple of estimated divergences and losses for each epoch.
        '''
        P_dataset = DataLoader(data_P, batch_size=self.batch_size, shuffle=True)
        Q_dataset = DataLoader(data_Q, batch_size=self.batch_size, shuffle=True)
        key = jax.random.PRNGKey(0)
        estimates = []
        losses = np.zeros(self.epochs)
        for i in tqdm(range(self.epochs), desc='Epochs'):
            train_loss = 0.0
            for P_batch, Q_batch in zip(P_dataset, Q_dataset):
                state, loss = self.train_step(P_batch, Q_batch, state, vars, key, labels, dropout_rng)
                train_loss += loss

            if save_estimates:
                estimate, _ = self.estimate(P_batch, Q_batch, state.params, vars, labels, dropout_rng)
                estimates.append(float(estimate))

            losses[i] = train_loss / len(data_P)
        return estimates, losses

    def get_discriminator(self):
        '''
        Returns the discriminator model.
        '''
        return self.discriminator

    def set_discriminator(self, discriminator):
        '''
        Sets a new discriminator model.

        Args:
            discriminator: New discriminator model.
        '''
        self.discriminator = discriminator

    def get_no_epochs(self):
        '''
        Returns the number of training epochs.
        '''
        return self.epochs

    def set_no_epochs(self, epochs):
        '''
        Sets the number of training epochs.

        Args:
            epochs: New number of epochs.
        '''
        self.epochs = epochs

    def get_batch_size(self):
        '''
        Returns the batch size.
        '''
        return self.batch_size

    def set_batch_size(self, batch_size):
        '''
        Sets the batch size.

        Args:
            batch_size: New batch size.
        '''
        self.batch_size = batch_size

    def get_learning_rate(self):
        '''
        Returns the learning rate.
        '''
        return self.learning_rate

    def set_learning_rate(self, lr):
        '''
        Sets the learning rate.

        Args:
            lr: New learning rate.
        '''
        self.learning_rate = lr


class IPM(Divergence):
    '''
    IPM (Integral Probability Metrics) class, a subclass of Divergence.
    Evaluates the IPM between distributions P and Q using a variational formula.
    '''
    def eval_var_formula(self, x, y, params, vars, labels=None, dropout_rng=None):
        '''
        Evaluates the variational formula for IPM.

        Args:
            x: Samples from distribution P.
            y: Samples from distribution Q.
            params: Discriminator parameters.
            vars: Additional discriminator variables.
            labels: Optional input labels.
            dropout_rng: Optional dropout key for stochasticity.

        Returns:
            Tuple of divergence loss and updated variables.
        '''
        D_P, vars_d = self.discriminate(x, params, vars, labels, dropout_rng)
        D_Q, vars_d = self.discriminate(y, params, vars, labels, dropout_rng)

        D_loss_P = jnp.mean(D_P)
        D_loss_Q = jnp.mean(D_Q)

        D_loss = D_loss_P - D_loss_Q
        return D_loss, vars_d

    def eval_var_formula_gen(self, x, params, vars, labels=None, dropout_rng=None):
        '''
        Evaluates the variational formula for IPM when applied to a generator model.

        Args:
            x: Generated samples.
            params: Discriminator parameters.
            vars: Additional discriminator variables.
            labels: Optional input labels.
            dropout_rng: Optional dropout key for stochasticity.

        Returns:
            Tuple of generator loss and updated variables.
        '''
        G_Q, vars_d = self.discriminate(x, params, vars, labels, dropout_rng)
        G_loss_Q = -jnp.mean(G_Q)
        return G_loss_Q, vars_d


class f_Divergence(Divergence):
    '''
    f-divergence class, parent class for f-divergence-based measures D_f(P||Q).
    Subclasses need to implement the Legendre transform of f (f_star).
    '''
    def f_star(self, y):
        '''
        Placeholder for the Legendre transform of the function f.
        Should be implemented by subclasses.

        Args:
            y: Input to the Legendre transform.

        Returns:
            None.
        '''
        return None
    
    def final_layer_activation(self, y):
        '''
        Final activation function applied to the output of the discriminator.

        Args:
            y: Output of the discriminator.

        Returns:
            Activated output.
        '''
        return y

    def eval_var_formula(self, x, y, params, vars, labels=None, dropout_rng=None):
        '''
        Evaluates the variational formula of f-divergence, D_f(P||Q).

        Args:
            x: Samples from distribution P.
            y: Samples from distribution Q.
            params: Discriminator parameters.
            vars: Additional discriminator variables.
            labels: Optional input labels.
            dropout_rng: Optional dropout key for stochasticity.

        Returns:
            Tuple of divergence loss and updated variables.
        '''
        D_P, vars_d = self.discriminate(x, params, vars, labels, dropout_rng)
        D_P = self.final_layer_activation(D_P)
        D_Q, vars_d = self.discriminate(y, params, vars, labels, dropout_rng)
        D_Q = self.final_layer_activation(D_Q)
        
        D_loss_P = jnp.mean(D_P)
        D_loss_Q = jnp.mean(self.f_star(D_Q))
        
        D_loss = D_loss_P - D_loss_Q
        return D_loss, vars_d

    def eval_var_formula_gen(self, x, params, vars, labels=None, dropout_rng=None):
        '''
        Evaluates the variational formula for f-divergence when applied to a generator.

        Args:
            x: Generated samples.
            params: Discriminator parameters.
            vars: Additional discriminator variables.
            labels: Optional input labels.
            dropout_rng: Optional dropout key for stochasticity.

        Returns:
            Tuple of generator loss and updated variables.
        '''
        G_Q, vars_d = self.discriminate(x, params, vars, labels, dropout_rng)
        G_Q = self.final_layer_activation(G_Q)
        G_loss_Q = -jnp.mean(self.f_star(G_Q + 1e-8))
        return G_loss_Q, vars_d


class KLD_DV(Divergence):
    '''
    KL Divergence class based on the Donsker-Varadhan variational formula.
    KL(P||Q), x~P, y~Q.
    '''
    def eval_var_formula(self, x, y, params, vars, labels=None, dropout_rng=None):
        '''
        Evaluates the variational formula for KL divergence.

        Args:
            x: Samples from distribution P.
            y: Samples from distribution Q.
            params: Discriminator parameters.
            vars: Additional discriminator variables.
            labels: Optional input labels.
            dropout_rng: Optional dropout key for stochasticity.

        Returns:
            Tuple of KL divergence loss and updated variables.
        '''
        D_P, vars_d = self.discriminate(x, params, vars, labels, dropout_rng)
        D_Q, vars_d = self.discriminate(y, params, vars, labels, dropout_rng)
        D_loss_P = jnp.mean(D_P)

        # Stabilize log-sum-exp
        max_val = jnp.max(D_Q)
        D_loss_Q = jnp.log(jnp.mean(jnp.exp(D_Q - max_val))) + max_val

        D_loss = D_loss_P - D_loss_Q
        return D_loss, vars_d
    
    def eval_var_formula_gen(self, x, params, vars, labels=None, dropout_rng=None):
        '''
        Evaluates the variational formula for KL divergence applied to a generator.

        Args:
            x: Generated samples.
            params: Discriminator parameters.
            vars: Additional discriminator variables.
            labels: Optional input labels.
            dropout_rng: Optional dropout key for stochasticity.

        Returns:
            Tuple of generator loss and updated variables.
        '''
        G_Q, vars_d = self.discriminate(x, params, vars, labels, dropout_rng)
        max_val = jnp.max(G_Q)
        G_loss_Q = -(jnp.log(jnp.mean(jnp.exp(G_Q - max_val))) + max_val)
        return G_loss_Q, vars_d


class KLD_LT(f_Divergence):
    '''
    Kullback-Leibler (KL) Divergence class based on the Legendre transform.
    KL(P||Q), x~P, y~Q.
    '''
    def f_star(self, y):
        '''
        Legendre transform of f(y) = y * log(y).

        Args:
            y: Input to the Legendre transform.

        Returns:
            Transformed value.
        '''
        f_star_y = jnp.exp(y - 1)
        return f_star_y


class Pearson_chi_squared_LT(f_Divergence):
    '''
    Pearson chi-squared divergence class based on the Legendre transform.
    chi^2(P||Q), x~P, y~Q.
    '''
    def f_star(self, y):
        '''
        Legendre transform of f(y) = (y - 1)^2.

        Args:
            y: Input to the Legendre transform.

        Returns:
            Transformed value.
        '''
        f_star_y = 0.25 * jnp.power(y, 2.0) + y
        return f_star_y


class Pearson_chi_squared_HCR(Divergence):
    '''
    Pearson chi-squared divergence class based on the Hammersley-Chapman-Robbins bound.
    chi^2(P||Q), x~P, y~Q.
    '''
    def eval_var_formula(self, x, y, params, vars, labels=None, dropout_rng=None):
        '''
        Evaluates the variational formula for Pearson chi-squared divergence based on the Hammersley-Chapman-Robbins bound.

        Args:
            x: Samples from distribution P.
            y: Samples from distribution Q.
            labels: Optional labels for the input data.

        Returns:
            Pearson chi-squared divergence loss.
        '''
        D_P, vars_d = self.discriminate(x, params, vars, labels, dropout_rng)
        D_Q, vars_d = self.discriminate(y, params, vars, labels, dropout_rng)

        D_loss_P = jnp.mean(D_P)
        D_loss_Q = jnp.mean(D_Q)

        D_loss = (D_loss_P - D_loss_Q)**2 / jnp.var(D_Q)
        return D_loss, vars_d
    
    def eval_var_formula_gen(self, x, params, vars, labels=None, dropout_rng=None):
        ''' Evaluates the generator's objective based on Pearson chi-squared divergence. '''
        G_Q, vars_d = self.discriminate(x, params, vars, labels, dropout_rng)
        G_loss_Q = -jnp.mean(G_Q)
        return G_loss_Q, vars_d
    
    
class squared_Hellinger_LT(f_Divergence):
    '''
    Squared Hellinger distance class based on the Legendre transform.
    H(P||Q), x~P, y~Q.
    ''' 
    def f_star(self, y):
        '''
        Legendre transform of f(y) = (sqrt(y) - 1)^2.

        Args:
            y: Input to the Legendre transform.

        Returns:
            Transformed value.
        '''
        f_star_y = y / (1 - y)
        return f_star_y

    def final_layer_activation(self, y):
        '''
        Final layer activation for squared Hellinger distance.

        Args:
            y: Input to the activation function.

        Returns:
            Activated output.
        '''
        out = 1.0 - jnp.exp(-y)
        return out


class Jensen_Shannon_LT(f_Divergence):
    '''
    Jensen-Shannon divergence class based on the Legendre transform.
    JS(P||Q), x~P, y~Q.
    '''
    def f_star(self, y):
        '''
        Legendre transform of f(y) = y * log(y) - (y + 1) * log((y + 1) / 2).

        Args:
            y: Input to the Legendre transform.

        Returns:
            Transformed value.
        '''
        f_star_y = -jnp.log(2.0 - jnp.exp(y))
        return f_star_y

    def final_layer_activation(self, y):
        '''
        Final layer activation function for Jensen-Shannon divergence.

        Args:
            y: Input to the activation function.

        Returns:
            Activated output.
        '''
        out = -jnp.log(0.5 + 0.5 * jnp.exp(-y))
        return out


class alpha_Divergence_LT(f_Divergence):
    '''
    Alpha-divergence class based on the Legendre transform.
    D_f_alpha(P||Q), x~P, y~Q.
    '''
    def __init__(self, discriminator, disc_optimizer, alpha, epochs, batch_size, discriminator_penalty=None):
        '''
        Initializes the alpha-divergence class.

        Args:
            discriminator: Discriminator model.
            disc_optimizer: Optimizer for the discriminator.
            alpha: Order of the alpha-divergence.
            epochs: Number of training epochs.
            batch_size: Batch size for training.
            discriminator_penalty: Optional penalty applied to discriminator loss.
        '''
        super().__init__(discriminator, disc_optimizer, epochs, batch_size, discriminator_penalty)
        self.alpha = alpha

    def get_order(self):
        '''
        Returns the order of the alpha-divergence.

        Returns:
            Alpha order.
        '''
        return self.alpha

    def set_order(self, alpha):
        '''
        Sets the order of the alpha-divergence.

        Args:
            alpha: New alpha order.
        '''
        self.alpha = alpha
    
    def f_star(self, y):
        '''
        Legendre transform of f_alpha based on the alpha value.

        Args:
            y: Input to the Legendre transform.

        Returns:
            Transformed value.
        '''
        if self.alpha > 1.0:
            f_star_y = ((self.alpha - 1.0) * jnp.maximum(y, 0.0))**(self.alpha / (self.alpha - 1.0)) / self.alpha + 1.0 / (self.alpha * (self.alpha - 1.0))
        elif 0.0 < self.alpha < 1.0:
            f_star_y = jnp.power((1.0 - self.alpha) * jnp.maximum(y, 0.0), self.alpha / (self.alpha - 1.0)) / self.alpha - 1.0 / (self.alpha * (self.alpha - 1.0))
        
        return f_star_y


class Renyi_Divergence(Divergence):
    '''
    Renyi divergence class, a subclass of Divergence.
    R_alpha(P||Q), x~P, y~Q.
    '''
    def __init__(self, discriminator, disc_optimizer, alpha, epochs, batch_size, discriminator_penalty=None):
        '''
        Initializes the Renyi divergence class.

        Args:
            discriminator: Discriminator model.
            disc_optimizer: Optimizer for the discriminator.
            alpha: Order of the Renyi divergence.
            epochs: Number of training epochs.
            batch_size: Batch size for training.
            discriminator_penalty: Optional penalty applied to discriminator loss.
        '''
        super().__init__(discriminator, disc_optimizer, epochs, batch_size, discriminator_penalty)
        self.alpha = alpha

    def get_order(self):
        '''
        Returns the order of the Renyi divergence.

        Returns:
            Alpha order.
        '''
        return self.alpha

    def set_order(self, alpha):
        '''
        Sets the order of the Renyi divergence.

        Args:
            alpha: New alpha order.
        '''
        self.alpha = alpha


class Renyi_Divergence_DV(Renyi_Divergence):
    '''
    Renyi divergence class based on the Renyi-Donsker-Varadhan variational formula.
    R_alpha(P||Q), x~P, y~Q.
    '''
    def eval_var_formula(self, x, y, params, vars, labels=None, dropout_rng=None):
        '''
        Evaluates the variational formula of Renyi divergence.

        Args:
            x: Samples from distribution P.
            y: Samples from distribution Q.
            params: Discriminator parameters.
            vars: Additional discriminator variables.
            labels: Optional input labels.
            dropout_rng: Optional dropout key for stochasticity.

        Returns:
            Tuple of Renyi divergence loss and updated variables.
        '''
        gamma = self.alpha
        beta = 1.0 - self.alpha

        D_P, vars_d = self.discriminate(x, params, vars, labels, dropout_rng)
        D_Q, vars_d = self.discriminate(y, params, vars, labels, dropout_rng)

        if beta == 0.0:
            D_loss_P = jnp.mean(D_P)
        else:
            max_val = jnp.max((-beta) * D_P)
            D_loss_P = -(1.0 / beta) * (jnp.log(jnp.mean(jnp.exp((-beta) * D_P - max_val))) + max_val)

        if gamma == 0.0:
            D_loss_Q = jnp.mean(D_Q)
        else:
            max_val = jnp.max((gamma) * D_Q)
            D_loss_Q = (1.0 / gamma) * (jnp.log(jnp.mean(jnp.exp(gamma * D_Q - max_val))) + max_val)

        D_loss = D_loss_P - D_loss_Q
        return D_loss, vars_d

    def eval_var_formula_gen(self, x, params, vars, labels=None, dropout_rng=None):
        '''
        Evaluates the variational formula of Renyi divergence for the generator.

        Args:
            x: Generated samples.
            params: Discriminator parameters.
            vars: Additional discriminator variables.
            labels: Optional input labels.
            dropout_rng: Optional dropout key for stochasticity.

        Returns:
            Tuple of generator loss and updated variables.
        '''
        gamma = self.alpha

        G_Q, vars_d = self.discriminate(x, params, vars, labels, dropout_rng)
        if gamma == 0.0:
            G_loss_Q = -jnp.mean(G_Q)
        else:
            max_val = jnp.max((gamma) * G_Q)
            G_loss_Q = -(1.0 / gamma) * (jnp.log(jnp.mean(jnp.exp(gamma * G_Q - max_val))) + max_val)
        return G_loss_Q, vars_d


class Renyi_Divergence_CC(Renyi_Divergence):
    '''
    Renyi divergence class based on the convex-conjugate variational formula.
    R_alpha(P||Q), x~P, y~Q.
    '''

    def __init__(self, discriminator, disc_optimizer, alpha, epochs, batch_size, final_act_func, discriminator_penalty=None):
        '''
        Initializes the Renyi divergence class using the convex-conjugate variational formula.

        Args:
            discriminator: Discriminator model.
            disc_optimizer: Optimizer for the discriminator.
            alpha: Order of the Renyi divergence.
            epochs: Number of training epochs.
            batch_size: Batch size for training.
            final_act_func: Final activation function to enforce positivity in the output.
            discriminator_penalty: Optional penalty applied to the discriminator.
        '''
        super().__init__(discriminator, disc_optimizer, alpha, epochs, batch_size, discriminator_penalty)
        self.final_act_func = final_act_func
    
    def final_layer_activation(self, y):
        '''
        Final layer activation function to enforce positive values.

        Args:
            y: Output of the discriminator.

        Returns:
            Activated output, ensuring positivity based on the final activation function.
        '''
        if self.final_act_func == 'abs':
            out = jnp.abs(y)
        elif self.final_act_func == 'softplus':
            out = jax.nn.softplus(y)
        elif self.final_act_func == 'poly-softplus':
            out = 1.0 + (1.0 / (1.0 + jnp.maximum(-y, 0.0)) - 1.0) * (1.0 - jnp.sign(y)) / 2.0 + y * (jnp.sign(y) + 1.0) / 2.0
        
        return out
    
    def eval_var_formula(self, x, y, params, vars, labels=None, dropout_rng=None):
        '''
        Evaluates the variational formula of Renyi divergence using the convex-conjugate variational formula.

        Args:
            x: Samples from distribution P.
            y: Samples from distribution Q.
            params: Discriminator parameters.
            vars: Additional discriminator variables.
            labels: Optional input labels.
            dropout_rng: Optional dropout key for stochasticity.

        Returns:
            Tuple of Renyi divergence loss and updated variables.
        '''
        D_P, vars_d = self.discriminate(x, params, vars, labels, dropout_rng)
        D_P = self.final_layer_activation(D_P)
        D_Q, vars_d = self.discriminate(y, params, vars, labels, dropout_rng)
        D_Q = self.final_layer_activation(D_Q)
        
        D_loss_Q = -jnp.mean(D_Q)
        D_loss_P = jnp.log(jnp.mean(jnp.power(D_P, (self.alpha - 1.0) / self.alpha))) / (self.alpha - 1.0)
        
        D_loss = D_loss_Q + D_loss_P + (jnp.log(self.alpha) + 1.0) / self.alpha
        return D_loss, vars_d
    
    def eval_var_formula_gen(self, x, params, vars, labels=None, dropout_rng=None):
        '''
        Evaluates the variational formula of Renyi divergence for the generator using the convex-conjugate variational formula.

        Args:
            x: Generated samples.
            params: Discriminator parameters.
            vars: Additional discriminator variables.
            labels: Optional input labels.
            dropout_rng: Optional dropout key for stochasticity.

        Returns:
            Tuple of generator loss and updated variables.
        '''
        G_Q, vars_d = self.discriminate(x, params, vars, labels, dropout_rng)
        G_Q = self.final_layer_activation(G_Q)
        G_loss = -jnp.mean(G_Q**((self.alpha - 1.0) / self.alpha))
        return G_loss, vars_d


class Renyi_Divergence_CC_rescaled(Renyi_Divergence_CC):
    '''
    Rescaled Renyi divergence class based on the rescaled convex-conjugate variational formula.
    alpha * R_alpha(P||Q), x~P, y~Q.
    '''

    def final_layer_activation(self, y):
        '''
        Final layer activation function to enforce positivity, scaled by alpha.

        Args:
            y: Output of the discriminator.

        Returns:
            Activated output, scaled by the alpha parameter.
        '''
        out = super().final_layer_activation(y)
        out = out / self.alpha
        
        return out

    def eval_var_formula(self, x, y, params, vars, labels=None, dropout_rng=None):
        '''
        Evaluates the variational formula of the rescaled Renyi divergence.

        Args:
            x: Samples from distribution P.
            y: Samples from distribution Q.
            params: Discriminator parameters.
            vars: Additional discriminator variables.
            labels: Optional input labels.
            dropout_rng: Optional dropout key for stochasticity.

        Returns:
            Tuple of rescaled Renyi divergence loss and updated variables.
        '''
        D_loss, vars_d = super().eval_var_formula(x, y, params, vars, labels, dropout_rng)
        D_loss = D_loss * self.alpha
        
        return D_loss, vars_d
    
    def eval_var_formula_gen(self, x, params, vars, labels=None, dropout_rng=None):
        '''
        Evaluates the variational formula for the generator of the rescaled Renyi divergence.

        Args:
            x: Generated samples.
            params: Discriminator parameters.
            vars: Additional discriminator variables.
            labels: Optional input labels.
            dropout_rng: Optional dropout key for stochasticity.

        Returns:
            Tuple of generator loss and updated variables.
        '''
        G_loss, vars_d = super().eval_var_formula_gen(x, params, vars, labels, dropout_rng)
        G_loss = G_loss * self.alpha
        return G_loss, vars_d


class Renyi_Divergence_WCR(Renyi_Divergence_CC):
    '''
    Rescaled Renyi divergence class as alpha approaches infinity (worst-case regret divergence).
    Dinfty(P||Q), x~P, y~Q.
    '''

    def eval_var_formula(self, x, y, params, vars, labels=None, dropout_rng=None):
        '''
        Evaluates the variational formula of the Renyi divergence class as alpha approaches infinity (worst-case regret divergence).

        Args:
            x: Samples from distribution P.
            y: Samples from distribution Q.
            params: Discriminator parameters.
            vars: Additional discriminator variables.
            labels: Optional input labels.
            dropout_rng: Optional dropout key for stochasticity.

        Returns:
            Tuple of worst-case regret divergence loss and updated variables.
        '''
        D_P, vars_d = self.discriminate(x, params, vars, labels, dropout_rng)
        D_P = self.final_layer_activation(D_P)
        D_Q, vars_d = self.discriminate(y, params, vars, labels, dropout_rng)
        D_Q = self.final_layer_activation(D_Q)
        
        D_loss_P = jnp.log(jnp.mean(D_P))
        D_loss_Q = -jnp.mean(D_Q)
        
        D_loss = D_loss_Q + D_loss_P + 1.0
        return D_loss, vars_d
    
    def eval_var_formula_gen(self, x, params, vars, labels=None, dropout_rng=None):
        '''
        Evaluates the variational formula for the generator of the worst-case regret divergence.

        Args:
            x: Generated samples.
            params: Discriminator parameters.
            vars: Additional discriminator variables.
            labels: Optional input labels.
            dropout_rng: Optional dropout key for stochasticity.

        Returns:
            Tuple of generator loss and updated variables.
        '''
        G_Q, vars_d = self.discriminate(x, params, vars, labels, dropout_rng)
        G_Q = self.final_layer_activation(G_Q)
        G_loss = -jnp.mean(G_Q)
        return G_loss, vars_d


class Discriminator_Penalty():
    '''
    Base class for implementing penalties on the discriminator during training.
    Enables the implementation of discriminator constraints to regularize the divergence objective.
    '''

    def __init__(self, penalty_weight):
        '''
        Initializes the Discriminator Penalty class.

        Args:
            penalty_weight: Weight of the penalty term applied to the divergence objective.
        '''
        self.penalty_weight = penalty_weight
    
    def get_penalty_weight(self):
        '''
        Returns the weight of the penalty.

        Returns:
            Penalty weight.
        '''
        return self.penalty_weight

    def set_penalty_weight(self, weight):
        '''
        Sets the weight of the penalty.

        Args:
            weight: New penalty weight.
        '''
        self.penalty_weight = weight
    
    def evaluate(self, discriminator, x, y, params, batch_stats, key, labels=None, dropout_rng=None):
        '''
        Evaluates the penalty term. Should be overridden by subclasses.

        Args:
            discriminator: Discriminator model.
            x: Samples from distribution P.
            y: Samples from distribution Q.
            params: Discriminator parameters.
            batch_stats: Additional statistics for the batch.
            key: Random key for JAX RNG.
            labels: Optional input labels.
            dropout_rng: Optional dropout key for stochasticity.

        Returns:
            None. (Subclasses should implement specific penalty evaluations.)
        '''
        return None


class Gradient_Penalty_1Sided(Discriminator_Penalty):
    '''
    One-sided gradient penalty to enforce a constraint: Lipschitz constant <= Lip_const.
    '''

    def __init__(self, penalty_weight, Lip_const):
        '''
        Initializes the one-sided gradient penalty class.

        Args:
            penalty_weight: Weight of the gradient penalty term.
            Lip_const: Target Lipschitz constant.
        '''
        super().__init__(penalty_weight)
        self.Lip_const = Lip_const

    def get_Lip_constant(self):
        '''
        Returns the target Lipschitz constant.

        Returns:
            Lipschitz constant.
        '''
        return self.Lip_const

    def set_Lip_constant(self, L):
        '''
        Sets the target Lipschitz constant.

        Args:
            L: New Lipschitz constant.
        '''
        self.Lip_const = L
    
    def evaluate(self, discriminator, x, y, params, batch_stats, key, labels=None, dropout_rng=None):
        '''
        Computes the one-sided gradient penalty to enforce the Lipschitz constant constraint.

        Args:
            discriminator: Discriminator model.
            x: Samples from distribution P.
            y: Samples from distribution Q.
            params: Discriminator parameters.
            batch_stats: Additional statistics for the batch.
            key: Random key for JAX RNG.
            labels: Optional input labels.
            dropout_rng: Optional dropout key for stochasticity.

        Returns:
            One-sided gradient penalty value.
        '''
        batch_size = x.shape[0]
        rng = jax.random.PRNGKey(0)
        ratio = jax.random.uniform(rng, shape=(batch_size,) + (1,) * (x.ndim - 1))

        # Interpolate between real and fake samples
        interpltd = x + ratio * (y - x)

        def discriminator_fn(interpltd):
            if batch_stats is not None:
                if labels is not None:
                    if dropout_rng is not None:
                        return discriminator.apply({'params': params, 'batch_stats': batch_stats}, interpltd, labels, rngs={'dropout': dropout_rng})
                    else:
                        return discriminator.apply({'params': params, 'batch_stats': batch_stats}, interpltd, labels)
                else:
                    if dropout_rng is not None:
                        return discriminator.apply({'params': params, 'batch_stats': batch_stats}, interpltd, rngs={'dropout': dropout_rng})
                    else:
                        return discriminator.apply({'params': params, 'batch_stats': batch_stats}, interpltd)
            else:
                if labels is not None:
                    if dropout_rng is not None:
                        return discriminator.apply({'params': params}, interpltd, labels, rngs={'dropout': dropout_rng})
                    else:
                        return discriminator.apply({'params': params}, interpltd, labels)
                else:
                    if dropout_rng is not None:
                        return discriminator.apply({'params': params}, interpltd, rngs={'dropout': dropout_rng})
                    else:
                        return discriminator.apply({'params': params}, interpltd)

        # Calculate gradients with respect to the interpolated samples
        grads = jax.grad(lambda interpltd: jnp.sum(discriminator_fn(interpltd)))(interpltd)

        # Calculate the norm of the gradients
        grads = grads.reshape((grads.shape[0], -1))
        norm_squared = jnp.sum(grads ** 2, axis=1)

        # Compute the one-sided gradient penalty
        gp = self.penalty_weight * jnp.mean(jnp.clip(norm_squared / self.Lip_const**2 - 1, a_min=0.0))
        return gp


class Gradient_Penalty_2Sided(Discriminator_Penalty):
    '''
    Two-sided gradient penalty to enforce a constraint: Lipschitz constant = Lip_const.
    '''

    def __init__(self, penalty_weight, Lip_const):
        '''
        Initializes the two-sided gradient penalty class.

        Args:
            penalty_weight: Weight of the gradient penalty term.
            Lip_const: Target Lipschitz constant.
        '''
        super().__init__(penalty_weight)
        self.Lip_const = Lip_const

    def get_Lip_constant(self):
        '''
        Returns the target Lipschitz constant.

        Returns:
            Lipschitz constant.
        '''
        return self.Lip_const

    def set_Lip_constant(self, L):
        '''
        Sets the target Lipschitz constant.

        Args:
            L: New Lipschitz constant.
        '''
        self.Lip_const = L
    
    def evaluate(self, discriminator, x, y, params, labels=None, dropout_rng=None):
        '''
        Computes the two-sided gradient penalty to enforce the Lipschitz constant constraint.

        Args:
            discriminator: Discriminator model.
            x: Samples from distribution P.
            y: Samples from distribution Q.
            params: Discriminator parameters.
            labels: Optional input labels.
            dropout_rng: Optional dropout key for stochasticity.

        Returns:
            Two-sided gradient penalty value.
        '''
        batch_size = x.shape[0]
        rng = jax.random.PRNGKey(0)
        ratio = jax.random.uniform(rng, shape=(batch_size,) + (1,) * (x.ndim - 1))

        # Interpolate between real and fake samples
        interpltd = x + ratio * (y - x)

        def compute_d_pred(interpltd):
            if labels is not None:
                if dropout_rng is not None:
                    return discriminator.apply({"params": params}, interpltd, labels, rngs={'dropout': dropout_rng})
                else:
                    return discriminator.apply({"params": params}, interpltd, labels)
            else:
                if dropout_rng is not None:
                    return discriminator.apply({"params": params}, interpltd, rngs={'dropout': dropout_rng})
                else:
                    return discriminator.apply({'params': params}, interpltd)

        D_pred = compute_d_pred(interpltd)

        # Compute gradients with respect to the interpolated samples
        grads = grad(lambda interpltd: jnp.sum(compute_d_pred(interpltd)))(interpltd)

        # Calculate the norm of the gradients
        norm = jnp.sqrt(jnp.sum(jnp.square(grads).reshape(batch_size, -1), axis=1))

        # Compute the two-sided gradient penalty
        gp = self.penalty_weight * jnp.mean(jnp.square(norm - self.Lip_const))
        return gp


class DataLoader:
    '''
    DataLoader class for loading and batching data during training.
    '''

    def __init__(self, data, batch_size, shuffle=True):
        '''
        Initializes the DataLoader.

        Args:
            data: Input dataset.
            batch_size: Size of each batch.
            shuffle: Whether to shuffle the data before loading batches.
        '''
        self.data = data
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_samples = len(data)
        self.index = jnp.arange(self.num_samples)
        if shuffle:
            self.index = jax.random.permutation(jax.random.PRNGKey(0), self.index)

    def __iter__(self):
        '''
        Initializes the iterator for batching the data.
        '''
        self.current_idx = 0
        return self

    def __next__(self):
        '''
        Returns the next batch of data.

        Raises:
            StopIteration: If there are no more batches to return.

        Returns:
            A batch of data.
        '''
        if self.current_idx >= self.num_samples:
            raise StopIteration
        batch_idx = self.index[self.current_idx:self.current_idx+self.batch_size]
        batch = jnp.take(self.data, batch_idx, axis=0)
        self.current_idx += self.batch_size
        return batch