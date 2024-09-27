import tensorflow as tf
import math
import time
from tqdm import tqdm


class Divergence(tf.keras.Model):
    '''
    Divergence D(P||Q) between random variables x~P, y~Q.
    Parent class where common parameters and functions are defined.
    '''

    def __init__(self, discriminator, disc_optimizer, epochs, batch_size, discriminator_penalty=None):
        '''
        Initializes the Divergence class.

        Args:
            discriminator: The discriminator neural network.
            disc_optimizer: Optimizer for the discriminator.
            epochs: Number of training epochs.
            batch_size: Size of each batch.
            discriminator_penalty: Optional penalty applied to the discriminator loss.
        '''
        super(Divergence, self).__init__()
        self.batch_size = batch_size
        self.epochs = epochs
        self.discriminator = discriminator
        self.disc_optimizer = disc_optimizer
        self.discriminator_penalty = discriminator_penalty

    def __repr__(self):
        ''' Returns a string representation of the discriminator model. '''
        return 'discriminator: {}'.format(self.discriminator)

    def discriminate(self, x, labels=None):
        '''
        Discriminates between samples from distributions P and Q.

        Args:
            x: Input data to discriminate.
            labels: Optional labels for the input data.

        Returns:
            Discriminator output.
        '''
        if labels is not None:
            y = self.discriminator([x, labels])
        else:
            y = self.discriminator(x)
        return y

    def eval_var_formula(self, x, y, labels=None):
        '''
        Evaluates the variational formula for the divergence measure.
        Should be implemented by subclasses.

        Args:
            x: Samples from distribution P.
            y: Samples from distribution Q.
            labels: Optional labels for the input data.

        Returns:
            None (to be overridden by subclasses).
        '''
        return None

    def estimate(self, x, y, labels=None):
        '''
        Estimates the divergence measure.

        Args:
            x: Samples from distribution P.
            y: Samples from distribution Q.
            labels: Optional labels for the input data.

        Returns:
            Estimated divergence loss.
        '''
        divergence_loss = self.eval_var_formula(x, y, labels)
        return divergence_loss

    def generator_loss(self, x, labels=None):
        '''
        Computes the generator loss.

        Args:
            x: Generated samples.
            labels: Optional labels for the input data.

        Returns:
            Generator loss.
        '''
        generator_loss = self.eval_var_formula_gen(x, labels)
        return generator_loss

    def discriminator_loss(self, x, y, labels=None):
        '''
        Computes the discriminator loss.

        Args:
            x: Samples from distribution P.
            y: Samples from distribution Q.
            labels: Optional labels for the input data.

        Returns:
            Discriminator loss.
        '''
        divergence_loss = self.eval_var_formula(x, y, labels)
        return divergence_loss

    def train_step(self, x, y, labels=None):
        '''
        Performs a training step for the discriminator.

        Args:
            x: Samples from distribution P.
            y: Samples from distribution Q.
            labels: Optional labels for the input data.

        Returns:
            Loss value for the current step.
        '''
        with tf.GradientTape() as disc_tape:
            loss = -self.discriminator_loss(x, y, labels)  # Negative to maximize discriminator loss

            if self.discriminator_penalty is not None:
                loss += self.discriminator_penalty.evaluate(self.discriminator, x, y, labels)

        gradients_of_loss = disc_tape.gradient(loss, self.discriminator.trainable_variables)
        self.disc_optimizer.apply_gradients(zip(gradients_of_loss, self.discriminator.trainable_variables))
        return loss

    def train(self, data_P, data_Q, labels=None, save_estimates=True):
        '''
        Trains the model for a number of epochs.

        Args:
            data_P: Data samples from distribution P.
            data_Q: Data samples from distribution Q.
            labels: Optional labels for the input data.
            save_estimates: Whether to save divergence estimates.

        Returns:
            A list of divergence estimates for each epoch.
        '''
        P_dataset = tf.data.Dataset.from_tensor_slices(data_P).shuffle(buffer_size=data_P.shape[0]).batch(self.batch_size)
        Q_dataset = tf.data.Dataset.from_tensor_slices(data_Q).shuffle(buffer_size=data_Q.shape[0]).batch(self.batch_size)

        estimates = []
        for epoch in tqdm(range(self.epochs), desc='Epochs'):
            for P_batch, Q_batch in zip(P_dataset, Q_dataset):
                self.train_step(P_batch, Q_batch, labels)

            if save_estimates:
                estimates.append(float(self.estimate(P_batch, Q_batch, labels)))

        return estimates

    def get_discriminator(self):
        ''' Returns the discriminator model. '''
        return self.discriminator

    def set_discriminator(self, discriminator):
        ''' Sets a new discriminator model. '''
        self.discriminator = discriminator

    def get_no_epochs(self):
        ''' Returns the number of training epochs. '''
        return self.epochs

    def set_no_epochs(self, epochs):
        ''' Sets the number of training epochs. '''
        self.epochs = epochs

    def get_batch_size(self):
        ''' Returns the batch size. '''
        return self.batch_size

    def set_batch_size(self, BATCH_SIZE):
        ''' Sets the batch size. '''
        self.batch_size = BATCH_SIZE

    def get_learning_rate(self):
        ''' Returns the learning rate. '''
        return self.learning_rate

    def set_learning_rate(self, lr):
        ''' Sets the learning rate. '''
        self.learning_rate = lr


class IPM(Divergence):
    '''
    Integral Probability Metric (IPM) class for evaluating IPMs.
    '''

    def eval_var_formula(self, x, y, labels=None):
        '''
        Evaluates the variational formula for IPM.

        Args:
            x: Samples from distribution P.
            y: Samples from distribution Q.
            labels: Optional labels for the input data.

        Returns:
            IPM loss.
        '''
        D_P = self.discriminate(x, labels)
        D_Q = self.discriminate(y, labels)

        D_loss_P = tf.reduce_mean(D_P)
        D_loss_Q = tf.reduce_mean(D_Q)

        D_loss = D_loss_P - D_loss_Q
        return D_loss

    def eval_var_formula_gen(self, x, labels=None):
        '''
        Evaluates the variational formula for IPM applied to the generator.

        Args:
            x: Generated samples.
            labels: Optional labels for the input data.

        Returns:
            Generator loss based on IPM.
        '''
        G_Q = self.discriminate(x, labels)
        G_loss_Q = -tf.reduce_mean(G_Q)
        return G_loss_Q


class f_Divergence(Divergence):
    '''
    f-divergence class, parent class for f-divergence-based measures D_f(P||Q).
    Subclasses must implement the Legendre transform f_star.
    '''

    def f_star(self, y):
        ''' Placeholder for the Legendre transform of f. Should be implemented by subclasses. '''
        return None
    
    def final_layer_activation(self, y):
        ''' Applies the final layer activation. '''
        return y

    def eval_var_formula(self, x, y, labels=None):
        '''
        Evaluates the variational formula for f-divergence.

        Args:
            x: Samples from distribution P.
            y: Samples from distribution Q.
            labels: Optional labels for the input data.

        Returns:
            f-divergence loss.
        '''
        D_P = self.discriminate(x, labels)
        D_Q = self.discriminate(y, labels)
            
        D_P = self.final_layer_activation(D_P)
        D_Q = self.final_layer_activation(D_Q)
        
        D_loss_P = tf.reduce_mean(D_P)
        D_loss_Q = tf.reduce_mean(self.f_star(D_Q))
        D_loss = D_loss_P - D_loss_Q
        return D_loss 
    
    def eval_var_formula_gen(self, x, labels=None):
        '''
        Evaluates the variational formula for f-divergence applied to the generator.

        Args:
            x: Generated samples.
            labels: Optional labels for the input data.

        Returns:
            Generator loss based on f-divergence.
        '''
        G_Q = self.discriminate(x, labels)
        G_Q = self.final_layer_activation(G_Q)
        G_loss_Q = -tf.reduce_mean(self.f_star(G_Q + 1e-8))
        return G_loss_Q


class KLD_LT(f_Divergence):
    '''
    Kullback-Leibler (KL) divergence class based on the Legendre transform.
    KL(P||Q), x~P, y~Q.
    '''

    def f_star(self, y):
        ''' Legendre transform of f(y) = y * log(y). '''
        f_star_y = tf.math.exp(y - 1)
        return f_star_y


class Pearson_chi_squared_LT(f_Divergence):
    '''
    Pearson chi-squared divergence class based on the Legendre transform.
    chi^2(P||Q), x~P, y~Q.
    '''

    def f_star(self, y):
        ''' Legendre transform of f(y) = (y - 1)^2. '''
        f_star_y = 0.25 * tf.math.pow(y, 2.0) + y
        return f_star_y


class squared_Hellinger_LT(f_Divergence):
    '''
    Squared Hellinger distance class based on the Legendre transform.
    H(P||Q), x~P, y~Q.
    '''

    def f_star(self, y):
        ''' Legendre transform of f(y) = (sqrt(y) - 1)^2. '''
        f_star_y = y / (1 - y)
        return f_star_y

    def final_layer_activation(self, y):
        ''' Applies the final layer activation for squared Hellinger distance. '''
        out = 1.0 - tf.math.exp(-y)
        return out


class Jensen_Shannon_LT(f_Divergence):
    '''
    Jensen-Shannon divergence class based on the Legendre transform.
    JS(P||Q), x~P, y~Q.
    '''

    def f_star(self, y):
        ''' Legendre transform of f(y) = y * log(y) - (y + 1) * log((y + 1) / 2). '''
        f_star_y = -tf.math.log(2.0 - tf.math.exp(y))
        return f_star_y

    def final_layer_activation(self, y):
        ''' Applies the final layer activation for Jensen-Shannon divergence. '''
        out = -tf.math.log(0.5 + 0.5 * tf.math.exp(-y))
        return out


class alpha_Divergence_LT(f_Divergence):
    '''
    Alpha-divergence class based on the Legendre transform.
    D_{f_alpha}(P||Q), x~P, y~Q.
    '''

    def __init__(self, discriminator, disc_optimizer, alpha, epochs, batch_size, discriminator_penalty=None):
        '''
        Initializes the alpha-divergence class.

        Args:
            discriminator: Discriminator model.
            disc_optimizer: Optimizer for the discriminator.
            alpha: Order of the alpha-divergence.
            epochs: Number of training epochs.
            batch_size: Size of each batch.
            discriminator_penalty: Optional penalty applied to the discriminator.
        '''
        super().__init__(discriminator, disc_optimizer, epochs, batch_size, discriminator_penalty)
        self.alpha = alpha

    def get_order(self):
        ''' Returns the order of the alpha-divergence. '''
        return self.alpha

    def set_order(self, alpha):
        ''' Sets the order of the alpha-divergence. '''
        self.alpha = alpha
    
    def f_star(self, y):
        ''' Legendre transform of f_alpha based on the alpha value. '''
        if self.alpha > 1.0:
            f_star_y = ((self.alpha - 1.0) * tf.nn.relu(y))**(self.alpha / (self.alpha - 1.0)) / self.alpha + 1.0 / (self.alpha * (self.alpha - 1.0))
        elif 0.0 < self.alpha < 1.0:
            f_star_y = tf.math.pow((1.0 - self.alpha) * tf.nn.relu(y), self.alpha / (self.alpha - 1.0)) / self.alpha - 1.0 / (self.alpha * (self.alpha - 1.0))
        return f_star_y


class Pearson_chi_squared_HCR(Divergence):
    '''
    Pearson chi-squared divergence class based on the Hammersley-Chapman-Robbins bound.
    chi^2(P||Q), x~P, y~Q.
    '''

    def eval_var_formula(self, x, y, labels=None):
        '''
        Evaluates the variational formula for Pearson chi-squared divergence based on the Hammersley-Chapman-Robbins bound.

        Args:
            x: Samples from distribution P.
            y: Samples from distribution Q.
            labels: Optional labels for the input data.

        Returns:
            Pearson chi-squared divergence loss.
        '''
        D_P = self.discriminate(x, labels)
        D_Q = self.discriminate(y, labels)

        D_loss_P = tf.reduce_mean(D_P)
        D_loss_Q = tf.reduce_mean(D_Q)

        D_loss = (D_loss_P - D_loss_Q)**2 / tf.math.reduce_variance(D_Q)
        return D_loss

    def eval_var_formula_gen(self, x, labels=None):
        ''' Evaluates the generator's objective based on Pearson chi-squared divergence. '''
        G_Q = self.discriminate(x, labels)
        G_loss_Q = -tf.reduce_mean(G_Q)
        return G_loss_Q


class KLD_DV(Divergence):
    '''
    KL divergence class based on the Donsker-Varadhan variational formula.
    KL(P||Q), x~P, y~Q.
    '''

    def eval_var_formula(self, x, y, labels=None):
        '''
        Evaluates the variational formula for KL divergence.

        Args:
            x: Samples from distribution P.
            y: Samples from distribution Q.
            labels: Optional labels for the input data.

        Returns:
            KL divergence loss.
        '''
        D_P = self.discriminate(x, labels)
        D_Q = self.discriminate(y, labels)

        D_loss_P = tf.reduce_mean(D_P)
        
        max_val = tf.reduce_max(D_Q)
        D_loss_Q = tf.math.log(tf.reduce_mean(tf.math.exp(D_Q - max_val))) + max_val
        D_loss = D_loss_P - D_loss_Q
        return D_loss

    def eval_var_formula_gen(self, x, labels=None):
        ''' Evaluates the generator's objective based on KL divergence. '''
        G_Q = self.discriminate(x, labels)

        max_val = tf.reduce_max(G_Q)
        G_loss = -(tf.math.log(tf.reduce_mean(tf.math.exp(G_Q - max_val))) + max_val)
        return G_loss


class Renyi_Divergence(Divergence):
    '''
    Renyi divergence class for computing Renyi divergence R_alpha(P||Q), x~P, y~Q.
    '''

    def __init__(self, discriminator, disc_optimizer, alpha, epochs, batch_size, discriminator_penalty=None):
        '''
        Initializes the Renyi divergence class.

        Args:
            discriminator: Discriminator model.
            disc_optimizer: Optimizer for the discriminator.
            alpha: Order of the Renyi divergence.
            epochs: Number of training epochs.
            batch_size: Size of each batch.
            discriminator_penalty: Optional penalty applied to the discriminator.
        '''
        super().__init__(discriminator, disc_optimizer, epochs, batch_size, discriminator_penalty)
        self.alpha = alpha

    def get_order(self):
        ''' Returns the order of the Renyi divergence. '''
        return self.alpha

    def set_order(self, alpha):
        ''' Sets the order of the Renyi divergence. '''
        self.alpha = alpha


class Renyi_Divergence_DV(Renyi_Divergence):
    '''
    Renyi divergence class based on the Renyi-Donsker-Varadhan variational formula.
    R_alpha(P||Q), x~P, y~Q.
    '''

    def eval_var_formula(self, x, y, labels=None):
        '''
        Evaluates the variational formula for Renyi divergence based on the Renyi-Donsker-Varadhan formula.

        Args:
            x: Samples from distribution P.
            y: Samples from distribution Q.
            labels: Optional labels for the input data.

        Returns:
            Renyi divergence loss.
        '''
        gamma = self.alpha
        beta = 1.0 - self.alpha

        D_P = self.discriminate(x, labels)
        D_Q = self.discriminate(y, labels)

        if beta == 0.0:
            D_loss_P = tf.reduce_mean(D_P)
        else:
            max_val = tf.reduce_max((-beta) * D_P)
            D_loss_P = -(1.0 / beta) * (tf.math.log(tf.reduce_mean(tf.math.exp((-beta) * D_P - max_val))) + max_val)

        if gamma == 0.0:
            D_loss_Q = tf.reduce_mean(D_Q)
        else:
            max_val = tf.reduce_max(gamma * D_Q)
            D_loss_Q = (1.0 / gamma) * (tf.math.log(tf.reduce_mean(tf.math.exp(gamma * D_Q - max_val))) + max_val)

        D_loss = D_loss_P - D_loss_Q
        return D_loss

    def eval_var_formula_gen(self, x, labels=None):
        ''' Evaluates the generator's objective based on Renyi divergence. '''
        gamma = self.alpha

        D_Q = self.discriminate(x, labels)

        if gamma == 0.0:
            G_loss = -tf.reduce_mean(D_Q)
        else:
            max_val = tf.reduce_max(gamma * D_Q)
            G_loss = -(1.0 / gamma) * (tf.math.log(tf.reduce_mean(tf.math.exp(gamma * D_Q - max_val))) + max_val)

        return G_loss


class Renyi_Divergence_CC(Renyi_Divergence):
    '''
    Renyi divergence class based on the convex-conjugate variational formula.
    R_alpha(P||Q), x~P, y~Q.
    '''

    def __init__(self, discriminator, disc_optimizer, alpha, epochs, batch_size, final_act_func, discriminator_penalty=None):
        '''
        Initializes the Renyi divergence class based on the convex-conjugate variational formula.

        Args:
            discriminator: Discriminator model.
            disc_optimizer: Optimizer for the discriminator.
            alpha: Order of the Renyi divergence.
            epochs: Number of training epochs.
            batch_size: Size of each batch.
            final_act_func: Final activation function for positive output.
            discriminator_penalty: Optional penalty applied to the discriminator.
        '''
        super().__init__(discriminator, disc_optimizer, alpha, epochs, batch_size, discriminator_penalty)
        self.final_act_func = final_act_func
    
    def final_layer_activation(self, y):
        ''' Applies the final layer activation to enforce positive values. '''
        if self.final_act_func == 'abs':
            out = tf.math.abs(y)
        elif self.final_act_func == 'softplus':
            out = tf.keras.activations.softplus(y)
        elif self.final_act_func == 'poly-softplus':
            out = 1.0 + (1.0 / (1.0 + tf.nn.relu(-y)) - 1.0) * (1.0 - tf.sign(y)) / 2.0 + y * (tf.sign(y) + 1.0) / 2.0
        return out
    
    def eval_var_formula(self, x, y, labels=None):
        '''
        Evaluates the variational formula for Renyi divergence based on the convex-conjugate formula.

        Args:
            x: Samples from distribution P.
            y: Samples from distribution Q.
            labels: Optional labels for the input data.

        Returns:
            Renyi divergence loss.
        '''
        D_P = self.discriminate(x, labels)
        D_Q = self.discriminate(y, labels)

        D_P = self.final_layer_activation(D_P)
        D_Q = self.final_layer_activation(D_Q)
        
        D_loss_P = -tf.reduce_mean(D_P)
        D_loss_Q = tf.math.log(tf.reduce_mean(D_Q**((self.alpha - 1.0) / self.alpha))) / (self.alpha - 1.0)
        
        D_loss = D_loss_P + D_loss_Q + (tf.math.log(self.alpha) + 1.0) / self.alpha
        return D_loss
    
    def eval_var_formula_gen(self, x, labels=None):
        ''' Evaluates the generator's objective based on Renyi divergence using the convex-conjugate formula. '''
        D_Q = self.discriminate(x, labels)
        D_Q = self.final_layer_activation(D_Q)

        G_loss = -tf.reduce_mean(D_Q**((self.alpha - 1.0) / self.alpha))
        
        return G_loss


class Renyi_Divergence_CC_rescaled(Renyi_Divergence_CC):
    '''
    Rescaled Renyi divergence class based on the rescaled convex-conjugate variational formula.
    alpha * R_alpha(P||Q), x~P, y~Q.
    '''

    def final_layer_activation(self, y):
        ''' Applies the final layer activation and scales it by alpha. '''
        out = super().final_layer_activation(y)
        out = out / self.alpha
        return out

    def eval_var_formula(self, x, y, labels=None):
        '''
        Evaluates the variational formula for the rescaled Renyi divergence.

        Args:
            x: Samples from distribution P.
            y: Samples from distribution Q.
            labels: Optional labels for the input data.

        Returns:
            Rescaled Renyi divergence loss.
        '''
        D_loss = super().eval_var_formula(x, y, labels)
        D_loss = D_loss * self.alpha
        return D_loss

    def eval_var_formula_gen(self, x, labels=None):
        ''' Evaluates the generator's objective based on rescaled Renyi divergence. '''
        G_loss = super().eval_var_formula_gen(x, labels)
        G_loss = G_loss * self.alpha
        return G_loss


class Renyi_Divergence_WCR(Renyi_Divergence_CC):
    '''
    Renyi divergence class as alpha approaches infinity (worst-case regret divergence).
    Dinfty(P||Q), where x ~ P and y ~ Q.
    '''

    def eval_var_formula(self, x, y, labels=None):
        '''
        Evaluates the variational formula for worst-case regret divergence as alpha approaches infinity.

        Args:
            x (tf.Tensor): Samples from distribution P.
            y (tf.Tensor): Samples from distribution Q.
            labels (tf.Tensor, optional): Optional labels for the input data.

        Returns:
            tf.Tensor: Worst-case regret divergence loss.
        '''
        D_P = self.discriminate(x, labels)
        D_Q = self.discriminate(y, labels)
            
        D_P = self.final_layer_activation(D_P)
        D_Q = self.final_layer_activation(D_Q)
        
        D_loss_P = tf.math.log(tf.reduce_mean(D_P))
        D_loss_Q = -tf.reduce_mean(D_Q)
        
        D_loss = D_loss_P + D_loss_Q + 1.0
        return D_loss
    
    def eval_var_formula_gen(self, X, labels=None):
        '''
        Evaluates the generator's objective based on worst-case regret divergence.

        Args:
            X (tf.Tensor): Samples from distribution Q.
            labels (tf.Tensor, optional): Optional labels for the input data.

        Returns:
            tf.Tensor: Generator's loss based on worst-case regret divergence.
        '''
        D_Q = self.discriminate(X, labels)
        D_Q = self.final_layer_activation(D_Q)

        G_loss = -tf.reduce_mean(D_Q)
        return G_loss

class Discriminator_Penalty():
    '''
    Discriminator penalty class penalizes the divergence objective functional during training.
    Allows for the (approximate) implementation of discriminator constraints.
    '''

    def __init__(self, penalty_weight):
        '''
        Initializes the Discriminator Penalty class.

        Args:
            penalty_weight: Weighting factor for the penalty term.
        '''
        self.penalty_weight = penalty_weight
    
    def get_penalty_weight(self):
        ''' Returns the weight of the penalty term. '''
        return self.penalty_weight

    def set_penalty_weight(self, weight):
        ''' Sets the weight of the penalty term. '''
        self.penalty_weight = weight
    
    def evaluate(self, discriminator, x, y, labels=None):
        '''
        Evaluates the penalty term. Should be overridden by subclasses.

        Args:
            discriminator: Discriminator model.
            x: Samples from distribution P.
            y: Samples from distribution Q.
            labels: Optional labels for the input data.

        Returns:
            None. (Subclasses should implement penalty evaluation.)
        '''
        return None


class Gradient_Penalty_1Sided(Discriminator_Penalty):
    '''
    One-sided gradient penalty class to enforce the Lipschitz constant <= Lip_const.
    '''

    def __init__(self, penalty_weight, Lip_const):
        '''
        Initializes the one-sided gradient penalty class.

        Args:
            penalty_weight: Weighting factor for the gradient penalty term.
            Lip_const: Target Lipschitz constant.
        '''
        super().__init__(penalty_weight)
        self.Lip_const = Lip_const

    def get_Lip_constant(self):
        ''' Returns the target Lipschitz constant. '''
        return self.Lip_const

    def set_Lip_constant(self, L):
        ''' Sets the target Lipschitz constant. '''
        self.Lip_const = L
    
    def evaluate(self, discriminator, x, y, labels=None):
        '''
        Computes the one-sided gradient penalty to enforce the Lipschitz constant <= Lip_const.

        Args:
            discriminator: Discriminator model.
            x: Samples from distribution P.
            y: Samples from distribution Q.
            labels: Optional labels for the input data.

        Returns:
            One-sided gradient penalty value.
        '''
        ratio = tf.random.uniform([tf.shape(x)[0]] + [1]*(x.ndim-1), dtype=x.dtype)

        # Interpolate between real and fake samples
        interpltd = x + ratio * (y - x)

        # Calculate gradients with respect to the interpolated samples
        with tf.GradientTape() as tape:
            tape.watch(interpltd)
            if labels is not None:
                D_pred = discriminator([interpltd, labels])
            else:
                D_pred = discriminator(interpltd)
        
        grads = tape.gradient(D_pred, interpltd)

        # Calculate the norm of the gradients
        norm_squared = tf.reduce_sum(tf.square(tf.reshape(grads, [tf.shape(grads)[0], -1])), axis=1)

        # Compute the one-sided gradient penalty
        gp = self.penalty_weight * tf.reduce_mean(tf.maximum(norm_squared / self.Lip_const**2 - 1, 0.0))
        return gp


class Gradient_Penalty_2Sided(Discriminator_Penalty):
    '''
    Two-sided gradient penalty class to enforce the Lipschitz constant = Lip_const.
    '''

    def __init__(self, penalty_weight, Lip_const):
        '''
        Initializes the two-sided gradient penalty class.

        Args:
            penalty_weight: Weighting factor for the gradient penalty term.
            Lip_const: Target Lipschitz constant.
        '''
        super().__init__(penalty_weight)
        self.Lip_const = Lip_const

    def get_Lip_constant(self):
        ''' Returns the target Lipschitz constant. '''
        return self.Lip_const

    def set_Lip_constant(self, L):
        ''' Sets the target Lipschitz constant. '''
        self.Lip_const = L
    
    def evaluate(self, discriminator, x, y, labels=None):
        '''
        Computes the two-sided gradient penalty to enforce the Lipschitz constant = Lip_const.

        Args:
            discriminator: Discriminator model.
            x: Samples from distribution P.
            y: Samples from distribution Q.
            labels: Optional labels for the input data.

        Returns:
            Two-sided gradient penalty value.
        '''
        ratio = tf.random.uniform([tf.shape(x)[0]] + [1]*(x.ndim-1), dtype=x.dtype)

        # Interpolate between real and fake samples
        interpltd = x + ratio * (y - x)

        # Calculate gradients with respect to the interpolated samples
        with tf.GradientTape() as tape:
            tape.watch(interpltd)
            if labels is not None:
                D_pred = discriminator([interpltd, labels])
            else:
                D_pred = discriminator(interpltd)
        
        grads = tape.gradient(D_pred, interpltd)

        # Calculate the norm of the gradients
        norm = tf.sqrt(tf.reduce_sum(tf.square(tf.reshape(grads, [tf.shape(grads)[0], -1])), axis=1))

        # Compute the two-sided gradient penalty
        gp = self.penalty_weight * tf.reduce_mean(tf.square(norm - self.Lip_const))
        return gp
