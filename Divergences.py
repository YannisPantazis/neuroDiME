import tensorflow as tf
import time


'''
Parent class where the common parameters and the common functions are defined
'''
class Divergence(tf.keras.Model):

    # initialize
    def __init__(self, discriminator, epochs, lr, BATCH_SIZE):
        super(Divergence, self).__init__()
        self.batch_size = BATCH_SIZE
        self.epochs = epochs
        self.learning_rate = lr

        self.discriminator = discriminator
        self.disc_optimizer = tf.keras.optimizers.Adam(self.learning_rate)

    def __repr__(self):
        return 'discriminator: {}'.format(self.discriminator)

    def discriminate(self, x): # g(x)
        y = self.discriminator(x)
        return y

    def eval_var_formula(self, x, y): # depends on the variational formula to be used
        return None

    def estimate(self, x, y): # same as self.eval_var_formula()
        divergence_loss = self.eval_var_formula(x, y)
        return divergence_loss

    def discriminator_loss(self, x, y): # same as self.estimate() (in principle)
        divergence_loss = self.eval_var_formula(x, y)
        return divergence_loss

    def train_step(self, x, y):
        # discriminator's parameters update
        with tf.GradientTape() as disc_tape:
            disc_loss = -self.discriminator_loss(x, y) # with minus because we maximize the discrimination loss

        gradients_of_disc = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)
        self.disc_optimizer.apply_gradients(zip(gradients_of_disc, self.discriminator.trainable_variables))

    def train(self, data_P, data_Q):
        # dataset slicing into minibatches
        P_dataset = tf.data.Dataset.from_tensor_slices(data_P).batch(self.batch_size)
        Q_dataset = tf.data.Dataset.from_tensor_slices(data_Q).batch(self.batch_size)

        for epoch in range(self.epochs):
            start = time.time()

            for P_batch, Q_batch in zip(P_dataset, Q_dataset):
                self.train_step(P_batch, Q_batch)

#            print('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))
#            print(float(self.estimate(P_batch, Q_batch)))
#            print()

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
        D_real = self.discriminate(x)
        D_fake = self.discriminate(y)

        D_loss_real = tf.reduce_mean(D_real)
        D_loss_fake = tf.reduce_mean(D_fake)

        D_loss = D_loss_real - D_loss_fake
        return D_loss


'''
f-divergence class (parent class)
'''
class f_Divergence(Divergence):
 
    # Legendre transform of f
    def f_star(self, y):
        return None
    
    def final_layer_activation(self, y):
        return y

    def eval_var_formula(self, x, y):
        D_real = self.discriminate(x)
        D_real = self.final_layer_activation(D_real)
        D_fake = self.discriminate(y)
        D_fake = self.final_layer_activation(D_fake)
        
        D_loss_real = tf.reduce_mean(D_real)
        D_loss_fake = tf.reduce_mean(self.f_star(D_fake))
        
        D_loss = D_loss_real - D_loss_fake
        return D_loss


'''
Kullback-Leibler (KL) divergence class (based on Legendre transform)
'''
class KLD_LT(f_Divergence):
 
    # Legendre transform of f(y)=y*log(y)
    def f_star(self, y):
        f_star_y = tf.math.exp(y-1)
        return f_star_y


'''
Pearson chi^2-divergence class (based on Legendre transform)
'''
class Pearson_chi_squared_LT(f_Divergence):
 
    # Legendre transform of f(y)=(y-1)^2
    def f_star(self, y):
        f_star_y = 0.25*tf.math.pow(y,2.0) + y
        return f_star_y


'''
squared Hellinger distance class (based on Legendre transform)
'''
class squared_Hellinger_LT(f_Divergence):
 
    # Legendre transform of f(y)=(sqrt(y)-1)^2
    def f_star(self, y):
        f_star_y = y / (1-y)
        return f_star_y

    def final_layer_activation(self, y):
        out = 1.0 - tf.math.exp(-y)
        return out


'''
Jensen-Shannon divergence class (based on Legendre transform)
'''
class Jensen_Shannon_LT(f_Divergence):
 
    # Legendre transform of f(y)=y*log(y)-(y+1)*log((y+1)/2)
    def f_star(self, y):
#        max_val = tf.reduce_max(y)
        f_star_y = -tf.math.log(2.0-tf.math.exp(y))
        return f_star_y

    def final_layer_activation(self, y):
        out = - tf.math.log(0.5 + 0.5*tf.math.exp(-y))
        return out


'''
alpha-divergence class (based on Legendre transform)
'''
class alpha_Divergence_LT(f_Divergence):
 
    # initialize
    def __init__(self, discriminator, alpha, epochs, lr, BATCH_SIZE):
        super(Divergence, self).__init__()
        
        Divergence.__init__(self, discriminator, epochs, lr, BATCH_SIZE)
        self.alpha = alpha # order
    
    def get_order(self):
        return self.alpha

    def set_order(self, alpha):
        self.alpha = alpha
    
    # Legendre transform of f_alpha
    def f_star(self, y):
        if self.alpha>1.0:
            f_star_y = ((self.alpha-1.0)*tf.nn.relu(y))**(self.alpha/(self.alpha-1.0)) / self.alpha + 1.0/(self.alpha*(self.alpha-1.0))
        elif (self.alpha<1.0) & (self.alpha>0.0):
            f_star_y = tf.math.pow((1.0-self.alpha)*tf.nn.relu(y), self.alpha/(self.alpha-1.0)) / self.alpha - 1.0/(self.alpha*(self.alpha-1.0))
        
        return f_star_y


'''
KL divergence class (based on the Donsker-Varahdan variational formula)
'''
class KLD_DV(Divergence):

    def eval_var_formula(self, x, y):
        D_real = self.discriminate(x)
        D_fake = self.discriminate(y)

        D_loss_real = tf.reduce_mean(D_real)
        
        max_val = tf.reduce_max(D_fake)
        D_loss_fake = tf.math.log(tf.reduce_mean(tf.math.exp(D_fake - max_val))) + max_val

        D_loss = D_loss_real - D_loss_fake
        return D_loss


'''
Renyi divergence class
'''
class Renyi_Divergence(Divergence):
 
    # initialize
    def __init__(self, discriminator, alpha, epochs, lr, BATCH_SIZE):
        super(Divergence, self).__init__()
        
        Divergence.__init__(self, discriminator, epochs, lr, BATCH_SIZE)
        self.alpha = alpha # RD order

    def get_order(self):
        return self.alpha

    def set_order(self, alpha):
        self.alpha = alpha


'''
Renyi divergence class (based on the Renyi-Donsker-Varahdan variational formula)
'''
class Renyi_Divergence_DV(Renyi_Divergence):

    def eval_var_formula(self, x, y):
        gamma = self.alpha
        beta = 1.0 - self.alpha

        D_real = self.discriminate(x)
        D_fake = self.discriminate(y)

        if beta == 0.0:
            D_loss_real = tf.reduce_mean(D_real)
        else:
            max_val = tf.reduce_max((-beta) * D_real)
            D_loss_real = -(1.0/beta) * (tf.math.log(tf.reduce_mean(tf.math.exp((-beta) * D_real - max_val))) + max_val)

        if gamma == 0.0:
            D_loss_fake = tf.reduce_mean(D_fake)
        else:
            max_val = tf.reduce_max((gamma) * D_fake)
            D_loss_fake = (1.0/gamma) * (tf.math.log(tf.reduce_mean(tf.math.exp(gamma * D_fake - max_val))) + max_val)

        D_loss = D_loss_real - D_loss_fake
        return D_loss


'''
Renyi divergence class (based on the convex-conjugate variational formula)
'''
class Renyi_Divergence_CC(Renyi_Divergence):
    
    # initialize
    def __init__(self, discriminator, alpha, epochs, lr, BATCH_SIZE, fl_act_func):
        super(Divergence, self).__init__()
        
        Renyi_Divergence.__init__(self, discriminator, alpha, epochs, lr, BATCH_SIZE)
        self.act_func = fl_act_func
    
    def final_layer_activation(self, y): # enforce positive values
        if self.act_func=='abs':
            out = tf.math.abs(y)
        elif self.act_func=='softplus':
            out = tf.keras.activations.softplus(y)
        elif self.act_func=='poly-softplus':
            out = 1.0+(1.0/(1.0+tf.nn.relu(-y))-1.0)*(1.0-tf.sign(y))/2.0 +y*(tf.sign(y)+1.0)/2.0
        
        return out
    
    def eval_var_formula(self, x, y):
        D_real = self.discriminate(x)
        D_real = self.final_layer_activation(D_real)
        D_fake = self.discriminate(y)
        D_fake = self.final_layer_activation(D_fake)
        
        D_loss_real = -tf.reduce_mean(D_real)
        D_loss_fake = tf.math.log(tf.reduce_mean(D_fake**((self.alpha-1.0)/self.alpha))) / (self.alpha-1.0)
        
        D_loss = D_loss_real + D_loss_fake + (tf.math.log(self.alpha)+1.0)/self.alpha
        return D_loss


'''
Rescaled Renyi divergence class (based on the rescaled convex-conjugate variational formula)
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
'''
class Renyi_Divergence_WCR(Renyi_Divergence_CC):
    
    def eval_var_formula(self, x, y):
        D_real = self.discriminate(x)
        D_real = self.final_layer_activation(D_real)
        D_fake = self.discriminate(y)
        D_fake = self.final_layer_activation(D_fake)
        
        D_loss_real = tf.math.log(tf.reduce_mean(D_real))
        D_loss_fake = -tf.reduce_mean(D_fake)
        
        D_loss = D_loss_real + D_loss_fake + 1.0
        return D_loss



'''
Divergence class with (one-sided) gradient penalty (enforce Lipschitz continuity)
'''
class Divergence_GP(Divergence):
 
    # initialize
    def __init__(self, discriminator, epochs, lr, BATCH_SIZE, L, gp_weight):
        super(Divergence, self).__init__()
        
        Divergence.__init__(self, discriminator, epochs, lr, BATCH_SIZE)
        self.Lip_const = L # Lipschitz constant
        self.gp_weight = gp_weight # weighting factor of gradient penalty

    def get_Lip_constant(self):
        return self.Lip_const

    def set_Lip_constant(self, L):
        self.Lip_const = L
    
    def get_gp_weight(self):
        return self.gp_weight

    def set_gp_weight(self, gp_weight):
        self.gp_weight = gp_weight
    
    def gradient_penalty_loss(self, x, y): # compute the gradient penalty
        temp_shape = [x.shape[0]] + [1 for _ in  range(len(x.shape)-1)]
        ratio = tf.random.uniform(temp_shape, 0.0, 1.0, dtype=tf.dtypes.float32)
        diff = y - x
        interpltd = x + (ratio * diff) # get the interpolated samples

        with tf.GradientTape() as gp_tape: # get the discriminator output
            gp_tape.watch(interpltd)
            D_pred = self.discriminator(interpltd, training=True)

        grads = gp_tape.gradient(D_pred, [interpltd])[0] # calculate the gradients
        if x.shape[1]==1: # calculate the norm
            norm = tf.sqrt(tf.square(grads))
        else:
            norm = tf.sqrt(tf.reduce_sum(tf.square(grads)))

        gp = tf.reduce_mean(tf.math.maximum(norm - self.Lip_const, 0.0)) # one-sided gradient penalty
#        gp = tf.reduce_mean((norm - self.Lip_const) ** 2) # two-sided gradient penalty
        return gp

    def train_step(self, x, y): # add gradient penalty to the discriminator's loss
        # discriminator's parameters update
        with tf.GradientTape() as disc_tape:
            disc_loss = -self.discriminator_loss(x, y) # with minus because we maximize the discrimination loss
            gp_loss = self.gradient_penalty_loss(x, y) # gradient penalty
            total_loss = disc_loss + gp_loss * self.gp_weight # add the gradient penalty to the original discriminator loss

        gradients_of_disc = disc_tape.gradient(total_loss, self.discriminator.trainable_variables)
        self.disc_optimizer.apply_gradients(zip(gradients_of_disc, self.discriminator.trainable_variables))



'''
KL divergence class with gradient penalty (based on the Donsker-Varahdan variational formula)
'''
class KLD_DV_GP(Divergence_GP):

    def eval_var_formula(self, x, y):
        D_real = self.discriminate(x)
        D_fake = self.discriminate(y)

        D_loss_real = tf.reduce_mean(D_real)
        
        max_val = tf.reduce_max(D_fake)
        D_loss_fake = tf.math.log(tf.reduce_mean(tf.math.exp(D_fake - max_val))) + max_val

        D_loss = D_loss_real - D_loss_fake
        return D_loss
        

'''
Wasserstein metric class with gradient penalty (enforce Lipschitz continuity)
'''
class Wasserstein_GP(Divergence_GP):

    def eval_var_formula(self, x, y):
        D_real = self.discriminate(x)
        D_fake = self.discriminate(y)

        D_loss_real = tf.reduce_mean(D_real)
        D_loss_fake = tf.reduce_mean(D_fake)

        D_loss = D_loss_real - D_loss_fake
        return D_loss


