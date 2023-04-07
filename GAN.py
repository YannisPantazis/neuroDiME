import tensorflow as tf

'''
Class for training a GAN using one of the above divergences
If reverse_order=False the GAN works to minimize min_theta D(P||g_theta(Z)) where P is the distribution to be leared, Z is the noise source and g_theta is the generator (with parameters theta).
If reverse_order=True the GAN works to minimize min_theta D(g_theta(Z)||P) where P is the distribution to be leared, Z is the noise source and g_theta is the generator (with parameters theta).
'''
class GAN():
   # initialize
    def __init__(self, divergence, generator, noise_source, epochs, disc_steps_per_gen_step, reverse_order=False, include_penalty_in_gen_loss=False, batch_size=None):
        
        self.divergence = divergence # Variational divergence 
        self.generator = generator
        self.epochs = epochs
        self.disc_steps_per_gen_step = disc_steps_per_gen_step
        self.gen_optimizer = tf.keras.optimizers.Adam(divergence.learning_rate)
        self.reverse_order = reverse_order
        self.include_penalty_in_gen_loss = include_penalty_in_gen_loss
        self.noise_source = noise_source   # noise_source must be a function that takes a batch_size as input and outputs a batch of noise samples (that will be fed into the generator)
        if batch_size is None:
            self.batch_size=self.divergence.batch_size
        else:
            self.batch_size=batch_size   
        
    def estimate_loss(self, x, z):
        if self.reverse_order:
            data1 = self.generator(z)
            data2 = x 
        else:
            data1 = x
            data2 = self.generator(z)
                        
        return self.divergence.estimate(data1, data2)     
        
    def gen_train_step(self, x, z):
        # generator's parameters update
        with tf.GradientTape() as gen_tape:
            if self.reverse_order:
                data1 = self.generator(z)
                data2 = x
            else:
                data1 = x
                data2 = self.generator(z)
                
            loss = self.divergence.discriminator_loss(data1, data2)     
            if self.include_penalty_in_gen_loss and self.divergence.discriminator_penalty is not None:
                loss=loss-self.divergence.discriminator_penalty.evaluate(self.divergence.discriminator, data1, data2)

        gradients_of_loss = gen_tape.gradient(loss, self.generator.trainable_variables)
        self.gen_optimizer.apply_gradients(zip(gradients_of_loss, self.generator.trainable_variables))
        
        
    def disc_train_step(self, x, z):
        # generator's parameters update        
       
        if self.reverse_order:
            data1 = self.generator(z)
            data2 = x 
        else:
            data1= x
            data2 = self.generator(z)
                
        self.divergence.train_step(data1, data2)
        
     
    def train(self, data_P,  num_gen_samples_to_save=None, save_loss_estimates=False):
        # dataset slicing into minibatches
        P_dataset = tf.data.Dataset.from_tensor_slices(data_P)


        P_dataset=P_dataset.shuffle(buffer_size=data_P.shape[0], reshuffle_each_iteration=True)


        P_dataset=P_dataset.batch(self.batch_size)

        generator_samples=[]
        loss_estimates=[]
        for epoch in range(self.epochs):
            for P_batch in P_dataset:

                Z_batch=self.noise_source(self.batch_size)
            
                for disc_step in range(self.disc_steps_per_gen_step):
                    self.disc_train_step(P_batch,Z_batch)
            
                self.gen_train_step(P_batch, Z_batch)

            if num_gen_samples_to_save is not None:
                generator_samples.append(self.generate_samples(num_gen_samples_to_save))

            if save_loss_estimates:
                loss_estimates.append(float(self.estimate_loss(P_batch, Z_batch)))

        return generator_samples, loss_estimates
        
        
    def generate_samples(self, N_samples):    
        generator_samples = self.generator(self.noise_source(N_samples))
        return generator_samples.numpy()
    

