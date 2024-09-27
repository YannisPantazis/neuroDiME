import os
import sys
import argparse
import time
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

current_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from models.GAN_CIFAR10_tf import *
from models.Divergences_tf import *

start = time.perf_counter()

fl_act_func_CC = 'poly-softplus' # abs, softplus, poly-softplus
optimizer = "Adam" #Adam, RMS
save_frequency = 10 #generator samples are saved every save_frequency epochs
num_gen_samples_to_save = 5000

def save_image(fake_images, path, normalize=False):
    fake_images = tf.stop_gradient(fake_images)  # Stop gradient tracking
    if normalize:
        fake_images = (fake_images + 1.0) / 2.0  # Normalize to [0, 1]
    fig = plt.figure(figsize=(8, 8))
    for i in range(min(fake_images.shape[0], 64)):  # Limit to first 64 images
        plt.subplot(8, 8, i + 1)
        plt.imshow(fake_images[i, :, :, :].numpy())
        plt.axis('off')
    plt.savefig(path)
    plt.close(fig)

# Convert the samples to a grid
def make_grid(images, nrow=4, padding=2):
    n, h, w, c = images.shape
    grid_height = h * nrow + padding * (nrow - 1)
    grid_width = w * nrow + padding * (nrow - 1)
    grid = np.ones((grid_height, grid_width, c))
    next_idx = 0
    
    for y in range(nrow):
        for x in range(nrow):
            if next_idx >= n:
                break
            img = images[next_idx]
            grid[y * (h + padding): y * (h + padding) + h, x * (w + padding): x * (w + padding) + w, :] = img
            next_idx += 1
        if next_idx >= n:
            break
    return grid

class GradientPenalty(Discriminator_Penalty):
    def __init__(self, gp_weight, L):
        super(GradientPenalty, self).__init__(gp_weight)
        self.L = L
    
    def get_Lip_constant(self):
        return self.L

    def set_Lip_constant(self, L):
        self.L = L
        
    def evaluate(self, discriminator, real_var, fake_var, labels):
        assert real_var.shape[0] == fake_var.shape[0], \
            'expected real and fake data to have the same batch size'

        batch_size = real_var.shape[0]
        alpha = tf.random.uniform([batch_size, 1, 1, 1])
        alpha = tf.tile(alpha, [1, tf.shape(real_var)[1], tf.shape(real_var)[2], tf.shape(real_var)[3]])

        alpha = tf.cast(alpha, tf.float32)
        real_var = tf.cast(real_var, tf.float32)
        fake_var = tf.cast(fake_var, tf.float32)
      
        interp_data = alpha * real_var + ((1 - alpha) * fake_var)
        interp_data = tf.Variable(interp_data, trainable=True)

        with tf.GradientTape() as tape:
            tape.watch(interp_data)
            disc_out = discriminator(interp_data)

        gradients = tape.gradient(disc_out, interp_data)

        gradients = tf.reshape(gradients, [batch_size, -1])
        gradient_penalty = self.get_penalty_weight() * tf.reduce_mean((tf.norm(gradients, ord=2, axis=1) - self.L) ** 2)

        return gradient_penalty
    

def main(mthd):
    # read input arguments
    parser = argparse.ArgumentParser(description='Neural-based Estimation of Divergences between Gaussians')
    parser.add_argument('--method', default='KLD-DV', type=str, metavar='method',
                        help='values: all, IPM, KLD-DV, KLD-LT, squared-Hel-LT, chi-squared-LT, JS-LT, alpha-LT, Renyi-DV, Renyi-CC, rescaled-Renyi-CC, Renyi-CC-WCR')
    parser.add_argument('--disc_steps_per_gen_step', default=3, type=int)
    parser.add_argument('--batch_size', default=256, type=int, metavar='m')
    parser.add_argument('--lr', default=2e-4, type=float)
    parser.add_argument('--epochs', default=100, type=int,
                        help='number of total epochs to run')
    parser.add_argument('--alpha', default=2.0, type=float, metavar='alpha')
    parser.add_argument('--Lip_constant', default=1.0, type=float, metavar='Lipschitz constant')
    parser.add_argument('--gp_weight', default=1.0, type=float, metavar='GP weight')
    parser.add_argument('--spectral_norm', choices=('True','False'), default='False')
    parser.add_argument('--bounded', choices=('True','False'), default='False')
    parser.add_argument('--reverse_order', choices=('True','False'), default='False')
    parser.add_argument('--use_GP', choices=('True','False'), default='False')
    parser.add_argument('--save_model', choices=('True','False'), default='False', type=str, metavar='save_model')
    parser.add_argument('--save_model_path', default='./trained_models/', type=str, metavar='save_model_path')
    parser.add_argument('--load_model', choices=('True','False'), default='False', type=str, metavar='load_model')  
    parser.add_argument('--load_model_path', default='trained_models/', type=str, metavar='load_model_path')
    parser.add_argument('--run_number', default=1, type=int, metavar='run_num')
    parser.add_argument('--conditional', choices=('True','False'), default='False', type=str, metavar='conditional')
    
    opt = parser.parse_args()
    opt_dict = vars(opt)
    print('parsed options:', opt_dict)

    # mthd = opt_dict['method']
    disc_steps_per_gen_step = opt_dict['disc_steps_per_gen_step']
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
    run_num=opt_dict['run_number']

    spec_norm = opt_dict['spectral_norm']=='True'
    bounded=opt_dict['bounded']=='True'
    reverse_order = opt_dict['reverse_order']=='True'
    use_GP = opt_dict['use_GP']=='True'
    save_model = opt_dict['save_model']=='True'
    save_model_path = opt_dict['save_model_path']
    load_model = opt_dict['load_model']=='True'
    load_model_path = opt_dict['load_model_path']
    conditional = opt_dict['conditional']=='True'
    
    print("Spectral_norm: "+str(spec_norm))
    print("Bounded: "+str(bounded))
    print("Reversed: "+str(reverse_order))
    print("Use Gradient Penalty: "+str(use_GP))
    
    (train_images, train_labels), (_, _) = tf.keras.datasets.cifar10.load_data()
    train_images = (train_images / 127.5) - 1.0
    train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels)).shuffle(60000, reshuffle_each_iteration=True).batch(m, drop_remainder=True)
    
    generator = Generator()
    discriminator = Discriminator()
    
    #construct optimizer
    if optimizer == 'Adam':
        disc_optimizer = tf.keras.optimizers.Adam(lr)
        gen_optimizer = tf.keras.optimizers.Adam(lr)

    if optimizer == 'RMS':
        disc_optimizer = tf.keras.optimizers.RMSprop(lr)
        gen_optimizer = tf.keras.optimizers.RMSprop(lr)
    
    # construct gradient penalty
    if use_GP:
        discriminator_penalty=GradientPenalty(gp_weight, L)
    else:
        discriminator_penalty=None
        
    discriminator.build(input_shape=(None, 32, 32, 3))
    generator.build(input_shape=(None, 124))
        
    discriminator.compile(optimizer=disc_optimizer)
    generator.compile(optimizer=gen_optimizer)


    # construct divergence
    if mthd=="IPM":
        div_dense = IPM(discriminator, disc_optimizer, epochs, m, discriminator_penalty)

    if mthd=="KLD-LT":
        div_dense = KLD_LT(discriminator, disc_optimizer, epochs, m, discriminator_penalty)
        
    if mthd=="KLD-DV":
        div_dense = KLD_DV(discriminator, disc_optimizer, epochs, m, discriminator_penalty)

    if mthd=="squared-Hel-LT":
        div_dense = squared_Hellinger_LT(discriminator, disc_optimizer, epochs, m, discriminator_penalty)

    if mthd=="chi-squared-LT":
        div_dense = Pearson_chi_squared_LT(discriminator, disc_optimizer, epochs, m, discriminator_penalty)

    if mthd=="chi-squared-HCR":
        div_dense = Pearson_chi_squared_HCR(discriminator, disc_optimizer, epochs, m, discriminator_penalty)

    if mthd=="JS-LT":
        div_dense = Jensen_Shannon_LT(discriminator, disc_optimizer, epochs, m, discriminator_penalty)    

    if mthd=="alpha-LT":
        div_dense = alpha_Divergence_LT(discriminator, disc_optimizer, alpha, epochs, m, discriminator_penalty)

    if mthd=="Renyi-DV":
        div_dense = Renyi_Divergence_DV(discriminator, disc_optimizer, alpha, epochs, m, discriminator_penalty)
        
    if mthd=="Renyi-CC":
        div_dense = Renyi_Divergence_CC(discriminator, disc_optimizer, alpha, epochs, m, fl_act_func_CC, discriminator_penalty)

    if mthd=="rescaled-Renyi-CC":
        div_dense = Renyi_Divergence_CC_rescaled(discriminator, disc_optimizer, alpha, epochs, m, fl_act_func_CC, discriminator_penalty)

    if mthd=="Renyi-CC-WCR":
        div_dense = Renyi_Divergence_WCR(discriminator, disc_optimizer, alpha, epochs, m, fl_act_func_CC, discriminator_penalty)

    if load_model:
        generated_images_path = f'generated_images_cifar10/'
        if not os.path.exists(generated_images_path):
            os.makedirs(generated_images_path)
   
        if conditional:
            load_model_path = f'{load_model_path}_conditional_CIFAR10'
        
        generator = tf.keras.models.load_model(f'{load_model_path}/generator_{mthd}.keras')
        generator.build(input_shape=(None, 124))
    
        noise = tf.random.normal([m, 128])
        fake_images = generator(noise)
        save_image(fake_images, f'{generated_images_path}/generated_images_{mthd}.png', normalize=True)
        print(f'Model loaded from {load_model_path}/generator_{mthd}')
    else:
        def noise_source(batch_size):
            return tf.random.normal([batch_size, 124])

        gen_losses = np.zeros(epochs)
        disc_losses = np.zeros(epochs)
        mean_scores = np.zeros(epochs)
        std_scores = np.zeros(epochs)
        iter = 0
        
        sample_images_path = ''
        if use_GP:
            if not os.path.exists(f'samples_{mthd}_GP_cifar10/'):
                os.makedirs(f'samples_{mthd}_GP_cifar10/')
            sample_images_path = f'samples_{mthd}_GP_cifar10/'
        else:
            if not os.path.exists(f'samples_{mthd}_cifar10/'):
                os.makedirs(f'samples_{mthd}_cifar10/')
            sample_images_path = f'samples_{mthd}_cifar10/'

        for epoch in tqdm(range(epochs), desc='Epochs'):
            d_loss = 0
            g_loss = 0
            for images, labels in train_dataset:
                labels = None
                # Train the discriminator
                noise = noise_source(m)
                samples = generator(noise)
                for _ in range(disc_steps_per_gen_step):
                    d_loss = div_dense.train_step(images, samples, labels)

                # Train the generator
                with tf.GradientTape() as tape:
                    samples_ = generator(noise)
                    g_loss = div_dense.generator_loss(samples_, labels)
                
                gradients = tape.gradient(g_loss, generator.trainable_variables)
                gen_optimizer.apply_gradients(zip(gradients, generator.trainable_variables))
                
                gen_losses[epoch] += g_loss.numpy()
                disc_losses[epoch] += d_loss.numpy()
                iter += 1
                if iter % 150 == 0:
                    fake = generator(noise)
                    if not os.path.exists(sample_images_path):
                        os.makedirs(sample_images_path)
                    save_image(fake, f'{sample_images_path}/sample_{epoch}.png', normalize=True)

            gen_losses[epoch] = g_loss.numpy() / len(train_dataset)
            disc_losses[epoch] = d_loss.numpy() / len(train_dataset)
            print('Epoch:', epoch, 'Discriminator Loss:', d_loss.numpy(), 'Generator Loss:', g_loss.numpy())  
            if math.isnan(d_loss.numpy()) or math.isnan(g_loss.numpy()):
                break


        # Save the loss vs epoch plot
        epoch_ax = np.arange(start=1, stop=epochs+1, step=1)
        _, ax = plt.subplots(1, 3, figsize=(15, 5))

        ax[0].plot(epoch_ax, disc_losses, color='blue')
        ax[0].set_xlim(1, epochs)
        ax[0].set_title("Discriminator Loss vs Epoch")
        ax[0].grid()

        ax[1].plot(epoch_ax, gen_losses, color='red')
        ax[1].set_xlim(1, epochs)
        ax[1].set_title("Generator Loss vs Epoch")
        ax[1].grid()

        ax[2].plot(epoch_ax, mean_scores, color='green', label='Inception Score Mean')
        ax[2].fill_between(epoch_ax, mean_scores-std_scores, mean_scores+std_scores, color='green', alpha=0.2, label='Inception Score Std')
        ax[2].set_xlim(1, epochs)
        ax[2].set_title("Inception Score Mean and Std vs Epoch")
        ax[2].grid()
        ax[2].legend()

        plt.tight_layout()
        
        if save_model:
            if conditional:
                save_model_path = f'{save_model_path}_conditional_CIFAR10'
                
            if not os.path.exists(save_model_path):
                os.makedirs(save_model_path)
                
            generator.save(f'{save_model_path}/generator_{mthd}.keras')    
            print(f'Model saved to {save_model_path}/generator_{mthd}')


        if use_GP:
            if not os.path.exists(f'losses_{mthd}_GP_cifar10/'):
                os.makedirs(f'losses_{mthd}_GP_cifar10/')

            plt.savefig(f'losses_{mthd}_GP_cifar10/loss_vs_epoch.png')
        else:
            if not os.path.exists(f'losses_{mthd}_cifar10/'):
                os.makedirs(f'losses_{mthd}_cifar10/')

            plt.savefig(f'losses_{mthd}_cifar10/loss_vs_epoch.png')
            plt.show()
            plt.close()    
        
        
    
if __name__ == '__main__':
    methods = ['IPM', 'KLD-DV', 'KLD-LT', 'squared-Hel-LT', 'chi-squared-LT', 'JS-LT', 'alpha-LT', 'Renyi-DV', 'Renyi-CC', 'rescaled-Renyi-CC', 'Renyi-CC-WCR']
    for method in methods:
        print(f'Running {method}...')
        main(method)