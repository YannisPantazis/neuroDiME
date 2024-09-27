import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
import tensorflow
from scipy.linalg import sqrtm
import json
import torch
from torchmetrics.image.fid import FrechetInceptionDistance
from tensorflow.keras.models import load_model

current_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from models.GAN_MNIST_tf import *
from models.Divergences_tf import *

start = time.perf_counter()

fl_act_func_CC = 'poly-softplus' # abs, softplus, poly-softplus
optimizer = "Adam" #Adam, RMS
save_frequency = 10 #generator samples are saved every save_frequency epochs
num_gen_samples_to_save = 5000

class GradientPenalty(Discriminator_Penalty):
    def __init__(self, weight, L):
        Discriminator_Penalty.__init__(self, weight)
        self.L = L

    def get_Lip_constant(self):
        return self.L

    def set_Lip_constant(self, L):
        self.L = L

    def evaluate(self, c, images, samples, y):
        assert images.shape == samples.shape
        batch_size = tf.shape(images)[0]
        jump = tf.random.uniform([batch_size, 1], 0.0, 1.0)
        jump_ = tf.reshape(jump, [batch_size, 1, 1, 1])
        interpolated = images * jump_ + (1 - jump_) * samples
        with tf.GradientTape() as tape:
            tape.watch(interpolated)
            c_ = c([interpolated, y])
        gradients = tape.gradient(c_, interpolated)
        gradients_norm = tf.sqrt(tf.reduce_sum(tf.square(gradients), axis=[1, 2, 3]))
        gradient_penalty = self.get_penalty_weight() * tf.reduce_mean((gradients_norm - self.L) ** 2)
        return gradient_penalty
    
def one_hot(test, i=10):
    return tf.one_hot(test, depth=i, dtype=tf.float32)

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

# Save the image grid
def save_image(grid, filename):
    plt.imshow(grid, cmap='gray')
    plt.axis('off')
    plt.savefig(filename, bbox_inches='tight', pad_inches=0)
    plt.close()


def generate_labeled_mnist(generator, num_rows=10):
    """
    Generates a grid of labeled MNIST digits, with each column corresponding to a digit 0-9, and each row containing different samples.
    
    Args:
        generator: The generator model used to create MNIST digits.
        gen_params: The parameters of the generator model.
        num_rows: The number of rows of samples for each digit (default is 10).
    
    Returns:
        grid: A grid of images where each column corresponds to a digit 0-9, and each row is a different sample.
    """
    # Number of classes (0-9 digits)
    num_classes = 10
    
    # Initialize an empty list to store images for each digit
    digit_images = []

    z = tf.random.normal([num_rows, 118])
    for digit in range(num_classes):
        # Create labels
        labels = np.array([digit] * num_rows, dtype=np.int32)
        labels_one_hot = tf.one_hot(labels, num_classes)
        
        # Generate images using the generator model
        generated_images = generator([labels_one_hot, z], training=False)
        
        # Convert generated images to numpy array
        digit_images.append(generated_images.numpy())
    
    # Stack the generated images vertically
    digit_images = np.vstack(digit_images)
    
    # Create a grid of images with num_rows rows and num_classes columns
    grid = make_grid(digit_images, nrow=num_classes)
    
    return grid


def generate_single_mnist(generator, num_rows=10):
    batch_size = 256
    z = tf.random.normal([batch_size, 118])
    labels = np.array([0] * batch_size, dtype=np.int32)
    labels = tf.one_hot(labels, 10)
    samples = generator([labels, z], training=False)
    idx = np.random.randint(0, samples.shape[0], 16)
    selected_samples = tf.gather(samples, idx)
    selected_samples = selected_samples.numpy()
    grid = make_grid(selected_samples)
    return grid


def main(mthd):
    # read input arguments
    parser = argparse.ArgumentParser(description='Neural-based Estimation of Divergences between Gaussians')
    parser.add_argument('--method', default='KLD-DV', type=str, metavar='method',
                        help='values: IPM, KLD-DV, KLD-LT, squared-Hel-LT, chi-squared-LT, JS-LT, alpha-LT, Renyi-DV, Renyi-CC, rescaled-Renyi-CC, Renyi-CC-WCR')
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
    parser.add_argument('--save_model_path', default='./trained_models', type=str, metavar='save_model_path')
    parser.add_argument('--load_model', choices=('True','False'), default='False', type=str, metavar='load_model')  
    parser.add_argument('--load_model_path', default='trained_models', type=str, metavar='load_model_path')
    parser.add_argument('--run_number', default=1, type=int, metavar='run_num')
    parser.add_argument('--conditional', choices=('True','False'), default='True', type=str, metavar='conditional')
    
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
    dataset = 'mnist'
    save_model = opt_dict['save_model']=='True'
    save_model_path = opt_dict['save_model_path']
    load_model_flag = opt_dict['load_model']=='True'
    load_model_path = opt_dict['load_model_path']
    conditional = opt_dict['conditional']=='True'
    
    print("Spectral_norm: "+str(spec_norm))
    print("Bounded: "+str(bounded))
    print("Reversed: "+str(reverse_order))
    print("Use Gradient Penalty: "+str(use_GP))
    
    (train_images, train_labels), (_, _) = tf.keras.datasets.mnist.load_data()
    train_images = train_images.astype('float32') / 255.0
    train_images = tf.expand_dims(train_images, axis=-1)  # Shape becomes (num_samples, 28, 28, 1)
    train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
    train_dataset = train_dataset.shuffle(60000, reshuffle_each_iteration=True).batch(batch_size=m, drop_remainder=True)
    
    if conditional:
        generator = Generator_MNIST_cond()
        discriminator = Discriminator_MNIST_cond()
    # else:
    #     generator = Generator_MNIST()
    #     discriminator = Discriminator_MNIST()
    
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

    discriminator.build(input_shape=(None, 28, 28, 1))
    generator.build(input_shape=(None, 10))
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

    def noise_source(batch_size):
        return tf.random.normal([batch_size, 118])
    
    fid_scores = []
    
    if load_model_flag:
        generated_images_path = f'generated_digits/'

        if conditional:
            load_model_path = f'{load_model_path}_conditional_MNIST'
            
        generator = load_model(f'{load_model_path}/generator_{mthd}.keras')

        grid = generate_labeled_mnist(generator, num_rows=10)
        # grid = generate_single_mnist(generator, num_rows=10)
        
        if not os.path.exists(generated_images_path):
            os.makedirs(generated_images_path)

        save_image(grid, f'{generated_images_path}/image_{mthd}.png')
        print(f'Generated images saved to {generated_images_path}/image_{mthd}.png')
    else:
        gen_losses = np.zeros(epochs)
        disc_losses = np.zeros(epochs)
        mean_scores = np.zeros(epochs)
        std_scores = np.zeros(epochs)
        iterations = 0
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        fid = FrechetInceptionDistance().to(device)
        
        sample_images_path = ''
        if use_GP:
            if conditional:
                sample_images_path = f'samples_{mthd}_GP_{dataset}_conditional/'
            else:
                sample_images_path = f'samples_{mthd}_GP_{dataset}/'
                
            if not os.path.exists(sample_images_path):
                os.makedirs(sample_images_path)
        else:
            if conditional:
                sample_images_path = f'samples_{mthd}_{dataset}_conditional/'
            else:
                sample_images_path = f'samples_{mthd}_{dataset}/'
            if not os.path.exists(sample_images_path):
                os.makedirs(sample_images_path)

        for epoch in tqdm(range(epochs), desc='Epochs'):
            d_loss = 0
            g_loss = 0
            for images, labels in train_dataset:
                y = one_hot(labels)
                z = noise_source(m)
                # Train the discriminator
                # generator.trainable = False
                # discriminator.trainable = True
                if conditional:
                    samples = generator([y, z], training=True)
                else:
                    samples = generator(z, training=True)
                    
                for _ in range(disc_steps_per_gen_step):
                    d_loss = div_dense.train_step(images, samples, y)

                # Train the generator
                # generator.trainable = True
                # discriminator.trainable = False
                with tf.GradientTape() as tape:
                    if conditional:
                        samples_ = generator([y, z], training=True)
                    else:
                        samples_ = generator(z, training=True)
                    g_loss = div_dense.generator_loss(samples_, y)
                    
                gradients = tape.gradient(g_loss, generator.trainable_variables)
                gen_optimizer.apply_gradients(zip(gradients, generator.trainable_variables))
                                    
                iterations += 1
                if iterations % 150 == 0:
                    idx = np.random.randint(0, samples_.shape[0], 16)
                    selected_samples = tf.gather(samples_, idx)
                    selected_samples = selected_samples.numpy()
                    grid = make_grid(selected_samples)
                    # Ensure the directory exists
                    if not os.path.exists(sample_images_path):
                        os.makedirs(sample_images_path)
                    save_image(grid, f'{sample_images_path}/samples_{epoch}.png')
            print('Epoch:', epoch, 'Discriminator Loss:', d_loss.numpy(), 'Generator Loss:', g_loss.numpy())  

            # Calculating the FID for each epoch
            
            # # Before updating FID, ensure images are in the correct format
            # def preprocess_for_fid(images):
            #     images = images.numpy()
                
            #     images = torch.from_numpy(images)
            #     # If images are in range [0, 1] with dtype=torch.float32
            #     images = (images * 255).to(torch.uint8)
            #     images = images.permute(0, 3, 1, 2)
            #     images = images.to(device)
            #     return images
            
            # # Function to convert grayscale to RGB
            # def convert_to_rgb(images):
            #     return tf.image.grayscale_to_rgb(images)

            # if conditional:
            #     fake_images = generator(y)
            # else:
            #     fake_images = generator()
            # real_images = images
            
            # # Assuming you have real_images and fake_images
            # real_images_preprocessed = preprocess_for_fid(convert_to_rgb(real_images))
            # fake_images_preprocessed = preprocess_for_fid(convert_to_rgb(fake_images))

            # # Update FID
            # fid.update(real_images_preprocessed, real=True)
            # fid.update(fake_images_preprocessed, real=False)

            # fid_score = fid.compute()
            # fid_scores.append(fid_score.item())
            # fid.reset()

            # print(f'FID Score: {fid_score.item()}')
            
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
                save_model_path = f'{save_model_path}_conditional_MNIST'
            
            if not os.path.exists(save_model_path):
                os.makedirs(save_model_path)
            
            generator.save(f'{save_model_path}/generator_{mthd}.keras')    
            print(f'Model saved to {save_model_path}/generator_{mthd}')

        if use_GP:
            if not os.path.exists(f'losses_{mthd}_GP_{dataset}/'):
                os.makedirs(f'losses_{mthd}_GP_{dataset}/')

            plt.savefig(f'losses_{mthd}_GP_{dataset}/loss_vs_epoch.png')
        else:
            if not os.path.exists(f'losses_{mthd}_{dataset}/'):
                os.makedirs(f'losses_{mthd}_{dataset}/')

            plt.savefig(f'losses_{mthd}_{dataset}/loss_vs_epoch.png')
            plt.show()
            plt.close()    
    
    return fid_scores
    
if __name__ == '__main__':
    method = ['IPM', 'KLD-DV', 'KLD-LT', 'squared-Hel-LT', 'chi-squared-LT', 'JS-LT', 'alpha-LT', 'Renyi-DV', 'Renyi-CC', 'rescaled-Renyi-CC', 'Renyi-CC-WCR']
    for mthd in method:
        fid_scores = main(mthd)
        json.dump(fid_scores, open(f'fid_scores_{mthd}.json', 'w'))
    # main()
