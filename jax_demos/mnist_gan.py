import os
import sys
import argparse
import matplotlib.pyplot as plt
import numpy as np
import time
import tensorflow as tf
import tensorflow_datasets as tfds
from typing import Any
from flax import serialization
import json 
from torchmetrics.image.fid import FrechetInceptionDistance
import torch
import jax.numpy as jnp

current_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from models.GAN_MNIST_jax import *
from models.Divergences_jax import *

start = time.perf_counter()

fl_act_func_CC = 'poly-softplus' # abs, softplus, poly-softplus
optimizer = "Adam" #Adam, RMS
save_frequency = 10 #generator samples are saved every save_frequency epochs
num_gen_samples_to_save = 5000

class GradientPenalty(Discriminator_Penalty):
    def __init__(self, penalty_weight, L):
        Discriminator_Penalty.__init__(self, penalty_weight)
        self.L = L

    def get_Lip_constant(self):
        return self.L

    def set_Lip_constant(self, L):
        self.L = L

    def evaluate(self, discriminator, images, samples, params, batch_stats, key, labels=None, dropout_rng=None):
        assert images.shape == samples.shape
        batch_size = images.shape[0]
        jump = random.uniform(key, shape=(batch_size, 1))
        jump_ = jump.reshape((batch_size, 1, 1, 1))
        interpolated = images * jump_ + (1 - jump_) * samples

        def compute_d_pred(interpolated):
            if labels is not None:
                if dropout_rng is not None:
                    return discriminator.apply({'params': params}, interpolated, labels, rngs={'dropout': dropout_rng})
                else:
                    return discriminator.apply({'params': params}, interpolated, labels) 
            else:
                if dropout_rng is not None:
                    return discriminator.apply({'params': params}, interpolated, rngs={'dropout': dropout_rng})
                else:
                    return discriminator.apply({'params': params}, interpolated)

        # Compute gradients with respect to the interpolated samples
        grads = grad(lambda interpolated: jnp.sum(compute_d_pred(interpolated)))(interpolated)

        # Calculate the norm of the gradients
        gradients = jnp.reshape(grads, (grads.shape[0], -1))
        gradients_norm = jnp.linalg.norm(gradients, axis=1)
        gradient_penalty = jnp.mean(jnp.square(gradients_norm - self.L))

        # Compute the gradient penalty
        gradient_penalty = self.get_penalty_weight() * gradient_penalty
        return gradient_penalty


def preprocess(image, label):
    image = tf.cast(image, tf.float32) / 255.0  # Normalize using TensorFlow
    return image, label


def batch_dataset(dataset, batch_size):
    dataset = dataset.cache()
    dataset = dataset.shuffle(1000)
    dataset = dataset.batch(batch_size, drop_remainder=True)
    dataset = dataset.prefetch(jax.device_count())
    return dataset


def make_grid(images, nrow=4, padding=2):
    n, h, w, c = images.shape
    nrow = min(nrow, n)  # Ensure nrow is not greater than the number of images
    ncol = (n + nrow - 1) // nrow  # Calculate number of columns needed

    grid_height = h * ncol + padding * (ncol - 1)
    grid_width = w * nrow + padding * (nrow - 1)
    grid = np.ones((grid_height, grid_width, c), dtype=images.dtype)

    next_idx = 0
    for y in range(ncol):
        for x in range(nrow):
            if next_idx >= n:
                break
            img = images[next_idx]
            y_start = y * (h + padding)
            x_start = x * (w + padding)
            grid[y_start:y_start + h, x_start:x_start + w, :] = img
            next_idx += 1
        if next_idx >= n:
            break

    return grid


def generate_labeled_mnist(generator, gen_params, num_rows=10):
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
    
    # Generate noise
    rng = jax.random.PRNGKey(0)
    
    # Initialize an empty list to store images for each digit
    digit_images = []
    
    for digit in range(num_classes):
        labels = jnp.array([digit] * num_rows)
        labels_one_hot = jax.nn.one_hot(labels, num_classes)
        generated_images = generator.apply({'params': gen_params}, labels=labels_one_hot)
        digit_images.append(np.array(generated_images))
    
    # Stack the generated images vertically
    digit_images = np.vstack(digit_images)
    
    # Create a grid of images with num_rows rows and num_classes columns
    grid = make_grid(digit_images, nrow=num_classes, padding=2)
    
    return grid


def save_image(grid, filename, cmap='gray'):
    plt.figure(figsize=(grid.shape[1] / 100, grid.shape[0] / 100), dpi=100)
    if grid.shape[-1] == 1:  # Grayscale image
        grid = grid.squeeze(-1)  # Remove the channel dimension
        plt.imshow(grid, cmap=cmap)
    else:  # Color image (assumed RGB)
        plt.imshow(grid)
    plt.axis('off')
    plt.savefig(filename, bbox_inches='tight', pad_inches=0)
    plt.close()


def show_generated_images(generated_images, filename, num_images=10):
    # Assuming generated_images is a numpy array of shape (n, height, width)
    # We will display 'num_images' images in a grid

    plt.figure(figsize=(20, 20))
    for i in range(num_images):
        plt.subplot(1, num_images, i+1)
        plt.imshow(generated_images[i].squeeze(), cmap='gray')
        plt.axis('off')
    plt.savefig(filename, bbox_inches='tight', pad_inches=0)
    # plt.show()


def set_range(batch):
    batch, labels = batch['image'], batch['label']  #  We now add the labels to the generator.
    batch = tf.image.convert_image_dtype(batch, tf.float32)
    batch = (batch - 0.5) / 0.5  # tanh range is -1, 1
    return (batch, labels)


def main(mthd):
    # read input arguments
    parser = argparse.ArgumentParser(description='Neural-based Estimation of Divergences between Gaussians')
    parser.add_argument('--method', default='KLD-DV', type=str, metavar='method',
                        help='values: IPM, KLD-DV, KLD-LT, squared-Hel-LT, chi-squared-LT, JS-LT, alpha-LT, Renyi-DV, Renyi-CC, rescaled-Renyi-CC, Renyi-CC-WCR')
    parser.add_argument('--disc_steps_per_gen_step', default=5, type=int)
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
    load_model = opt_dict['load_model']=='True'
    load_model_path = opt_dict['load_model_path']
    
    print("Spectral_norm: "+str(spec_norm))
    print("Bounded: "+str(bounded))
    print("Reversed: "+str(reverse_order))
    print("Use Gradient Penalty: "+str(use_GP))
    
    # Load the dataset
    # Load the CIFAR-10 dataset
    mnist_data = tfds.load('mnist')['train']
    batches_in_epoch = len(mnist_data) // m
    data_gen = iter(tfds.as_numpy(
                mnist_data
                .map(set_range)
                .cache()
                .shuffle(len(mnist_data), seed=42)
                .repeat()
                .batch(m)))


    
    # Initialize models
    generator = Generator_MNIST_cond()
    discriminator = Discriminator_MNIST_cond()

    # Initialize parameters
    rng = jax.random.PRNGKey(0)
    rng, dropout_rng = jax.random.split(rng)
    rng, gen_rng = jax.random.split(rng)
    rng, disc_rng = jax.random.split(rng)
    labels_init = jnp.ones((1, 10))
    images_init = jnp.ones((1, 28, 28, 1))
    noise_init = jnp.ones((1, 118))
    gen_variables = generator.init(gen_rng, labels_init, noise_init)
    disc_variables = discriminator.init(disc_rng, images_init, labels_init)
    
    if optimizer == 'Adam':
        disc_optimizer = optax.adam(learning_rate=lr)
        gen_optimizer = optax.adam(learning_rate=lr)
    elif optimizer == 'RMS':
        disc_optimizer = optax.rmsprop(learning_rate=lr)
        gen_optimizer = optax.rmsprop(learning_rate=lr)
        
    # Initialize train states
    gen_train_state = train_state.TrainState.create(apply_fn=generator.apply, params=gen_variables['params'], tx=gen_optimizer)
    disc_train_state = train_state.TrainState.create(apply_fn=discriminator.apply, params=disc_variables['params'], tx=disc_optimizer)
    
    if use_GP:
        discriminator_penalty = GradientPenalty(gp_weight, L)
    else:
        discriminator_penalty = None

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
        div_dense = Renyi_Divergence_WCR(discriminator, disc_optimizer, math.inf, epochs, m, fl_act_func_CC, discriminator_penalty)

    def noise_source(batch_size):
        """
        Generate noise for the generator input.

        Args:
            batch_size (int): The size of the batch.
            key (jax.random.PRNGKey): A JAX random key.

        Returns:
            jnp.ndarray: A batch of noise with shape [batch_size, 124, 1, 1].
        """
        return jax.random.normal(jax.random.PRNGKey(0), (batch_size, 124, 1, 1))

    gen_losses = np.zeros(epochs)
    disc_losses = np.zeros(epochs)
    mean_scores = np.zeros(epochs)
    std_scores = np.zeros(epochs)
    iterations = 0
    
    sample_images_path = ''
    if use_GP:
        if not os.path.exists(f'samples_{mthd}_GP_{dataset}/'):
            os.makedirs(f'samples_{mthd}_GP_{dataset}/')
        sample_images_path = f'samples_{mthd}_GP_{dataset}/'
    else:
        if not os.path.exists(f'samples_{mthd}_{dataset}/'):
            os.makedirs(f'samples_{mthd}_{dataset}/')
        sample_images_path = f'samples_{mthd}_{dataset}/'

    # Loading model parameters
    if load_model:
        generated_images_path = f'generated_digits/'
        generator_path = os.path.join(load_model_path, f'generator_{mthd}_params.pkl')
        discriminator_path = os.path.join(load_model_path, f'discriminator_{mthd}_params.pkl')

        with open(generator_path, 'rb') as f:
            gen_params = serialization.from_bytes(gen_train_state.params, f.read())
        with open(discriminator_path, 'rb') as f:
            disc_params = serialization.from_bytes(disc_train_state.params, f.read())

        # Update the model states with the loaded parameters
        gen_train_state = gen_train_state.replace(params=gen_params)
        disc_train_state = disc_train_state.replace(params=disc_params)

        print(f"Loaded generator parameters from {generator_path}")
        print(f"Loaded discriminator parameters from {discriminator_path}")

        grid = generate_labeled_mnist(generator, gen_train_state.params)
        
        if not os.path.exists(generated_images_path):
            os.makedirs(generated_images_path)
                        
        save_image(grid, f'{generated_images_path}/image_{mthd}.png')
        print(f"Saved generated images to {generated_images_path}")
    else:
        fid = FrechetInceptionDistance()

        iterations = 0
        fid_scores = []
        for epoch in tqdm(range(epochs), desc='Epochs'):
            d_loss = 0
            g_loss = 0
            for _ in range(batches_in_epoch):
                images, labels = next(data_gen)
                images, labels = jnp.array(images), jnp.array(labels)
                key = jax.random.PRNGKey(epoch)

                y = nn.one_hot(labels, 10)
                
                # Train the discriminator
                z = random.normal(key, (labels.shape[0], 118))
                samples = generator.apply({'params': gen_train_state.params}, labels=y, z=z)
                for _ in range(disc_steps_per_gen_step):
                    disc_train_state, d_loss = div_dense.train_step(images, samples, disc_train_state, disc_variables, key, y, dropout_rng)
                
                # Train the generator
                gen_train_state, g_loss = div_dense.gen_train_step(gen_train_state, disc_train_state, disc_variables, gen_variables, key, y, dropout_rng)
                
                iterations += 1
                if iterations % 150 == 0:
                    samples = generator.apply({'params': gen_train_state.params}, labels=y, z=z)
                    d_samples = discriminator.apply({'params':disc_train_state.params}, samples, y, training=False, rngs={'dropout': dropout_rng})
                    rng, idx_rng = jax.random.split(rng)
                    idx = jax.random.randint(idx_rng, (16,), 0, d_samples.shape[0])
                    selected_samples = jnp.take(samples, idx, axis=0)
                    selected_samples = np.array(selected_samples)
                    grid = make_grid(selected_samples)

                    if not os.path.exists(sample_images_path):
                        os.makedirs(sample_images_path)
                        
                    save_image(grid, f'{sample_images_path}/samples_{epoch}.png')

            print(f'Epoch {epoch}, Discriminator Loss: {d_loss}, Generator Loss: {g_loss}')
            
            # Calculating the FID for each epoch
            
            # Before updating FID, ensure images are in the correct format
            def preprocess_for_fid(images):
                images = np.array(images)

                images = torch.from_numpy(images)
                # If images are in range [0, 1] with dtype=torch.float32
                images = (images * 255).to(torch.uint8)
                images = images.permute(0, 3, 1, 2)
                return images
            
            # Function to convert grayscale to RGB
            def convert_to_rgb(images):
                return jnp.repeat(images, 3, axis=-1)

            fake_images = generator.apply({'params': gen_train_state.params}, labels=y, z=z)
            real_images = images
            # Assuming you have real_images and fake_images
            real_images_preprocessed = preprocess_for_fid(convert_to_rgb(real_images))
            fake_images_preprocessed = preprocess_for_fid(convert_to_rgb(fake_images))

            # Update FID
            fid.update(real_images_preprocessed, real=True)
            fid.update(fake_images_preprocessed, real=False)

            fid_score = fid.compute()
            fid_scores.append(fid_score.item())
            fid.reset()
            print(f'FID Score: {fid_score.item()}')
        
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
        
        # Save model parameters
        if save_model:
            if not os.path.exists(save_model_path):
                os.makedirs(save_model_path)
            generator_path = os.path.join(save_model_path, f'generator_{mthd}_params.pkl')
            discriminator_path = os.path.join(save_model_path, f'discriminator_{mthd}_params.pkl')

            with open(generator_path, 'wb') as f:
                f.write(serialization.to_bytes(gen_train_state.params))
            with open(discriminator_path, 'wb') as f:
                f.write(serialization.to_bytes(disc_train_state.params))


            print(f"Saved generator parameters to {generator_path}")
            print(f"Saved discriminator parameters to {discriminator_path}")
            
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
    for mthd in ["IPM", "KLD-DV", "KLD-LT", "squared-Hel-LT", "chi-squared-LT", "JS-LT", "alpha-LT", "Renyi-DV", "Renyi-CC", "rescaled-Renyi-CC", "Renyi-CC-WCR"]:
        fid_scores = []
        fid_scores = main(mthd)
        
        with open(f"fid_scores_{mthd}.json", "w") as file:
            json.dump(fid_scores, file)

