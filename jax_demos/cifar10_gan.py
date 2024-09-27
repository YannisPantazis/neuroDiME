import os
import sys
import argparse
import matplotlib.pyplot as plt
import numpy as np
import time
from typing import Any
import tensorflow as tf
import optax
import tensorflow_datasets as tfds
from flax.training import train_state
import torch
from torchmetrics.image.fid import FrechetInceptionDistance
import json 
from flax import serialization

current_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from models.GAN_CIFAR10_jax import *
from models.Divergences_jax import *

start = time.perf_counter()

fl_act_func_CC = 'poly-softplus' # abs, softplus, poly-softplus
optimizer = "Adam" #Adam, RMS
save_frequency = 10 #generator samples are saved every save_frequency epochs
num_gen_samples_to_save = 5000

def prepare_dataset(X):
    X = tf.cast(X, tf.float32)
    # Normalization, pixels in [-1, 1]
    X = (X / 255.0) * 2.0 - 1.0
    # shape=(batch_size, 32, 32, 3)
    return X

def save_image(fake_images, path, normalize=False):
    fake_images = jax.lax.stop_gradient(fake_images)  # Stop gradient tracking
    if normalize:
        fake_images = (fake_images + 1.0) / 2.0  # Normalize to [0, 1]

    # Convert to NumPy array
    fake_images = np.array(fake_images)
    
    # Reshape to (batch_size, channels, height, width)
    fig = plt.figure(figsize=(8, 8))
    for i in range(min(fake_images.shape[0], 64)):  # Limit to first 64 images
        plt.subplot(8, 8, i + 1)
        img = fake_images[i, :, :, :].clip(0, 1)  # Ensure values are in [0, 1] for display
        plt.imshow(img)
        plt.axis('off')
    
    plt.savefig(path)
    plt.close(fig)


def make_grid(images, nrow=4, padding=1):
    n, h, w, c = images.shape
    grid_height = h * nrow + padding * (nrow - 1)
    grid_width = w * nrow + padding * (nrow - 1)
    grid = jnp.ones((grid_height, grid_width, c))

    next_idx = 0
    for y in range(nrow):
        for x in range(nrow):
            if next_idx >= n:
                break
            img = images[next_idx]
            grid = grid.at[
                y * (h + padding): y * (h + padding) + h,
                x * (w + padding): x * (w + padding) + w, :
            ].set(img)
            next_idx += 1
        if next_idx >= n:
            break
    return grid


class GradientPenalty(Discriminator_Penalty):
    def __init__(self, gp_weight, L):
        Discriminator_Penalty.__init__(self, gp_weight)
        self.L = L
    
    def get_Lip_constant(self):
        return self.L

    def set_Lip_constant(self, L):
        self.L = L

    def evaluate(self, discriminator, real_images, fake_images, params, batch_stats, key, labels=None, dropout_rng=None):
        assert real_images.shape == fake_images.shape

        # Interpolate between real and fake images
        alpha = jax.random.uniform(key, shape=(real_images.shape[0], 1, 1, 1))
        interpolated_images = alpha * real_images + (1 - alpha) * fake_images

        # Compute gradients of the discriminator with respect to the interpolated images
        def discriminator_loss_fn(interpolated_images):
            if labels is not None:
                if batch_stats is None:
                    return discriminator.apply({'params': params}, interpolated_images, labels)
                else:
                    return discriminator.apply({'params': params, 'batch_stats': batch_stats}, interpolated_images, labels)
            else:
                if batch_stats is None:
                    return discriminator.apply({'params': params}, interpolated_images)
                else:
                    return discriminator.apply({'params': params, 'batch_stats': batch_stats}, interpolated_images)
        
        gradients = grad(lambda interpolated: jnp.sum(discriminator_loss_fn(interpolated)))(interpolated_images)

        # Reshape gradients to compute the norm
        gradients = jnp.reshape(gradients, (gradients.shape[0], -1))
        gradient_norms = jnp.linalg.norm(gradients, axis=1)

        # Calculate the gradient penalty
        gradient_penalty = jnp.mean((gradient_norms - self.L) ** 2)

        return gradient_penalty * self.get_penalty_weight()


def noise_source(batch_size, noise_dim, key):
    return jax.random.normal(key, (batch_size, noise_dim))


def generate_labeled_cifar10(generator, gen_params, num_rows=10):
    num_classes = 10
    
    # Generate noise
    rng = jax.random.PRNGKey(0)
    z = noise_source(num_rows, 128, rng)
    
    digit_images = []
    
    for class_ in range(num_classes):
        labels = jnp.array([class_] * num_rows)
        labels_one_hot = jax.nn.one_hot(labels, num_classes)
        generated_images = generator.apply({'params': gen_params}, labels=labels_one_hot, z=z)
        digit_images.append(((np.array(generated_images) + 1) * 127.5).astype(np.uint8))
    
    # Stack the generated images vertically
    digit_images = np.vstack(digit_images)
    
    # Create a grid of images with num_rows rows and num_classes columns
    grid = make_grid(digit_images, nrow=num_classes, padding=2)
    
    return grid


def save_image(grid, filename):
    grid = grid.astype(np.uint8)
    plt.figure(figsize=(grid.shape[1] / 100, grid.shape[0] / 100), dpi=100)
    plt.imshow(grid)
    plt.axis('off')
    plt.savefig(filename, bbox_inches='tight', pad_inches=0)
    plt.close()

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
    parser.add_argument('--disc_steps_per_gen_step', default=3, type=int)
    parser.add_argument('--batch_size', default=256, type=int, metavar='m')
    parser.add_argument('--lr', default=2e-4, type=float)
    parser.add_argument('--epochs', default=150, type=int,
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
    dataset = 'cifar10'
    save_model = opt_dict['save_model']=='True'
    save_model_path = opt_dict['save_model_path']
    load_model = opt_dict['load_model']=='True'
    load_model_path = opt_dict['load_model_path']
    conditional = opt_dict['conditional']=='True'
    
    print("Spectral_norm: "+str(spec_norm))
    print("Bounded: "+str(bounded))
    print("Reversed: "+str(reverse_order))
    print("Use Gradient Penalty: "+str(use_GP))
    print("Method: "+mthd)
    noise_dim = 128
    
    # Load the CIFAR-10 dataset
    cifar10_data = tfds.load('cifar10')['train']
    batches_in_epoch = len(cifar10_data) // m
    data_gen = iter(tfds.as_numpy(
                cifar10_data
                .map(set_range)
                .cache()
                .shuffle(len(cifar10_data), seed=42)
                .repeat()
                .batch(m)))
    
    if conditional:
        generator = Generator_cond()
        discriminator = Discriminator_cond()
    else:
        generator = Generator()
        discriminator = Discriminator()
    
    key = jax.random.PRNGKey(0)
    gen_key, disc_key = jax.random.split(key)
    images_init = jnp.ones((m, 32, 32, 3))
    noise_init = jnp.ones((m, noise_dim))
    labels_init = jnp.ones((m, 10))
    if conditional:
        gen_variables = generator.init(gen_key, labels_init, noise_init)
        disc_variables = discriminator.init(disc_key, images_init, labels_init)
    else:
        gen_variables = generator.init(gen_key, noise_init)
        disc_variables = discriminator.init(disc_key, images_init)

    if optimizer == 'Adam':
        disc_optimizer = optax.adam(learning_rate=lr, b1=0.5, b2=0.999)
        gen_optimizer = optax.adam(learning_rate=lr, b1=0.5, b2=0.999)
    elif optimizer == 'RMS':
        disc_optimizer = optax.rmsprop(learning_rate=lr)
        gen_optimizer = optax.rmsprop(learning_rate=lr)
    
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

    gen_losses = np.zeros(epochs)
    disc_losses = np.zeros(epochs)
    mean_scores = np.zeros(epochs)
    std_scores = np.zeros(epochs)
    iterations = 0
    
    sample_images_path = ''
    if use_GP:
        if conditional:
            sample_images_path = f'samples_{mthd}_GP_{dataset}_conditional/'
            if not os.path.exists(sample_images_path):
                os.makedirs(sample_images_path)
        else:
            sample_images_path = f'samples_{mthd}_GP_{dataset}/'
            if not os.path.exists(sample_images_path):
                os.makedirs(sample_images_path)
    else:
        if conditional:
            sample_images_path = f'samples_{mthd}_{dataset}_conditional'
            if not os.path.exists(sample_images_path):
                os.makedirs(sample_images_path)
        else:
            sample_images_path = f'samples_{mthd}_{dataset}/'
            if not os.path.exists(sample_images_path):
                os.makedirs(sample_images_path)
    
    if load_model:
        if conditional:
            generated_images_path = f'generated_cifar10_images_{mthd}_conditional/'
            generator_path = os.path.join(save_model_path + '_conditional_CIFAR10', f'generator_{mthd}_params.pkl')
        else:
            generator_path = os.path.join(load_model_path + '_CIFAR10', f'generator_{mthd}_params.pkl')
        
        with open(generator_path, 'rb') as f:
            gen_params = serialization.from_bytes(gen_train_state.params, f.read())
        
        gen_train_state = gen_train_state.replace(params=gen_params)
        print(f"Loaded generator parameters from {generator_path}")
        
        grid = generate_labeled_cifar10(generator, gen_train_state.params)
        
        if not os.path.exists(generated_images_path):
            os.makedirs(generated_images_path)
        
        save_image(grid, f'{generated_images_path}/generated_images_{mthd}_conditional.png')
        print(f"Saved generated images to {generated_images_path}/generated_images_{mthd}_conditional.png")
        return None
    else:
        fid = FrechetInceptionDistance()
        fid_scores = []
        for epoch in tqdm(range(epochs), desc='Epochs'):
            d_loss = 0
            g_loss = 0
            for _ in range(batches_in_epoch):
                data = next(data_gen)
                images = data[0]

                # Train the discriminator
                images = jnp.array(images)
                key = jax.random.PRNGKey(epoch)

                z = noise_source(m, noise_dim, key)
                if conditional:
                    labels = data[1]
                    labels = jax.nn.one_hot(labels, 10)
                    samples = generator.apply({"params": gen_train_state.params}, z=z, labels=labels)
                else:
                    samples = generator.apply({"params": gen_train_state.params}, z=z)
                    
                for _ in range(disc_steps_per_gen_step):
                    disc_train_state, d_loss = div_dense.train_step(images, samples, disc_train_state, disc_variables, key, labels)
                
                # Train the generator
                gen_train_state, g_loss = div_dense.gen_train_step(gen_train_state, disc_train_state, disc_variables, gen_variables, key, z, labels)

                iterations += 1
                if iterations % 150 == 0:
                    if conditional:
                        samples_ = generator.apply({"params": gen_train_state.params}, z=z, labels=labels)
                    else:
                        samples_ = generator.apply({"params": gen_train_state.params}, z=z)
                    idx = jax.random.randint(key, (64,), 0, m)
                    selected_images = jnp.take(samples_, idx, axis=0)
                    selected_images = np.array(selected_images)
                    
                    if not os.path.exists(sample_images_path):
                        os.makedirs(sample_images_path)
                    
                    save_image(selected_images, f'{sample_images_path}/samples_{epoch}.png', normalize=True)
                    
            print(f'Epoch {epoch}, Discriminator Loss: {d_loss}, Generator Loss: {g_loss}')
            if math.isnan(d_loss) or math.isnan(g_loss):
                break
            # Calculating the FID for each epoch
                
            # Before updating FID, ensure images are in the correct format
            def preprocess_for_fid(images):
                images = np.array(images)

                images = torch.from_numpy(images)
                # If images are in range [0, 1] with dtype=torch.float32
                images = (images * 255).to(torch.uint8)
                images = images.permute(0, 3, 1, 2)
                images = images
                return images

            if conditional:
                fake_images = generator.apply({'params': gen_train_state.params}, z=z, labels=labels)
            else:
                fake_images = generator.apply({'params': gen_train_state.params}, z=z)
            real_images = images
            # Assuming you have real_images and fake_images
            real_images_preprocessed = preprocess_for_fid(real_images)
            fake_images_preprocessed = preprocess_for_fid(fake_images)

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
            save_model_path = save_model_path + '_CIFAR10'
            
            if conditional:
                save_model_path = save_model_path + '_conditional_CIFAR10'
                
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