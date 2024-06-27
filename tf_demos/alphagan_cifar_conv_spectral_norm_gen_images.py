import tensorflow as tf
from tensorflow.keras import datasets, layers, models, optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import entropy
from PIL import Image
from tqdm import tqdm
import sys

# Add the path to the system path
current_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from models.GAN_CIFAR10 import *  # Adjust this import based on your actual module structure

# Constants
BATCH_SIZE = 64  # Critic batch size
GEN_BS_MULTIPLE = 2  # Generator batch size, as a multiple of BATCH_SIZE
DIM_G = 128  # Generator dimensionality
DIM_D = 128  # Critic dimensionality
NORMALIZATION_G = True  # Use batchnorm in generator?
NORMALIZATION_D = False  # Use batchnorm (or layernorm) in critic? This doesn't do anything at the moment.
OUTPUT_DIM = 3072  # Number of pixels in CIFAR-10 (32*32*3)
DECAY = True  # Whether to decay LR over learning
INCEPTION_FREQUENCY = 100  # How frequently to calculate Inception score
j = 0

LR = 2e-4  # Initial learning rate
ITERS = 75
alpha = 0
rev = 0
print_every = 5

# WGAN-GP parameter
CONDITIONAL = False  # Whether to train a conditional or unconditional model
ACGAN = False  # If CONDITIONAL, whether to use ACGAN or "vanilla" conditioning
ACGAN_SCALE = 1.0  # How to scale the critic's ACGAN loss relative to WGAN loss
ACGAN_SCALE_G = 0.1  # How to scale generator's ACGAN loss relative to WGAN loss

if CONDITIONAL and (not ACGAN) and (not NORMALIZATION_D):
    print("WARNING! Conditional model without normalization in D might be effectively unconditional!")
device = 'GPU' if tf.config.list_physical_devices('GPU') else 'CPU'

print(f"Using device: {device}")

# Data loading and preprocessing
def preprocess_image(image):
    image = (image - 127.5) / 127.5  # Normalize images to [-1, 1]
    return image


def generate_image(generator, frame):
    """Generate a batch of images and save them to a grid."""
    n_samples = 12
    fixed_noise = tf.random.normal([n_samples, 128])
    fixed_labels = tf.constant(np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9] * n_samples)[:n_samples], dtype=tf.int32)

    # Ensure the generator is in evaluation mode
    training = False
    samples = generator(n_samples, labels=fixed_labels, noise=fixed_noise, training=training)
    samples = ((samples + 1) * 127.5).numpy().clip(0, 255).astype(np.uint8)
    print(samples.shape)
    samples = samples.reshape(n_samples, 32, 32, 3)

    n_rows = (n_samples + 3) // 4
    fig, axs = plt.subplots(
        nrows=n_rows,
        ncols=4,
        figsize=(8, 2*n_rows),
        subplot_kw={'xticks': [], 'yticks': [], 'frame_on': False}
    )
    for i, axis in enumerate(axs.flat[:n_samples]):
        axis.imshow(samples[i], cmap='binary')
    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    os.makedirs('samples', exist_ok=True)
    plt.savefig(f'samples/gen_images_{frame}.png')
    # plt.show()  # Uncomment to display the images
    plt.close(fig)


def train(generator, discriminator, gen_opt, disc_opt, train_dataset):
    """Train the generator and discriminator for a number of epochs."""
    disc_costs = np.zeros(ITERS)
    gen_costs = np.zeros(ITERS)
    
    for iteration in tqdm(range(ITERS)):
        gen_loss = 0
        disc_loss = 0
        
        for real_data, labels in train_dataset:
            # real_data = tf.convert_to_tensor(real_data, dtype=tf.float32)
            real_data = tf.cast(real_data, tf.float32)
            
            with tf.GradientTape() as disc_tape, tf.GradientTape() as gen_tape:
                # Generator forward pass
                noise = tf.random.normal([BATCH_SIZE, DIM_G])
                fake_data = generator(BATCH_SIZE, labels, noise, training=True)
                
                # Discriminator forward pass
                disc_real, _ = discriminator(real_data, labels, training=True)
                disc_fake, _ = discriminator(fake_data, labels, training=True)
                
                # Calculate discriminator loss
                if alpha == 0: # Standard WGAN loss
                    disc_cost_real = tf.reduce_mean(disc_real)
                    disc_cost_fake = tf.reduce_mean(disc_fake)
                    disc_cost = disc_cost_fake - disc_cost_real
                elif alpha == 1:
                    if rev == 0:
                        disc_cost_real = tf.reduce_mean(disc_real)
                        disc_cost_fake = tf.reduce_mean(tf.math.log(tf.reduce_mean(tf.exp(disc_fake))))
                        disc_cost = disc_cost_fake - disc_cost_real
                    elif rev == 1:
                        disc_cost_real = tf.reduce_mean(tf.math.log(tf.reduce_mean(tf.exp(disc_real))))
                        disc_cost_fake = tf.reduce_mean(disc_fake)
                        disc_cost = disc_cost_fake - disc_cost_real
                elif alpha == -1: # alpha = infinity
                    if rev == 0:
                        disc_cost_real = tf.reduce_mean(disc_real)
                        disc_cost_fake = tf.reduce_mean(tf.nn.relu(disc_fake))
                        disc_cost = disc_cost_fake - disc_cost_real
                    elif rev == 1:
                        disc_cost_real = tf.reduce_mean(tf.nn.relu(disc_real))
                        disc_cost_fake = tf.reduce_mean(disc_fake)
                        disc_cost = disc_cost_fake - disc_cost_real
                else: # Reversed generalized alphaGAN
                    if rev == 0:
                        disc_cost_real = tf.reduce_mean(disc_real)
                        disc_cost_fake = tf.reduce_mean(f_alpha_star(disc_fake, alpha))
                        disc_cost = disc_cost_fake - disc_cost_real
                    elif rev == 1:
                        disc_cost_real = tf.reduce_mean(f_alpha_star(disc_real, alpha))
                        disc_cost_fake = tf.reduce_mean(disc_fake)
                        disc_cost = disc_cost_fake - disc_cost_real
                
                disc_loss += disc_cost.numpy()
                
                # Update discriminator
                disc_gradients = disc_tape.gradient(disc_cost, discriminator.trainable_variables)
                disc_opt.apply_gradients(zip(disc_gradients, discriminator.trainable_variables))

                # Apply gradient clipping
                for var in discriminator.trainable_variables:
                    var.assign(tf.clip_by_value(var, -0.01, 0.01))

                # Generator forward pass
                noise = tf.random.normal([BATCH_SIZE, DIM_G])
                fake_data = generator(BATCH_SIZE, labels, noise, training=True)
                disc_fake, _ = discriminator(fake_data, labels, training=True)
                
                # Calculate generator loss
                if alpha == 0: # Standard WGAN loss
                    gen_cost = -tf.reduce_mean(disc_fake)
                elif alpha == 1:
                    if rev == 0:
                        gen_cost = -tf.reduce_mean(tf.math.log(tf.reduce_mean(tf.exp(disc_fake))))
                    elif rev == 1:
                        gen_cost = tf.reduce_mean(disc_fake)
                elif alpha == -1: # alpha = infinity
                    if rev == 0:
                        gen_cost = -tf.reduce_mean(tf.nn.relu(disc_fake))
                    elif rev == 1:
                        gen_cost = tf.reduce_mean(disc_fake)
                else: # Reversed generalized alphaGAN
                    if rev == 0:
                        gen_cost = -tf.reduce_mean(f_alpha_star(disc_fake, alpha))
                    elif rev == 1:
                        gen_cost = tf.reduce_mean(disc_fake)
                
                gen_loss += gen_cost.numpy()
                    
                # Update generator
                gen_gradients = gen_tape.gradient(gen_cost, generator.trainable_variables)
                gen_opt.apply_gradients(zip(gen_gradients, generator.trainable_variables))
        
        disc_costs[iteration] = disc_loss / len(train_dataset)
        gen_costs[iteration] = gen_loss / len(train_dataset)
        
        # Save checkpoints
        if iteration % print_every == 0:
            generator.save_weights(f'cifar_resnet_sn/checkpoint_generator_{iteration}.h5')
            discriminator.save_weights(f'cifar_resnet_sn/checkpoint_discriminator_{iteration}.h5')
            print(f'Iteration {iteration}, Generator loss: {gen_costs[iteration]}, Discriminator loss: {disc_costs[iteration]}')
            
            generate_image(generator, iteration)
            # Calculate and log the inception score (if needed)
            # inception_mean, inception_std = get_inception_score(generator, 5000)
            # print(f"Inception score at iteration {iteration}: {inception_mean} ± {inception_std}")

    return disc_costs, gen_costs


def main():
    if not os.path.exists('cifar_resnet_sn/'):
        os.makedirs('cifar_resnet_sn/')

    # Initialize models
    generator = Generator(dim_g=DIM_G, output_dim=OUTPUT_DIM)
    discriminator = Discriminator(dim_d=DIM_D)
    
    
    # Build the models by passing a sample input through them
    generator(n_samples=1, labels=tf.zeros((1,), dtype=tf.int32), noise=tf.zeros((1, 128)))
    discriminator(tf.zeros((1, 32, 32, 3)), labels=tf.zeros((1,), dtype=tf.int32))

    generator.summary()
    print()
    discriminator.summary()
    print('Using device:', device)

    # Optimizers
    gen_opt = tf.keras.optimizers.Adam(learning_rate=LR, beta_1=0.0, beta_2=0.9)
    disc_opt = tf.keras.optimizers.Adam(learning_rate=LR, beta_1=0.0, beta_2=0.9)

    def preprocess_image(image):
        image = (image - 127.5) / 127.5  # Normalize images to [-1, 1]
        return image

    (train_images, train_labels), _ = tf.keras.datasets.cifar10.load_data()
    train_images = preprocess_image(train_images)

    train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels)).shuffle(50000).batch(BATCH_SIZE)

    d_loss, g_loss = train(generator, discriminator, gen_opt, disc_opt, train_dataset)

    # Save the loss vs epoch plot
    epoch_ax = np.arange(start=1, stop=ITERS + 1, step=1)
    _, ax = plt.subplots(1, 2, figsize=(15, 5))

    ax[0].plot(epoch_ax, d_loss, color='blue')
    ax[0].set_xlim(1, ITERS)
    ax[0].set_title("Discriminator Loss vs Epoch")
    ax[0].grid()

    ax[1].plot(epoch_ax, g_loss, color='red')
    ax[1].set_xlim(1, ITERS)
    ax[1].set_title("Generator Loss vs Epoch")
    ax[1].grid()

    plt.tight_layout()
    plt.savefig('cifar_resnet_sn/loss_vs_epoch.png')
    plt.show()
    plt.close()


if __name__ == "__main__":
    main()