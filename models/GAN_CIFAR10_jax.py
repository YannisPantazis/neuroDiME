import jax.numpy as jnp
from flax import linen as nn

class Discriminator(nn.Module):
    '''
    Discriminator class for an unconditional GAN model.
    Applies several convolutional layers followed by LeakyReLU activations to classify the input.

    Methods:
        __call__: The forward pass through the network.
    '''
    
    @nn.compact
    def __call__(self, x):
        '''
        Forward pass through the discriminator.

        Args:
            x (jnp.ndarray): Input image tensor of shape (batch_size, height, width, channels).

        Returns:
            jnp.ndarray: A scalar score for each image in the batch.
        '''
        x = nn.Conv(features=64, kernel_size=(3, 3), strides=(2, 2))(x)
        x = nn.leaky_relu(x, negative_slope=0.2)
        
        x = nn.Conv(features=128, kernel_size=(3, 3), strides=(2, 2))(x)
        x = nn.leaky_relu(x, negative_slope=0.2)
        
        x = nn.Conv(features=256, kernel_size=(3, 3), strides=(2, 2))(x)
        x = nn.leaky_relu(x, negative_slope=0.2)
        
        x = nn.Conv(features=512, kernel_size=(3, 3), strides=(2, 2))(x)
        x = nn.leaky_relu(x, negative_slope=0.2)
        
        x = x.reshape((x.shape[0], -1))  # Flatten the feature map
        x = nn.Dense(features=1)(x)  # Output a single scalar value
        return x


class Generator(nn.Module):
    '''
    Generator class for an unconditional GAN model.
    Takes a latent code and generates images using deconvolutional layers.

    Methods:
        __call__: The forward pass through the network.
    '''
    
    @nn.compact
    def __call__(self, z):
        '''
        Forward pass through the generator.

        Args:
            z (jnp.ndarray): Input latent vector of shape (batch_size, latent_dim).

        Returns:
            jnp.ndarray: Generated image tensor of shape (batch_size, 32, 32, 3).
        '''
        x = nn.Dense(features=4 * 4 * 512)(z)
        x = nn.leaky_relu(x, negative_slope=0.2)
        x = x.reshape((x.shape[0], 4, 4, 512))  # Reshape to start conv layers
        
        x = nn.ConvTranspose(features=256, kernel_size=(4, 4), strides=(2, 2), padding='SAME')(x)
        x = nn.leaky_relu(x, negative_slope=0.2)  # Output shape: (batch_size, 8, 8, 256)
        
        x = nn.ConvTranspose(features=128, kernel_size=(4, 4), strides=(2, 2), padding='SAME')(x)
        x = nn.leaky_relu(x, negative_slope=0.2)  # Output shape: (batch_size, 16, 16, 128)
        
        x = nn.ConvTranspose(features=64, kernel_size=(4, 4), strides=(2, 2), padding='SAME')(x)
        x = nn.leaky_relu(x, negative_slope=0.2)  # Output shape: (batch_size, 32, 32, 64)
        
        x = nn.ConvTranspose(features=3, kernel_size=(3, 3), strides=(1, 1), padding='SAME')(x)
        x = nn.tanh(x)  # Output shape: (batch_size, 32, 32, 3)
        return x


class Discriminator_cond(nn.Module):
    '''
    Conditional Discriminator class for a GAN model.
    Takes both images and labels as input and discriminates between real and fake images conditioned on the labels.

    Methods:
        __call__: The forward pass through the network.
    '''
    
    @nn.compact
    def __call__(self, x, labels):
        '''
        Forward pass through the conditional discriminator.

        Args:
            x (jnp.ndarray): Input image tensor of shape (batch_size, height, width, channels).
            labels (jnp.ndarray): One-hot encoded labels of shape (batch_size, num_classes).

        Returns:
            jnp.ndarray: A scalar score for each image in the batch, conditioned on labels.
        '''
        labels = labels.reshape((labels.shape[0], 1, 1, 10))
        labels = jnp.broadcast_to(labels, (x.shape[0], x.shape[1], x.shape[2], 10))
        x = jnp.concatenate([x, labels], axis=-1)
        
        x = nn.Conv(features=64, kernel_size=(3, 3), strides=(2, 2))(x)
        x = nn.leaky_relu(x, negative_slope=0.2)
        
        x = nn.Conv(features=128, kernel_size=(3, 3), strides=(2, 2))(x)
        x = nn.leaky_relu(x, negative_slope=0.2)
        
        x = nn.Conv(features=256, kernel_size=(3, 3), strides=(2, 2))(x)
        x = nn.leaky_relu(x, negative_slope=0.2)
        
        x = nn.Conv(features=512, kernel_size=(3, 3), strides=(2, 2))(x)
        x = nn.leaky_relu(x, negative_slope=0.2)
        
        x = x.reshape((x.shape[0], -1))  # Flatten the feature map
        x = nn.Dense(features=1)(x)  # Output a single scalar value
        return x


class Generator_cond(nn.Module):
    '''
    Conditional Generator class for a GAN model.
    Takes latent codes and labels to generate conditional images.

    Methods:
        __call__: The forward pass through the network.
    '''
    
    @nn.compact
    def __call__(self, labels, z):
        '''
        Forward pass through the conditional generator.

        Args:
            labels (jnp.ndarray): One-hot encoded labels of shape (batch_size, num_classes).
            z (jnp.ndarray): Latent vectors of shape (batch_size, latent_dim).

        Returns:
            jnp.ndarray: Generated image tensor of shape (batch_size, 32, 32, 3), conditioned on labels.
        '''
        x = jnp.concatenate([z, labels], axis=-1)
        x = nn.Dense(features=4 * 4 * 512)(x)
        x = nn.leaky_relu(x, negative_slope=0.2)
        x = x.reshape((x.shape[0], 4, 4, 512))  # Reshape to start conv layers
        
        x = nn.ConvTranspose(features=256, kernel_size=(4, 4), strides=(2, 2), padding='SAME')(x)
        x = nn.leaky_relu(x, negative_slope=0.2)  # Output shape: (batch_size, 8, 8, 256)
        
        x = nn.ConvTranspose(features=128, kernel_size=(4, 4), strides=(2, 2), padding='SAME')(x)
        x = nn.leaky_relu(x, negative_slope=0.2)  # Output shape: (batch_size, 16, 16, 128)
        
        x = nn.ConvTranspose(features=64, kernel_size=(4, 4), strides=(2, 2), padding='SAME')(x)
        x = nn.leaky_relu(x, negative_slope=0.2)  # Output shape: (batch_size, 32, 32, 64)
        
        x = nn.ConvTranspose(features=3, kernel_size=(3, 3), strides=(1, 1), padding='SAME')(x)
        x = nn.tanh(x)  # Output shape: (batch_size, 32, 32, 3)
        return x
