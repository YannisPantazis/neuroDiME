import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.training import train_state
from jax import random
import optax


class Generator_MNIST_cond(nn.Module):
    latent_dim: int = 118  # Dimension of latent vector z
    num_classes: int = 10  # Number of classes for labels (MNIST has 10)

    @nn.compact
    def __call__(self, labels, z):
        """
        Args:
            labels: One-hot encoded class labels, shape (batch_size, num_classes).
            z: Latent vector, shape (batch_size, latent_dim).
        
        Returns:
            Generated image, shape (batch_size, 28, 28, 1).
        """
        # Concatenate latent vector z and labels
        x = jnp.concatenate([z, labels], axis=-1)

        # Fully connected layers with ReLU activations
        x = nn.Dense(256)(x)
        x = nn.relu(x)
        x = nn.Dense(512)(x)
        x = nn.relu(x)
        x = nn.Dense(1024)(x)
        x = nn.relu(x)
        x = nn.Dense(7 * 7 * 128)(x)
        x = nn.relu(x)

        # Reshape to 7x7 image with 128 channels
        x = x.reshape((-1, 7, 7, 128))

        # Transposed convolution layers (to upscale the image)
        x = nn.ConvTranspose(features=64, kernel_size=(5, 5), strides=(2, 2), padding='SAME')(x)
        x = nn.relu(x)
        x = nn.ConvTranspose(features=32, kernel_size=(5, 5), strides=(2, 2), padding='SAME')(x)
        x = nn.relu(x)
        x = nn.ConvTranspose(features=1, kernel_size=(5, 5), strides=(1, 1), padding='SAME')(x)

        # Tanh activation to output images in range [-1, 1]
        x = nn.tanh(x)

        return x


class Discriminator_MNIST_cond(nn.Module):
    @nn.compact
    def __call__(self, x, labels, training: bool = True):
        """
        Args:
            x: Input images, shape (batch_size, 28, 28, 1).
            labels: One-hot encoded class labels, shape (batch_size, num_classes).
            training: Whether the model is in training mode (for Dropout).
        
        Returns:
            Output logit, shape (batch_size, 1).
        """
        # Tile the labels and concatenate with the input image along the channel axis
        labels = jnp.tile(labels[:, None, None, :], (1, 28, 28, 1))
        x = jnp.concatenate([x, labels], axis=-1)

        # Convolutional layers with Leaky ReLU activations
        x = nn.Conv(features=64, kernel_size=(5, 5), strides=(2, 2), padding='SAME')(x)
        x = nn.leaky_relu(x, negative_slope=0.2)
        x = nn.Dropout(0.3, deterministic=not training)(x)
        x = nn.Conv(features=128, kernel_size=(5, 5), strides=(2, 2), padding='SAME')(x)
        x = nn.leaky_relu(x, negative_slope=0.2)
        x = nn.Dropout(0.3, deterministic=not training)(x)

        # Flatten the feature map
        x = x.reshape((x.shape[0], -1))

        # Fully connected layer followed by output layer
        x = nn.Dense(1024)(x)
        x = nn.leaky_relu(x, negative_slope=0.2)
        x = nn.Dense(1)(x)  # Output logit (real/fake classification)

        return x
