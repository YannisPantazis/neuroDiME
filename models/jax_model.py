import jax
import jax.numpy as jnp
from jax import random
from flax import linen as nn


class Discriminator(nn.Module):
    """Discriminator Class which is responsible of initializing the discriminator"""
    
    input_dim: int
    spec_norm: bool
    bounded: bool
    layers_list: list

    @nn.compact
    def __call__(self, x):

        if self.spec_norm:
            for h_dim in self.layers_list:
                x = nn.SpectralNorm(nn.Dense(h_dim))(x)
                x = nn.relu(x)
            x = nn.SpectralNorm(nn.Dense(1))(x)
        else:
            for h_dim in self.layers_list:
                x = nn.Dense(h_dim)(x)
                x = nn.relu(x)
            x = nn.Dense(1)(x)

        if self.bounded:
            x = bounded_activation(x)

        return x


class Generator(nn.Module):
    """Generator Class which is responsible of initializing the generator"""

    X_dim: int
    Z_dim: int
    spec_norm: bool
    layers_list: list

    @nn.compact
    def __call__(self, x):

        if self.spec_norm:
            for h_dim in self.layers_list:
                x = nn.SpectralNorm(nn.Dense(h_dim))(x)
                x = nn.relu(x)
            x = nn.SpectralNorm(nn.Dense(self.X_dim))
        else:
            for h_dim in self.layers_list:
                x = nn.Dense(h_dim)(x)
                x = nn.relu(x)
            x = nn.Dense(self.X_dim)(x)
        
        return x


def bounded_activation(x):
    M = 100.0
    return M * jnp.tanh(x / M)