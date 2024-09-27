import tensorflow as tf
from keras import layers


def Generator_MNIST_cond(latent_dim=118):
    """
    Conditional Generator model for MNIST dataset.

    Args:
        latent_dim (int): Dimension of the latent input vector. Default is 118.

    Returns:
        Model: A Keras model that takes as input a latent vector and a one-hot encoded label and outputs a generated image.
    
    Input:
        label (Tensor): One-hot encoded label of shape (batch_size, 10).
        z (Tensor): Latent input vector of shape (batch_size, latent_dim).

    Output:
        Tensor: Generated image of shape (batch_size, 24, 24, 1).
    """

    # Input for the label (conditional input)
    label = tf.keras.layers.Input(shape=(10,), dtype='float32')
    z = tf.keras.layers.Input(shape=(latent_dim,), dtype='float32')
        
    # Concatenate latent vector z and conditional label
    x = tf.keras.layers.concatenate([z, label])
    
    # First dense layer followed by reshaping
    x = layers.Dense(256)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.2)(x)
    x = tf.keras.layers.Reshape((4, 4, 16))(x)  # Adjust reshaping to match reduced dimensions
    
    # First convolutional layer
    x = layers.Conv2D(64, kernel_size=3, strides=1, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.2)(x)
    
    # First transpose convolutional layer
    x = layers.Conv2DTranspose(128, kernel_size=3, strides=2, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.2)(x)
    
    # Second transpose convolutional layer
    x = layers.Conv2DTranspose(256, kernel_size=3, strides=2, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.2)(x)
    
    # Third transpose convolutional layer
    x = layers.Conv2DTranspose(512, kernel_size=3, strides=2, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.2)(x)
    
    # Second convolutional layer
    x = layers.Conv2D(128, kernel_size=3, strides=1, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.2)(x)
    
    # Third convolutional layer
    x = layers.Conv2D(32, kernel_size=3, strides=1, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.2)(x)
    
    # Fourth convolutional layer
    x = layers.Conv2D(8, kernel_size=3, strides=1, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.2)(x)
    
    # Final convolutional layer with sigmoid activation
    x = layers.Conv2D(1, kernel_size=3, strides=1, padding='same')(x)
    out = layers.Activation('sigmoid')(x)
    
    # Cropping the output 
    out = out[:, 2:-2, 2:-2, :]
    
    # Create the model
    return tf.keras.Model(inputs=[label, z], outputs=out)


def Discriminator_MNIST_cond():
    """
    Conditional Discriminator model for MNIST dataset.

    Returns:
        Model: A Keras model that takes as input an image and a one-hot encoded label and outputs a scalar value indicating real or fake.

    Input:
        x_input (Tensor): Input MNIST image of shape (batch_size, 28, 28).
        z_input (Tensor): One-hot encoded label of shape (batch_size, 10).

    Output:
        Tensor: Discriminator output scalar for each image in the batch.
    """

    # Input for the image (flattened)
    x_input = tf.keras.layers.Input(shape=(28, 28), dtype='float32')
    z_input = tf.keras.layers.Input(shape=(10,), dtype='float32')  # Conditional input
    
    # Flatten the image input
    x = tf.keras.layers.Reshape((784,))(x_input)
    
    # Concatenate the flattened image with the conditional input
    y = tf.keras.layers.concatenate([z_input, x])
    
    # Dense layers with Layer Normalization and LeakyReLU activation
    w = layers.Dense(794)(y)
    w = layers.LayerNormalization()(w)
    w = layers.LeakyReLU(alpha=0.2)(w)
    
    w = layers.Dense(794)(w)
    w = layers.LayerNormalization()(w)
    w = layers.LeakyReLU(alpha=0.2)(w)
    
    w = layers.Dense(256)(w)
    w = layers.LayerNormalization()(w)
    w = layers.LeakyReLU(alpha=0.2)(w)
    
    w = layers.Dense(128)(w)
    w = layers.LayerNormalization()(w)
    w = layers.LeakyReLU(alpha=0.2)(w)
    
    w = layers.Dense(64)(w)
    w = layers.LayerNormalization()(w)
    w = layers.LeakyReLU(alpha=0.2)(w)
    
    w = layers.Dense(32)(w)
    w = layers.LayerNormalization()(w)
    w = layers.LeakyReLU(alpha=0.2)(w)
    
    w = layers.Dense(16)(w)
    w = layers.LeakyReLU(alpha=0.2)(w)
    
    # Output layer (no activation, typical for GAN discriminator output)
    out = layers.Dense(1)(w)
    
    # Create and return the model
    return tf.keras.Model(inputs=[x_input, z_input], outputs=out)

