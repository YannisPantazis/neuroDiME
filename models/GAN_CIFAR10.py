import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.layers import BatchNormalization, LayerNormalization, Embedding
from tensorflow.keras.models import Sequential
import tensorflow_addons as tfa

DIM_G = 128  # Generator dimensionality
DIM_D = 128  # Critic dimensionality
NORMALIZATION_G = True  # Use batchnorm in generator?
NORMALIZATION_D = False  # Use batchnorm (or layernorm) in critic? This doesn't do anything at the moment.
CONDITIONAL = False  # Whether to train a conditional or unconditional model
ACGAN = False  # If CONDITIONAL, whether to use ACGAN or "vanilla" conditioning
ACGAN_SCALE = 1.  # How to scale the critic's ACGAN loss relative to WGAN loss
ACGAN_SCALE_G = 0.1  # How to scale generator's ACGAN loss relative to WGAN loss


class ConditionalBatchNorm2d(layers.Layer):
    def __init__(self, num_features, num_classes):
        super(ConditionalBatchNorm2d, self).__init__()
        self.bn = BatchNormalization(center=False, scale=False)
        self.embed = Embedding(num_classes, num_features * 2)
        self.embed.build((None,))  # Build the embedding layer with an unspecified batch size
        self.embed.set_weights([tf.concat([tf.ones((num_classes, num_features)), tf.zeros((num_classes, num_features))], axis=1)])

    def call(self, x, y):
        out = self.bn(x, training=False)
        gamma, beta = tf.split(self.embed(y), 2, axis=1)
        gamma = tf.reshape(gamma, [-1, 1, 1, gamma.shape[-1]])
        beta = tf.reshape(beta, [-1, 1, 1, beta.shape[-1]])
        out = gamma * out + beta
        return out


class Normalize(layers.Layer):
    def __init__(self, input_dim, n_labels=10, CONDITIONAL=False, ACGAN=False, NORMALIZATION_D=False, NORMALIZATION_G=False):
        super(Normalize, self).__init__()
        self.conditional = CONDITIONAL and not ACGAN
        self.normalization_d = NORMALIZATION_D
        self.normalization_g = NORMALIZATION_G
        self.n_labels = n_labels
        self.input_dim = input_dim

        if self.normalization_d:
            self.norm_layer = LayerNormalization(axis=[1, 2, 3])
        elif self.normalization_g and self.conditional:
            self.norm_layer = ConditionalBatchNorm2d(input_dim, n_labels)
        elif self.normalization_g:
            self.norm_layer = BatchNormalization()
        else:
            self.norm_layer = None

    def call(self, x, labels=None):
        if self.norm_layer is None:
            return x

        if self.conditional and labels is not None:
            return self.norm_layer(x, labels)
        else:
            return self.norm_layer(x)


class ConvMeanPool(layers.Layer):
    """Class that encapsulates the convolution operation followed by the mean pooling operation"""

    def __init__(self, output_dim, filter_size, he_init=True, biases=True):
        super(ConvMeanPool, self).__init__()
        
        self.conv = layers.Conv2D(filters=output_dim,
                                  kernel_size=filter_size,
                                  strides=1,
                                  padding='same',  # Assuming 'same' padding
                                  use_bias=biases,
                                  kernel_initializer='he_normal' if he_init else 'glorot_uniform')
        
        if biases:
            self.conv.bias_initializer = tf.zeros_initializer()

    def call(self, inputs):
        output = self.conv(inputs)
        
        # Mean pooling operation
        output = (output[:, ::2, ::2, :] + 
                  output[:, 1::2, ::2, :] + 
                  output[:, ::2, 1::2, :] + 
                  output[:, 1::2, 1::2, :]) / 4.0
        
        return output
    
    
class MeanPoolConv(layers.Layer):
    """Class that encapsulates the mean pooling operation followed by the convolution operation"""

    def __init__(self, output_dim, filter_size, he_init=True, biases=True):
        super(MeanPoolConv, self).__init__()
        
        self.conv = layers.Conv2D(filters=output_dim,
                                  kernel_size=filter_size,
                                  strides=1,
                                  padding='same',  # Assuming 'same' padding
                                  use_bias=biases,
                                  kernel_initializer='he_normal' if he_init else 'glorot_uniform')
        
        if biases:
            self.conv.bias_initializer = tf.zeros_initializer()

    def call(self, inputs):
        # Mean pooling operation
        pooled_output = (inputs[:, ::2, ::2, :] + 
                         inputs[:, 1::2, ::2, :] + 
                         inputs[:, ::2, 1::2, :] + 
                         inputs[:, 1::2, 1::2, :]) / 4.0
        
        # Convolution
        output = self.conv(pooled_output)
        
        return output
    
    
class MeanPoolConvSpecNorm(layers.Layer):
    """Class that encapsulates the mean pooling operation followed by the convolution operation with spectral normalization"""

    def __init__(self, output_dim, filter_size, he_init=True, biases=True):
        super(MeanPoolConvSpecNorm, self).__init__()
        
        self.conv = layers.Conv2D(filters=output_dim,
                                  kernel_size=filter_size,
                                  strides=1,
                                  padding='same',  # Assuming 'same' padding
                                  use_bias=biases,
                                  kernel_initializer='he_normal' if he_init else 'glorot_uniform')
        
        if biases:
            self.conv.bias_initializer = tf.zeros_initializer()
        
        # Apply spectral normalization
        self.conv = tfa.layers.SpectralNormalization(self.conv)

    def call(self, inputs):
        # Mean pooling operation
        pooled_output = (inputs[:, ::2, ::2, :] + 
                         inputs[:, 1::2, ::2, :] + 
                         inputs[:, ::2, 1::2, :] + 
                         inputs[:, 1::2, 1::2, :]) / 4.0
        
        # Convolution with spectral normalization
        output = self.conv(pooled_output)
        
        return output


class ConvMeanPoolSpecNorm(tf.keras.layers.Layer):
    """Class that encapsulates the convolution operation with spectral normalization followed by the mean pooling operation"""

    def __init__(self, output_dim, filter_size, he_init=True, biases=True):
        super(ConvMeanPoolSpecNorm, self).__init__()

        self.conv = layers.Conv2D(filters=output_dim,
                                  kernel_size=filter_size,
                                  strides=1,
                                  padding='same',  # Assuming 'same' padding
                                  use_bias=biases,
                                  kernel_initializer='he_normal' if he_init else 'glorot_uniform')

        if biases:
            self.conv.bias_initializer = tf.zeros_initializer()

        # Apply spectral normalization
        self.conv = tfa.layers.SpectralNormalization(self.conv)

    def call(self, inputs):
        output = self.conv(inputs)

        # Mean Pooling Operation
        output = (output[:, ::2, ::2, :] + output[:, 1::2, ::2, :] +
                  output[:, ::2, 1::2, :] + output[:, 1::2, 1::2, :]) / 4

        return output


class UpsampleConv(tf.keras.layers.Layer):
    def __init__(self, output_dim, filter_size, he_init=True, biases=True):
        super(UpsampleConv, self).__init__()
        
        self.conv = layers.Conv2D(filters=output_dim,
                                  kernel_size=filter_size,
                                  strides=1,
                                  padding='same',  # Assuming 'same' padding
                                  use_bias=biases,
                                  kernel_initializer='he_normal' if he_init else 'glorot_uniform')

        if biases:
            self.conv.bias_initializer = tf.zeros_initializer()

    def call(self, inputs):
        # Upsampling operation
        # Replicate the input tensor
        output = tf.concat([inputs, inputs, inputs, inputs], axis=-1)
        
        # Apply pixel shuffle for depth-to-space
        output = tf.nn.depth_to_space(output, block_size=2)
        
        # Convolution
        output = self.conv(output)
        
        return output
    

class UpsampleConvSpecNorm(tf.keras.layers.Layer):
    def __init__(self, output_dim, filter_size, he_init=True, biases=True):
        super(UpsampleConvSpecNorm, self).__init__()
        
        self.conv = layers.Conv2D(filters=output_dim,
                                  kernel_size=filter_size,
                                  strides=1,
                                  padding='same',  # Assuming 'same' padding
                                  use_bias=biases,
                                  kernel_initializer='he_normal' if he_init else 'glorot_uniform')

        if biases:
            self.conv.bias_initializer = tf.zeros_initializer()

        # Apply spectral normalization
        self.conv = tfa.layers.SpectralNormalization(self.conv)

    def call(self, inputs):
        # Upsampling operation
        # Replicate the input tensor
        output = tf.concat([inputs, inputs, inputs, inputs], axis=-1)
        
        # Apply pixel shuffle for depth-to-space
        output = tf.nn.depth_to_space(output, block_size=2)
        
        # Convolution with spectral normalization
        output = self.conv(output)
        
        return output


class ResidualBlock(tf.keras.layers.Layer):
    def __init__(self, input_dim, output_dim, filter_size, resample=None, no_dropout=False, labels=None):
        super(ResidualBlock, self).__init__()

        self.resample = resample

        if resample == 'down':
            self.conv_1 = layers.Conv2D(input_dim, filter_size, padding='same')
            self.conv_2 = ConvMeanPool(output_dim, filter_size)
            self.conv_shortcut = ConvMeanPool(output_dim, 1, he_init=False, biases=True)
        elif resample == 'up':
            self.conv_1 = UpsampleConv(output_dim, filter_size)
            self.conv_2 = layers.Conv2D(output_dim, filter_size, padding='same')
            self.conv_shortcut = UpsampleConv(output_dim, 1, he_init=False, biases=True)
        elif resample is None:
            self.conv_1 = layers.Conv2D(input_dim, filter_size, padding='same')
            self.conv_2 = layers.Conv2D(output_dim, filter_size, padding='same')
            self.conv_shortcut = layers.Conv2D(output_dim, 1, padding='valid')
        else:
            raise Exception('invalid resample value')

        self.normalize1 = Normalize(input_dim)
        self.normalize2 = Normalize(output_dim)

        self.shortcut = None
        if output_dim != input_dim or resample is not None:
            self.shortcut = self.conv_shortcut

    def call(self, inputs, labels=None):
        shortcut = inputs if self.shortcut is None else self.shortcut(inputs)

        output = self.normalize1(inputs, labels)
        output = tf.nn.relu(output)
        output = self.conv_1(output)

        output = self.normalize2(output, labels)
        output = tf.nn.relu(output)
        output = self.conv_2(output)

        return shortcut + output


class ConvBlockSpecNorm(tf.keras.layers.Layer):
    def __init__(self, input_dim, output_dim, filter_size, resample=None, no_dropout=False):
        super(ConvBlockSpecNorm, self).__init__()

        if resample == 'down':
            self.conv_1 = tfa.layers.SpectralNormalization(
                layers.Conv2D(input_dim, filter_size, padding='same')
            )
            self.conv_2 = ConvMeanPoolSpecNorm(output_dim, filter_size)
        elif resample == 'up':
            self.conv_1 = UpsampleConvSpecNorm(output_dim, filter_size)
            self.conv_2 = tfa.layers.SpectralNormalization(
                layers.Conv2D(output_dim, filter_size, padding='same')
            )
        elif resample is None:
            self.conv_1 = tfa.layers.SpectralNormalization(
                layers.Conv2D(input_dim, filter_size, padding='same')
            )
            self.conv_2 = tfa.layers.SpectralNormalization(
                layers.Conv2D(output_dim, filter_size, padding='same')
            )
        else:
            raise Exception('invalid resample value')

    def call(self, inputs):
        output = tf.nn.relu(inputs)
        output = self.conv_1(output)
        output = tf.nn.relu(output)
        output = self.conv_2(output)
        return output


class OptimizedConvBlockDisc1(tf.keras.layers.Layer):
    def __init__(self, dim_d):
        super(OptimizedConvBlockDisc1, self).__init__()
        self.conv_1 = layers.Conv2D(filters=dim_d, kernel_size=3, strides=1, padding='same')
        self.conv_2 = ConvMeanPool(output_dim=dim_d, filter_size=3)
        self.conv_shortcut = MeanPoolConv(output_dim=dim_d, filter_size=1, he_init=False, biases=True)

    def call(self, inputs):
        shortcut = self.conv_shortcut(inputs)

        output = self.conv_1(inputs)
        output = tf.nn.relu(output)
        output = self.conv_2(output)

        return shortcut + output


class OptimizedConvBlockDisc1SpecNorm(tf.keras.layers.Layer):
    def __init__(self, dim_d):
        super(OptimizedConvBlockDisc1SpecNorm, self).__init__()
        self.conv_1 = tfa.layers.SpectralNormalization(
            layers.Conv2D(filters=dim_d, kernel_size=3, strides=1, padding='same')
        )
        self.conv_2 = ConvMeanPoolSpecNorm(output_dim=dim_d, filter_size=3)

    def call(self, inputs):
        output = self.conv_1(inputs)
        output = tf.nn.relu(output)
        output = self.conv_2(output)
        return output


class Generator(tf.keras.Model):
    def __init__(self, dim_g, output_dim):
        super(Generator, self).__init__()
        self.dim_g = dim_g
        self.output_dim = output_dim
        self.linear = layers.Dense(4*4*dim_g)
        self.res_block1 = ResidualBlock(dim_g, dim_g, 3, resample='up')
        self.res_block2 = ResidualBlock(dim_g, dim_g, 3, resample='up')
        self.res_block3 = ResidualBlock(dim_g, dim_g, 3, resample='up')
        self.norm = Normalize(dim_g)
        self.conv = layers.Conv2D(3, 3, padding='same')

    def call(self, n_samples, labels=None, noise=None):
        if noise is None:
            noise = tf.random.normal([n_samples, 128])
        output = self.linear(noise)
        output = tf.reshape(output, [-1, 4, 4, self.dim_g])

        output = self.res_block1(output, labels=labels)
        output = self.res_block2(output, labels=labels)
        output = self.res_block3(output, labels=labels)
        
        output = self.norm(output)
        output = tf.nn.relu(output)
        output = self.conv(output)
        output = tf.nn.tanh(output)
        return tf.reshape(output, [-1, self.output_dim])


class Discriminator(tf.keras.Model):
    def __init__(self, dim_d):
        super(Discriminator, self).__init__()
        self.optimized_block = OptimizedConvBlockDisc1SpecNorm(dim_d)
        self.conv_block_2 = ConvBlockSpecNorm(dim_d, dim_d, 3, resample='down')
        self.conv_block_3 = ConvBlockSpecNorm(dim_d, dim_d, 3, resample=None)
        self.conv_block_4 = ConvBlockSpecNorm(dim_d, dim_d, 3, resample=None)
        self.linear = tfa.layers.SpectralNormalization(
            layers.Dense(1)
        )

    def call(self, inputs, labels=None):
        output = tf.reshape(inputs, [-1, 32, 32, 3])
        output = self.optimized_block(output)
        output = self.conv_block_2(output)
        output = self.conv_block_3(output)
        output = self.conv_block_4(output)
        
        output = tf.nn.relu(output)
        output = tf.reduce_mean(output, axis=[1, 2])
        output = self.linear(output)
        return tf.reshape(output, [-1]), None


# For alpha > 1
def f_alpha_star(y, alpha):
    return (tf.pow(tf.nn.relu(y), alpha / (alpha - 1.0)) * tf.pow((alpha - 1.0), alpha / (alpha - 1.0)) / alpha + 1 / (alpha * (alpha - 1.0)))
