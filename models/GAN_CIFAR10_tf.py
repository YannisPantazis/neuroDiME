import tensorflow as tf
from keras import layers

def avg_pool2d(x):
    '''
    Implements a twice-differentiable 2x2 average pooling operation.

    Args:
        x (tf.Tensor): Input tensor of shape (batch_size, height, width, channels).

    Returns:
        tf.Tensor: Averaged pooled tensor.
    '''
    return (x[:, ::2, ::2, :] + x[:, 1::2, ::2, :] + x[:, ::2, 1::2, :] + x[:, 1::2, 1::2, :]) / 4.0


class GeneratorBlock(tf.keras.layers.Layer):
    '''
    ResNet-style block for the generator model, with optional upsampling.

    Args:
        in_chans (int): Number of input channels.
        out_chans (int): Number of output channels.
        upsample (bool): Whether to apply 2x upsampling.

    Methods:
        call: Passes input through the generator block.
    '''

    def __init__(self, in_chans, out_chans, upsample=False, name=None, **kwargs):
        super(GeneratorBlock, self).__init__(name=name, **kwargs)
        self.in_chans = in_chans
        self.out_chans = out_chans
        self.upsample = upsample

        # Define the shortcut convolution if input and output channels differ
        if in_chans != out_chans:
            self.shortcut_conv = layers.Conv2D(out_chans, kernel_size=1, padding='same')
        else:
            self.shortcut_conv = None

        self.bn1 = layers.BatchNormalization()
        self.conv1 = layers.Conv2D(in_chans, kernel_size=3, padding='same')
        self.bn2 = layers.BatchNormalization()
        self.conv2 = layers.Conv2D(out_chans, kernel_size=3, padding='same')

    def build(self, input_shape):
        ''' 
        Build the layer.
        Args:
            input_shape: The shape of the input tensor.
        '''
        if self.upsample:
            self.upsample_layer = layers.UpSampling2D(size=(2, 2))
        self.conv1.build(input_shape)
        output_shape = self.conv1.compute_output_shape(input_shape)
        self.bn1.build(output_shape)
        self.conv2.build(output_shape)
        self.built = True

    def compute_output_shape(self, input_shape):
        ''' 
        Compute output shape of the block.
        Args:
            input_shape: The shape of the input tensor.
        
        Returns:
            Tuple of the output shape.
        '''
        output_shape = list(input_shape)
        if self.upsample:
            output_shape[1] = output_shape[1] * 2
            output_shape[2] = output_shape[2] * 2
        output_shape[3] = self.conv2.filters
        return tuple(output_shape)

    def get_config(self):
        ''' 
        Get the configuration of the block for serialization.
        Returns:
            dict: Configuration dictionary.
        '''
        config = super(GeneratorBlock, self).get_config()
        config.update({
            'in_chans': self.in_chans,
            'out_chans': self.out_chans,
            'upsample': self.upsample,
        })
        return config

    @classmethod
    def from_config(cls, config):
        ''' 
        Instantiate the block from a configuration.
        Args:
            config: Configuration dictionary.
        
        Returns:
            GeneratorBlock: An instance of GeneratorBlock.
        '''
        return cls(**config)

    def call(self, inputs):
        ''' 
        Forward pass of the generator block.
        
        Args:
            inputs (tf.Tensor): Input tensor.
        
        Returns:
            tf.Tensor: Output tensor after passing through the block.
        '''
        x = inputs

        if self.upsample:
            shortcut = tf.image.resize(x, size=[x.shape[1] * 2, x.shape[2] * 2], method='nearest')
        else:
            shortcut = x

        if self.shortcut_conv is not None:
            shortcut = self.shortcut_conv(shortcut)

        x = self.bn1(x)
        x = tf.nn.relu(x)
        if self.upsample:
            x = tf.image.resize(x, size=[x.shape[1] * 2, x.shape[2] * 2], method='nearest')
        x = self.conv1(x)

        x = self.bn2(x)
        x = tf.nn.relu(x)
        x = self.conv2(x)

        return x + shortcut


class Generator(tf.keras.Model):
    '''
    Generator model that consists of a dense layer followed by multiple generator blocks.

    Methods:
        call: Passes input latent vectors through the generator network.
    '''

    def __init__(self):
        super(Generator, self).__init__()

        self.feats = 128
        self.input_linear = layers.Dense(4 * 4 * self.feats)
        self.block1 = GeneratorBlock(self.feats, self.feats, upsample=True)
        self.block2 = GeneratorBlock(self.feats, self.feats, upsample=True)
        self.block3 = GeneratorBlock(self.feats, self.feats, upsample=True)
        self.output_bn = layers.BatchNormalization()
        self.output_conv = layers.Conv2D(3, kernel_size=3, padding='same')

        self._initialize_weights()
        self.last_output = None

    def _initialize_weights(self):
        ''' Apply Xavier initialization (Glorot uniform) to weights. '''
        relu_gain = tf.keras.initializers.VarianceScaling(scale=2.0, mode='fan_in', distribution='truncated_normal')
        for layer in self.layers:
            if isinstance(layer, (layers.Conv2D, layers.Dense)):
                if layer == self.input_linear:
                    initializer = tf.keras.initializers.GlorotUniform()
                else:
                    initializer = relu_gain
                layer.kernel_initializer = initializer
                layer.bias_initializer = tf.zeros_initializer()

    def build(self, input_shape):
        ''' 
        Build the generator.
        Args:
            input_shape: The shape of the input tensor.
        '''
        self.input_linear.build(input_shape)
        output_shape = (input_shape[0], 4, 4, self.feats)
        self.block1.build(output_shape)
        output_shape = self.block1.compute_output_shape(output_shape)
        self.block2.build(output_shape)
        output_shape = self.block2.compute_output_shape(output_shape)
        self.block3.build(output_shape)
        output_shape = self.block3.compute_output_shape(output_shape)
        self.output_bn.build(output_shape)
        self.output_conv.build(output_shape)

        self.built = True

    def call(self, noise, training=True):
        ''' 
        Forward pass of the generator.
        Args:
            noise (tf.Tensor): Latent vectors.
            training (bool): If the model is training.
        
        Returns:
            tf.Tensor: Generated image tensor.
        '''
        z = noise

        x = self.input_linear(z)
        x = tf.reshape(x, [-1, 4, 4, 128])
        x = self.block1(x, training=training)
        x = self.block2(x, training=training)
        x = self.block3(x, training=training)
        x = self.output_bn(x, training=training)
        x = tf.nn.relu(x)
        x = self.output_conv(x)
        x = tf.nn.tanh(x)

        self.last_output = x

        return x


class DiscriminatorBlock(tf.keras.layers.Layer):
    '''ResNet-style block for the discriminator model, with optional downsampling.

    Args:
        in_chans (int): Number of input channels.
        out_chans (int): Number of output channels.
        downsample (bool): Whether to apply 2x downsampling.
        first (bool): Whether this is the first block in the discriminator.

    Methods:
        call: Passes input through the discriminator block.
    '''

    def __init__(self, in_chans, out_chans, downsample=False, first=False):
        super(DiscriminatorBlock, self).__init__()
        self.in_chans = in_chans
        self.out_chans = out_chans
        self.downsample = downsample
        self.first = first

        if in_chans != out_chans:
            self.shortcut_conv = layers.Conv2D(out_chans, kernel_size=1)
        else:
            self.shortcut_conv = None

        self.conv1 = layers.Conv2D(out_chans, kernel_size=3, padding='same')
        self.conv2 = layers.Conv2D(out_chans, kernel_size=3, padding='same')

    def build(self, input_shape):
        ''' 
        Build the layer.
        Args:
            input_shape: The shape of the input tensor.
        '''
        self.conv1.build(input_shape)
        output_shape = self.conv1.compute_output_shape(input_shape)
        self.conv2.build(output_shape)
        self.built = True

    def call(self, inputs, training=False):
        ''' 
        Forward pass of the discriminator block.
        Args:
            inputs (tf.Tensor): Input tensor.
        
        Returns:
            tf.Tensor: Output tensor after passing through the block.
        '''
        x = inputs

        if self.downsample:
            shortcut = avg_pool2d(x)
        else:
            shortcut = x

        if self.shortcut_conv is not None:
            shortcut = self.shortcut_conv(shortcut)

        if not self.first:
            x = tf.nn.relu(x)
        x = self.conv1(x)
        x = tf.nn.relu(x)
        x = self.conv2(x)
        
        if self.downsample:
            x = avg_pool2d(x)
        
        return x + shortcut


class Discriminator(tf.keras.Model):
    '''
    Discriminator model, which evaluates whether inputs are real or fake.

    Methods:
        call: Passes input images through the discriminator network.
    '''

    def __init__(self):
        super(Discriminator, self).__init__()

        feats = 128
        self.block1 = DiscriminatorBlock(3, feats, downsample=True, first=True)
        self.block2 = DiscriminatorBlock(feats, feats, downsample=True)
        self.block3 = DiscriminatorBlock(feats, feats, downsample=False)
        self.block4 = DiscriminatorBlock(feats, feats, downsample=False)
        self.output_linear = layers.Dense(1)

        self._initialize_weights()

    def _initialize_weights(self):
        ''' Initialize the weights of the layers using Xavier initialization. '''
        relu_gain = tf.keras.initializers.VarianceScaling(scale=2.0, mode='fan_in', distribution='truncated_normal')
        for layer in self.layers:
            if isinstance(layer, layers.Conv2D) or isinstance(layer, layers.Dense):
                layer.kernel_initializer = relu_gain
                if hasattr(layer, 'bias') and layer.bias is not None:
                    layer.bias_initializer = tf.zeros_initializer()

    def build(self, input_shape):
        ''' 
        Build the discriminator.
        Args:
            input_shape: The shape of the input tensor.
        '''
        self.block1.build(input_shape)
        self.block2.build(self.block1.compute_output_shape(input_shape))
        self.block3.build(self.block2.compute_output_shape(self.block1.compute_output_shape(input_shape)))
        self.block4.build(self.block3.compute_output_shape(self.block2.compute_output_shape(self.block1.compute_output_shape(input_shape))))
        
        final_shape = self.block4.compute_output_shape(self.block3.compute_output_shape(self.block2.compute_output_shape(self.block1.compute_output_shape(input_shape))))
        final_shape = tf.TensorShape(final_shape).as_list()
        final_shape[1] = final_shape[2] = 1
        
        self.output_linear.build(final_shape)
        self.built = True

    def call(self, inputs, training=True):
        ''' 
        Forward pass of the discriminator.
        Args:
            inputs (tf.Tensor): Input images.
            training (bool): Whether the model is training.
        
        Returns:
            tf.Tensor: Discriminator score for each image.
        '''
        x = inputs

        x = self.block1(x, training=training)
        x = self.block2(x, training=training)
        x = self.block3(x, training=training)
        x = self.block4(x, training=training)
        x = tf.nn.relu(x)
        x = tf.reduce_mean(x, axis=[1, 2])  # Global average pooling
        x = self.output_linear(x)

        return x
