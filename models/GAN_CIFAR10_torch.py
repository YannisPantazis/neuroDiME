import torch.nn as nn
from collections import OrderedDict as OrderedDict
import torch.nn.init as nninit
import torch

def avg_pool2d(x):
    '''
    Implements a twice-differentiable 2x2 average pooling operation.

    Args:
        x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width).

    Returns:
        torch.Tensor: Averaged pooled tensor.
    '''
    return (x[:, :, ::2, ::2] + x[:, :, 1::2, ::2] + x[:, :, ::2, 1::2] + x[:, :, 1::2, 1::2]) / 4


class GeneratorBlock(nn.Module):
    '''
    ResNet-style block for the generator model, with optional upsampling.

    Args:
        in_chans (int): Number of input channels.
        out_chans (int): Number of output channels.
        upsample (bool): Whether to apply 2x upsampling.

    Methods:
        forward: Passes input through the generator block.
    '''

    def __init__(self, in_chans, out_chans, upsample=False):
        super().__init__()

        self.upsample = upsample

        # Define a shortcut connection for ResNet-style skip connection
        if in_chans != out_chans:
            self.shortcut_conv = nn.Conv2d(in_chans, out_chans, kernel_size=1)
        else:
            self.shortcut_conv = None
        self.bn1 = nn.BatchNorm2d(in_chans)
        self.conv1 = nn.Conv2d(in_chans, in_chans, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(in_chans)
        self.conv2 = nn.Conv2d(in_chans, out_chans, kernel_size=3, padding=1)

    def forward(self, *inputs):
        '''
        Forward pass of the generator block.

        Args:
            inputs (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after passing through the block.
        '''
        x = inputs[0]

        if self.upsample:
            shortcut = nn.functional.upsample(x, scale_factor=2, mode='nearest')
        else:
            shortcut = x

        if self.shortcut_conv is not None:
            shortcut = self.shortcut_conv(shortcut)

        x = self.bn1(x)
        x = nn.functional.relu(x, inplace=False)
        if self.upsample:
            x = nn.functional.upsample(x, scale_factor=2, mode='nearest')
        x = self.conv1(x)
        x = self.bn2(x)
        x = nn.functional.relu(x, inplace=False)
        x = self.conv2(x)

        return x + shortcut


class Generator(nn.Module):
    '''
    Generator model that consists of linear layers followed by multiple generator blocks.

    Methods:
        forward: Passes input latent vectors through the generator network.
    '''

    def __init__(self):
        super().__init__()

        feats = 128
        self.input_linear = nn.Linear(128, 4 * 4 * feats)
        self.block1 = GeneratorBlock(feats, feats, upsample=True)
        self.block2 = GeneratorBlock(feats, feats, upsample=True)
        self.block3 = GeneratorBlock(feats, feats, upsample=True)
        self.output_bn = nn.BatchNorm2d(feats)
        self.output_conv = nn.Conv2d(feats, 3, kernel_size=3, padding=1)

        # Apply Xavier initialization to the weights
        relu_gain = nninit.calculate_gain('relu')
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                gain = relu_gain if module != self.input_linear else 1.0
                nninit.xavier_uniform_(module.weight.data, gain=gain)
                module.bias.data.zero_()

        self.last_output = None

    def forward(self, *inputs, labels=None):
        '''
        Forward pass of the generator.

        Args:
            inputs (torch.Tensor): Latent vectors.
            labels (torch.Tensor): (Optional) Labels, if required.

        Returns:
            torch.Tensor: Generated image tensor.
        '''
        x = inputs[0]
        
        x = self.input_linear(x)
        x = x.view(-1, 128, 4, 4)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.output_bn(x)
        x = nn.functional.relu(x, inplace=False)
        x = self.output_conv(x)
        x = nn.functional.tanh(x)

        self.last_output = x

        return x


class DiscriminatorBlock(nn.Module):
    '''
    ResNet-style block for the discriminator model, with optional downsampling.

    Args:
        in_chans (int): Number of input channels.
        out_chans (int): Number of output channels.
        downsample (bool): Whether to apply 2x downsampling.
        first (bool): Whether this is the first block in the discriminator.

    Methods:
        forward: Passes input through the discriminator block.
    '''

    def __init__(self, in_chans, out_chans, downsample=False, first=False):
        super().__init__()

        self.downsample = downsample
        self.first = first

        if in_chans != out_chans:
            self.shortcut_conv = nn.Conv2d(in_chans, out_chans, kernel_size=1)
        else:
            self.shortcut_conv = None
        self.conv1 = nn.Conv2d(in_chans, out_chans, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_chans, out_chans, kernel_size=3, padding=1)

    def forward(self, *inputs):
        '''
        Forward pass of the discriminator block.

        Args:
            inputs (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after passing through the block.
        '''
        x = inputs[0]

        if self.downsample:
            shortcut = avg_pool2d(x)
        else:
            shortcut = x

        if self.shortcut_conv is not None:
            shortcut = self.shortcut_conv(shortcut)

        if not self.first:
            x = nn.functional.relu(x, inplace=False)
        x = self.conv1(x)
        x = nn.functional.relu(x, inplace=False)
        x = self.conv2(x)
        if self.downsample:
            x = avg_pool2d(x)

        return x + shortcut


class Discriminator(nn.Module):
    '''
    Discriminator (or critic) model for GAN.

    Methods:
        forward: Passes input images through the discriminator.
    '''

    def __init__(self):
        super().__init__()

        feats = 128
        self.block1 = DiscriminatorBlock(3, feats, downsample=True, first=True)
        self.block2 = DiscriminatorBlock(feats, feats, downsample=True)
        self.block3 = DiscriminatorBlock(feats, feats, downsample=False)
        self.block4 = DiscriminatorBlock(feats, feats, downsample=False)
        self.output_linear = nn.Linear(128, 1)

        # Apply Xavier initialization to the weights
        relu_gain = nninit.calculate_gain('relu')
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                gain = relu_gain if module != self.block1.conv1 else 1.0
                nninit.xavier_uniform_(module.weight.data, gain=gain)
                module.bias.data.zero_()

    def forward(self, *inputs, labels=None):
        '''
        Forward pass of the discriminator.

        Args:
            inputs (torch.Tensor): Input image tensor.

        Returns:
            torch.Tensor: Scalar score for each image.
        '''
        x = inputs[0]

        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = nn.functional.relu(x, inplace=False)
        x = x.mean(-1, keepdim=False).mean(-1, keepdim=False)
        x = x.view(-1, 128)
        x = self.output_linear(x)

        return x


class Discriminator_cond(nn.Module):
    '''
    Conditional Discriminator (or critic) model for conditional GAN.

    Methods:
        forward: Passes input images and labels through the discriminator.
    '''

    def __init__(self):
        super().__init__()

        feats = 128
        num_classes = 10
        label_emb_size = 50
        self.block1 = DiscriminatorBlock(3 + label_emb_size, feats, downsample=True, first=True)
        self.block2 = DiscriminatorBlock(feats, feats, downsample=True)
        self.block3 = DiscriminatorBlock(feats, feats, downsample=False)
        self.block4 = DiscriminatorBlock(feats, feats, downsample=False)
        self.output_linear = nn.Linear(128, 1)

        self.label_embedding = nn.Embedding(num_classes, label_emb_size)
        
        # Apply Xavier initialization to the weights
        relu_gain = nninit.calculate_gain('relu')
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                gain = relu_gain if module != self.block1.conv1 else 1.0
                nninit.xavier_uniform_(module.weight.data, gain=gain)
                module.bias.data.zero_()

    def forward(self, x, labels):
        '''
        Forward pass of the conditional discriminator.

        Args:
            x (torch.Tensor): Input image tensor.
            labels (torch.Tensor): One-hot encoded labels.

        Returns:
            torch.Tensor: Scalar score for each image.
        '''
        label_emb = self.label_embedding(labels)
        label_emb = label_emb.view(label_emb.size(0), label_emb.size(1), 1, 1)
        label_emb = label_emb.expand(label_emb.size(0), label_emb.size(1), x.size(2), x.size(3))
        
        x = torch.cat([x, label_emb], dim=1)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = nn.functional.relu(x, inplace=False)
        x = x.mean(-1, keepdim=False).mean(-1, keepdim=False)
        x = x.view(-1, 128)
        x = self.output_linear(x)

        return x


class Generator_cond(nn.Module):
    '''
    Conditional Generator model for conditional GAN.

    Methods:
        forward: Passes input latent vectors and labels through the generator.
    '''

    def __init__(self):
        super().__init__()

        feats = 128
        num_classes = 10
        label_emb_size = 50
        self.label_embedding = nn.Embedding(num_classes, label_emb_size)
        self.input_linear = nn.Linear(128 + label_emb_size, 4 * 4 * feats)
        self.block1 = GeneratorBlock(feats, feats, upsample=True)
        self.block2 = GeneratorBlock(feats, feats, upsample=True)
        self.block3 = GeneratorBlock(feats, feats, upsample=True)
        self.output_bn = nn.BatchNorm2d(feats)
        self.output_conv = nn.Conv2d(feats, 3, kernel_size=3, padding=1)

        # Apply Xavier initialization to the weights
        relu_gain = nninit.calculate_gain('relu')
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                gain = relu_gain if module != self.input_linear else 1.0
                nninit.xavier_uniform_(module.weight.data, gain=gain)
                module.bias.data.zero_()

        self.last_output = None

    def forward(self, z, labels):
        '''
        Forward pass of the conditional generator.

        Args:
            z (torch.Tensor): Latent vectors.
            labels (torch.Tensor): One-hot encoded labels.

        Returns:
            torch.Tensor: Generated image tensor.
        '''
        label_emb = self.label_embedding(labels)
        x = torch.cat([z, label_emb], dim=1)
        x = self.input_linear(x)
        x = x.view(-1, 128, 4, 4)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.output_bn(x)
        x = nn.functional.relu(x, inplace=False)
        x = self.output_conv(x)
        x = nn.functional.tanh(x)

        self.last_output = x

        return x
