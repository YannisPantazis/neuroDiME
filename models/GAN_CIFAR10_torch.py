import torch.nn as nn
from torch.nn.utils import spectral_norm
from collections import OrderedDict as OrderedDict
import torch
import torch.nn.functional as F

DIM_G = 128 # Generator dimensionality
DIM_D = 128 # Critic dimensionality
NORMALIZATION_G = True # Use batchnorm in generator?
NORMALIZATION_D = False # Use batchnorm (or layernorm) in critic? This doesn't do anything at the moment.
CONDITIONAL = False # Whether to train a conditional or unconditional model
ACGAN = False # If CONDITIONAL, whether to use ACGAN or "vanilla" conditioning
ACGAN_SCALE = 1. # How to scale the critic's ACGAN loss relative to WGAN loss
ACGAN_SCALE_G = 0.1 # How to scale generator's ACGAN loss relative to WGAN loss



class ConditionalBatchNorm2d(nn.Module):
    def __init__(self, num_features, num_classes):
        super(ConditionalBatchNorm2d, self).__init__()
        self.bn = nn.BatchNorm2d(num_features, affine=False)
        self.embed = nn.Embedding(num_classes, num_features * 2)
        self.embed.weight.data[:, :num_features].fill_(1)  # Initialize scale to 1
        self.embed.weight.data[:, num_features:].zero_()  # Initialize bias to 0

    def forward(self, x, y):
        out = self.bn(x)
        gamma, beta = self.embed(y).chunk(2, dim=1)
        gamma = gamma.unsqueeze(2).unsqueeze(3)
        beta = beta.unsqueeze(2).unsqueeze(3)
        out = gamma * out + beta
        return out


class Normalize(nn.Module):
    def __init__(self, input_dim, n_labels=10, CONDITIONAL=False, ACGAN=False, NORMALIZATION_D=False, NORMALIZATION_G=False):
        super(Normalize, self).__init__()
        self.conditional = CONDITIONAL and not ACGAN
        self.normalization_d = NORMALIZATION_D
        self.normalization_g = NORMALIZATION_G
        self.n_labels = n_labels
        self.input_dim = input_dim

        if self.normalization_d:
            self.norm_layer = nn.LayerNorm([input_dim, 1, 1])
        elif self.normalization_g and self.conditional:
            self.norm_layer = ConditionalBatchNorm2d(input_dim, n_labels)
        elif self.normalization_g:
            self.norm_layer = nn.BatchNorm2d(input_dim)
        else:
            self.norm_layer = None

    def forward(self, x, labels=None):
        if self.norm_layer is None:
            return x

        if self.conditional and labels is not None:
            return self.norm_layer(x, labels)
        else:
            return self.norm_layer(x)


class ConvMeanPool(nn.Module):
    """Class that encapsulates the convolution operation followed by the mean pooling operation"""

    def __init__(self, input_dim, output_dim, filter_size, he_init=True, biases=True):
        super(ConvMeanPool, self).__init__()
        
        self.conv = nn.Conv2d(in_channels=input_dim,
                              out_channels=output_dim,
                              kernel_size=filter_size,
                              stride=1,
                              padding=filter_size//2,  # Assuming 'same' padding
                              bias=biases)
        
        if he_init:
            nn.init.kaiming_normal_(self.conv.weight, nonlinearity='relu')
        if biases:
            nn.init.zeros_(self.conv.bias)
    
    def forward(self, inputs):
        output = self.conv(inputs)
        
        # Mean pooling operation
        output = (output[:, :, ::2, ::2] + 
                  output[:, :, 1::2, ::2] + 
                  output[:, :, ::2, 1::2] + 
                  output[:, :, 1::2, 1::2]) / 4.0
        
        return output


class MeanPoolConv(nn.Module):
    """Class that encapsulates the mean pooling operation followed by the convolution operation"""

    def __init__(self, input_dim, output_dim, filter_size, he_init=True, biases=True):
        super(MeanPoolConv, self).__init__()
        
        self.conv = nn.Conv2d(in_channels=input_dim,
                              out_channels=output_dim,
                              kernel_size=filter_size,
                              stride=1,
                              padding=filter_size//2,  # Assuming 'same' padding
                              bias=biases)
        
        if he_init:
            nn.init.kaiming_normal_(self.conv.weight, nonlinearity='relu')
        if biases:
            nn.init.zeros_(self.conv.bias)
    
    def forward(self, inputs):
        # Mean pooling operation
        pooled_output = (inputs[:, :, ::2, ::2] + 
                         inputs[:, :, 1::2, ::2] + 
                         inputs[:, :, ::2, 1::2] + 
                         inputs[:, :, 1::2, 1::2]) / 4.0
        
        # Convolution
        output = self.conv(pooled_output)
        
        return output


class MeanPoolConvSpecNorm(nn.Module):
    """Class that encapsulates the mean pooling operation followed by the convolution operation with spectral normalization"""

    def __init__(self, input_dim, output_dim, filter_size, he_init=True, biases=True):
        super(MeanPoolConvSpecNorm, self).__init__()
        
        self.conv = spectral_norm(nn.Conv2d(in_channels=input_dim,
                                            out_channels=output_dim,
                                            kernel_size=filter_size,
                                            stride=1,
                                            padding=filter_size//2,  # Assuming 'same' padding
                                            bias=biases))
        
        if he_init:
            nn.init.kaiming_normal_(self.conv.weight, nonlinearity='relu')
        if biases:
            nn.init.zeros_(self.conv.bias)
    
    def forward(self, inputs):
        # Mean pooling operation
        pooled_output = (inputs[:, :, ::2, ::2] + 
                         inputs[:, :, 1::2, ::2] + 
                         inputs[:, :, ::2, 1::2] + 
                         inputs[:, :, 1::2, 1::2]) / 4.0
        
        # Convolution with spectral normalization
        output = self.conv(pooled_output)
        
        return output


class ConvMeanPoolSpecNorm(nn.Module):
    """Class that encapsulates the convolution operation with spectral normalization followed by the mean pooling operation"""

    def __init__(self, input_dim, output_dim, filter_size, he_init=True, biases=True):
        super().__init__()
        
        self.conv = spectral_norm(torch.nn.Conv2d(input_dim, output_dim, filter_size, stride=1, padding=1//2, bias=biases))
        
        if he_init:
            torch.nn.init.kaiming_normal_(self.conv.weight, nonlinearity='relu')
        if biases:
            self.bias = torch.nn.Parameter(torch.zeros(output_dim))
        else:
            self.bias = None

    def forward(self, inputs):
        output = self.conv(inputs)
        
        # Mean Pooling Operation
        output = (output[:, :, ::2, ::2] + output[:, :, 1::2, ::2] + output[:, :, ::2, 1::2] + output[:, :, 1::2, 1::2]) / 4
        
        return output


class UpsampleConv(nn.Module):
    def __init__(self, input_dim, output_dim, filter_size, he_init=True, biases=True):
        super(UpsampleConv, self).__init__()
        
        self.conv = nn.Conv2d(in_channels=input_dim,
                              out_channels=output_dim,
                              kernel_size=filter_size,
                              stride=1,
                              padding=filter_size//2,  # Assuming 'same' padding
                              bias=biases)
        
        if he_init:
            nn.init.kaiming_normal_(self.conv.weight, nonlinearity='relu')
        if biases:
            nn.init.zeros_(self.conv.bias)
    
    def forward(self, inputs):
        # Upsampling operation
        # Replicate the input tensor
        output = torch.cat([inputs, inputs, inputs, inputs], dim=1)
        
        # Apply pixel shuffle for depth-to-space
        output = F.pixel_shuffle(output, upscale_factor=2)
        
        # Convolution
        output = self.conv(output)
        
        return output
    
    
class UpsampleConvSpecNorm(nn.Module):
    def __init__(self, input_dim, output_dim, filter_size, he_init=True, biases=True):
        super(UpsampleConvSpecNorm, self).__init__()
        
        self.conv = spectral_norm(nn.Conv2d(in_channels=input_dim,
                                            out_channels=output_dim,
                                            kernel_size=filter_size,
                                            stride=1,
                                            padding=filter_size//2,  # Assuming 'same' padding
                                            bias=biases))
        
        if he_init:
            nn.init.kaiming_normal_(self.conv.weight, nonlinearity='relu')
        if biases:
            nn.init.zeros_(self.conv.bias)
    
    def forward(self, inputs):
        # Upsampling operation
        # Replicate the input tensor
        output = torch.cat([inputs, inputs, inputs, inputs], dim=1)
        
        # Apply pixel shuffle for depth-to-space
        output = F.pixel_shuffle(output, upscale_factor=2)
        
        # Convolution with spectral normalization
        output = self.conv(output)
        
        return output


class ResidualBlock(nn.Module):
    def __init__(self, input_dim, output_dim, filter_size, resample=None, no_dropout=False, labels=None):
        super(ResidualBlock, self).__init__()
        
        if resample == 'down':
            self.conv_1 = nn.Conv2d(input_dim, input_dim, filter_size, padding=filter_size//2)
            self.conv_2 = ConvMeanPool(input_dim, output_dim, filter_size)
            self.conv_shortcut = ConvMeanPool(input_dim, output_dim, 1, he_init=False, biases=True)
        elif resample == 'up':
            self.conv_1 = UpsampleConv(input_dim, output_dim, filter_size)
            self.conv_2 = nn.Conv2d(output_dim, output_dim, filter_size, padding=filter_size//2)
            self.conv_shortcut = UpsampleConv(input_dim, output_dim, 1, he_init=False, biases=True)
        elif resample is None:
            self.conv_1 = nn.Conv2d(input_dim, output_dim, filter_size, padding=filter_size//2)
            self.conv_2 = nn.Conv2d(output_dim, output_dim, filter_size, padding=filter_size//2)
            self.conv_shortcut = nn.Conv2d(input_dim, output_dim, 1, padding=0)
        else:
            raise Exception('invalid resample value')
        
        self.normalize1 = Normalize(input_dim)
        self.normalize2 = Normalize(output_dim)
        
        self.shortcut = None
        if output_dim != input_dim or resample is not None:
            self.shortcut = self.conv_shortcut

    def forward(self, inputs, labels=None):
        shortcut = inputs if self.shortcut is None else self.shortcut(inputs)

        output = self.normalize1(inputs, labels)
        output = F.relu(output)
        output = self.conv_1(output)
        
        output = self.normalize2(output, labels)
        output = F.relu(output)
        output = self.conv_2(output)
        
        return shortcut + output


class ConvBlockSpecNorm(nn.Module):
    def __init__(self, input_dim, output_dim, filter_size, resample=None, no_dropout=False):
        super(ConvBlockSpecNorm, self).__init__()
        
        if resample == 'down':
            self.conv_1 = spectral_norm(nn.Conv2d(input_dim, input_dim, filter_size, padding=filter_size//2))
            self.conv_2 = ConvMeanPoolSpecNorm(input_dim, output_dim, filter_size)
        elif resample == 'up':
            self.conv_1 = UpsampleConvSpecNorm(input_dim, output_dim, filter_size)
            self.conv_2 = spectral_norm(nn.Conv2d(output_dim, output_dim, filter_size, padding=filter_size//2))
        elif resample is None:
            self.conv_1 = spectral_norm(nn.Conv2d(input_dim, output_dim, filter_size, padding=filter_size//2))
            self.conv_2 = spectral_norm(nn.Conv2d(output_dim, output_dim, filter_size, padding=filter_size//2))
        else:
            raise Exception('invalid resample value')
    
    def forward(self, inputs):
        output = F.relu(inputs)
        output = self.conv_1(output)
        output = F.relu(output)
        output = self.conv_2(output)
        return output
    
    
class OptimizedConvBlockDisc1(nn.Module):
    def __init__(self):
        super(OptimizedConvBlockDisc1, self).__init__()
        self.conv_1 = nn.Conv2D(in_channels=3, out_channels=DIM_D, kernel_size=3)
        self.conv_2 = ConvMeanPool(input_dim=DIM_D, output_dim=DIM_D, filter_size=3)
        self.conv_shortcut = MeanPoolConv(input_dim=3, output_dim=DIM_D, filter_size=1, he_init=False, biases=True)

    def forward(self, inputs):
        shortcut = self.conv_shortcut(inputs)
        
        output = self.conv_1(inputs)
        output = F.relu(output)
        output = self.conv_2(output)
        
        return shortcut + output
    
    
class OptimizedConvBlockDisc1SpecNorm(nn.Module):
    def __init__(self):
        super(OptimizedConvBlockDisc1SpecNorm, self).__init__()
        self.conv_1 = spectral_norm(nn.Conv2d(in_channels=3, out_channels=DIM_D, kernel_size=3))
        self.conv_2 = ConvMeanPoolSpecNorm(input_dim=DIM_D, output_dim=DIM_D, filter_size=3)

    def forward(self, inputs):
        output = self.conv_1(inputs)
        output = F.relu(output)
        output = self.conv_2(output)
        return output
    
    
class Generator(nn.Module):
    def __init__(self, dim_g, output_dim):
        super(Generator, self).__init__()
        self.dim_g = dim_g
        self.output_dim = output_dim
        self.linear = nn.Linear(128, 4*4*dim_g)
        self.res_block1 = ResidualBlock(dim_g, dim_g, 3, resample='up')
        self.res_block2 = ResidualBlock(dim_g, dim_g, 3, resample='up')
        self.res_block3 = ResidualBlock(dim_g, dim_g, 3, resample='up')
        self.norm = Normalize(dim_g)
        self.conv = nn.Conv2d(dim_g, 3, 3, padding=1)
    
    def forward(self, n_samples, labels=None, noise=None):
        if noise is None:
            noise = torch.randn(n_samples, 128)
        output = self.linear(noise)
        output = output.view(-1, self.dim_g, 4, 4)

        output = self.res_block1(output, labels=labels)

        output = self.res_block2(output, labels=labels)

        output = self.res_block3(output, labels=labels)
        
        output = self.norm(output)
        output = F.relu(output)
        output = self.conv(output)
        output = torch.tanh(output)
        return output.view(-1, self.output_dim)
    

class Discriminator(nn.Module):
    def __init__(self, input_dim, dim_d):
        super(Discriminator, self).__init__()
        self.optimized_block = OptimizedConvBlockDisc1SpecNorm()
        self.conv_block_2 = ConvBlockSpecNorm(dim_d, dim_d, 3, resample='down')
        self.conv_block_3 = ConvBlockSpecNorm(dim_d, dim_d, 3, resample=None)
        self.conv_block_4 = ConvBlockSpecNorm(dim_d, dim_d, 3, resample=None)
        self.linear = spectral_norm(nn.Linear(dim_d, 1))

    def forward(self, inputs, labels=None):
        output = inputs.view(-1, 3, 32, 32)
        output = self.optimized_block(output)
        output = self.conv_block_2(output)
        output = self.conv_block_3(output)
        output = self.conv_block_4(output)
        
        output = F.relu(output)
        output = torch.mean(output, dim=[2, 3])
        output = self.linear(output)
        return output.view(-1), None


#for alpha>1
def f_alpha_star(y,alpha):
    return torch.math.pow(F.relu(y),alpha/(alpha-1.0))*torch.math.pow((alpha-1.0),alpha/(alpha-1.0))/alpha+1/(alpha*(alpha-1.0))

