import torch.nn as nn
from torch.nn.utils import spectral_norm
from collections import OrderedDict as OrderedDict
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import os
from torchvision.models import inception_v3
from scipy.linalg import sqrtm
from torch.nn.functional import adaptive_avg_pool2d
from tqdm import tqdm
from torchvision import datasets, transforms
from PIL import Image

DIM_G = 128 # Generator dimensionality
DIM_D = 128 # Critic dimensionality
NORMALIZATION_G = True # Use batchnorm in generator?
NORMALIZATION_D = False # Use batchnorm (or layernorm) in critic? This doesn't do anything at the moment.
CONDITIONAL = False # Whether to train a conditional or unconditional model
ACGAN = False # If CONDITIONAL, whether to use ACGAN or "vanilla" conditioning
ACGAN_SCALE = 1. # How to scale the critic's ACGAN loss relative to WGAN loss
ACGAN_SCALE_G = 0.1 # How to scale generator's ACGAN loss relative to WGAN loss
device = 'cuda' if torch.cuda.is_available() else 'cpu'

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
    def __init__(self, input_dim):
        super(OptimizedConvBlockDisc1, self).__init__()
        self.conv_1 = nn.Conv2D(in_channels=input_dim, out_channels=DIM_D, kernel_size=3)
        self.conv_2 = ConvMeanPool(input_dim=DIM_D, output_dim=DIM_D, filter_size=3)
        self.conv_shortcut = MeanPoolConv(input_dim=input_dim, output_dim=DIM_D, filter_size=1, he_init=False, biases=True)

    def forward(self, inputs):
        shortcut = self.conv_shortcut(inputs)
        
        output = self.conv_1(inputs)
        output = F.relu(output)
        output = self.conv_2(output)
        
        return shortcut + output
    
    
class OptimizedConvBlockDisc1SpecNorm(nn.Module):
    def __init__(self, input_dim):
        super(OptimizedConvBlockDisc1SpecNorm, self).__init__()
        self.conv_1 = spectral_norm(nn.Conv2d(in_channels=input_dim, out_channels=DIM_D, kernel_size=3))
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
        self.conv = nn.Conv2d(dim_g, 1 if output_dim == 784 else 3, 3, padding=1)
        if output_dim == 784:
            self.adjust_conv = nn.Conv2d(1, 1, 3, padding=1)  # Additional conv layer to adjust size for MNIST

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
        
        if self.output_dim == 1 * 28 * 28:
            output = F.interpolate(output, size=(28, 28), mode='bilinear', align_corners=False)
            return output.view(-1, 1, 28, 28)  # MNIST
        else:
            return output.view(-1, 3, 32, 32)  # CIFAR-10
    

class Discriminator(nn.Module):
    def __init__(self, input_dim, dim_d):
        super(Discriminator, self).__init__()
        self.optimized_block = OptimizedConvBlockDisc1SpecNorm(input_dim)
        self.conv_block_2 = ConvBlockSpecNorm(dim_d, dim_d, 3, resample='down')
        self.conv_block_3 = ConvBlockSpecNorm(dim_d, dim_d, 3, resample=None)
        self.conv_block_4 = ConvBlockSpecNorm(dim_d, dim_d, 3, resample=None)
        self.linear = spectral_norm(nn.Linear(dim_d, 1))

        
    def forward(self, inputs, labels=None):
        if inputs.shape[1] == 1:
            output = inputs.view(-1, 1, 28, 28)
        else:
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

def calculate_fid(real_images, generated_images, batch_size=50):
    model = inception_v3(pretrained=True, transform_input=False).to(device)
    model.eval()

    def get_activations(images):
        n_batches = images.size(0) // batch_size
        pred_arr = np.empty((images.size(0), 2048))

        for i in range(n_batches):
            batch = images[i * batch_size: (i + 1) * batch_size].to(device)
            print(batch.shape)
            pred = model(batch)[0]
            pred = adaptive_avg_pool2d(pred, (1, 1)).squeeze().cpu().data.numpy()
            pred_arr[i * batch_size: (i + 1) * batch_size] = pred

        return pred_arr

    real_activations = get_activations(real_images)
    generated_activations = get_activations(generated_images)

    mu1, sigma1 = real_activations.mean(axis=0), np.cov(real_activations, rowvar=False)
    mu2, sigma2 = generated_activations.mean(axis=0), np.cov(generated_activations, rowvar=False)

    ssdiff = np.sum((mu1 - mu2) ** 2.0)
    covmean = sqrtm(sigma1.dot(sigma2))

    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid


def transform_image(image):
    # Check if the image is already a tensor
    if isinstance(image, torch.Tensor):
        return image  # If it is, return it as is
    elif isinstance(image, np.ndarray):
        image = Image.fromarray(image)  # Convert ndarray to PIL Image
    elif not isinstance(image, Image.Image):
        raise TypeError(f"Expected image to be of type PIL Image or ndarray, but got {type(image)}")

    transform = transforms.Compose([
            transforms.Resize((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize the image
            ])

    # Apply the necessary transformations
    return transform(image)

class GAN_CIFAR10():
    '''
    Class for training a GAN for CIFAR-10 using one of the provided divergences
    If reverse_order=False the GAN works to minimize min_theta D(P||g_theta(Z)) where P is the distribution to be leared, Z is the noise source and g_theta is the generator (with parameters theta).
    If reverse_order=True the GAN works to minimize min_theta D(g_theta(Z)||P) where P is the distribution to be leared, Z is the noise source and g_theta is the generator (with parameters theta).
    '''
    # initialize
    def __init__(self, divergence, generator, gen_optimizer, noise_source, epochs, disc_steps_per_gen_step, method, dataset, batch_size=None, reverse_order=False, include_penalty_in_gen_loss=False, conditional=False):
        self.divergence = divergence # Variational divergence
        self.generator = generator
        self.epochs = epochs
        self.disc_steps_per_gen_step = disc_steps_per_gen_step
        self.gen_optimizer = gen_optimizer
        self.reverse_order = reverse_order
        self.include_penalty_in_gen_loss = include_penalty_in_gen_loss
        self.noise_source = noise_source
        self.conditional = conditional
        self.method = method
        self.dataset = dataset
        
        if batch_size is None:
            self.batch_size = self.divergence.batch_size
        else:
            self.batch_size = batch_size
        
    def estimate_loss(self, x, z, labels):
        ''' Estimating the loss '''
        # z = torch.from_numpy(z).float()
        if self.reverse_order:
            data1 = self.generator(self.batch_size, labels, z)
            data2 = x
        else:
            data1 = x
            data2 = self.generator(self.batch_size, labels, z)

        return self.divergence.estimate(data1, data2)
    
    def gen_train_step(self, x, z, labels):
        ''' generator's parameters update '''
        self.gen_optimizer.zero_grad()
        # x.requires_grad_(True)
        # z.requires_grad_(True)

        # z = torch.from_numpy(z).float()
        if self.reverse_order:
            data1 = self.generator(self.batch_size, labels, z)
            data2 = x
        else:
            data1 = x
            data2 = self.generator(self.batch_size, labels, z)

        loss = self.divergence.discriminator_loss(data1, data2, labels)
        if self.include_penalty_in_gen_loss and self.divergence.discriminator_penalty is not None:
            loss = loss - self.divergence.discriminator_penalty.evaluate(self.divergence.discriminator, data1, data2, labels)
        
        loss.backward()
        self.gen_optimizer.step()

        return loss.item()

    def disc_train_step(self, x, z, labels):
        ''' discriminator's parameters update '''
        # z = torch.from_numpy(z).float()
        if self.reverse_order:
            data1 = self.generator(self.batch_size, labels, z)
            data2 = x
        else:
            data1 = x
            data2 = self.generator(self.batch_size, labels, z)
        
        loss = self.divergence.train_step(data1, data2, labels)

        if self.divergence.discriminator_penalty is not None:
            for p in self.divergence.discriminator.parameters():
                p.data.clamp_(-0.01, 0.01)

        return loss.item()


    def train(self, dataloader, save_frequency=None, num_gen_samples_to_save=None, save_loss_estimates=False):
        ''' training function of our GAN '''
        generator_samples = []
        loss_estimates = []
        fids = []
        gen_losses = np.zeros(self.epochs)
        disc_losses = np.zeros(self.epochs)

        for epoch in tqdm(range(self.epochs), desc='Epochs'):
            disc_loss = 0
            gen_loss = 0

            for data, labels in dataloader:
                noise_batch = self.noise_source(self.batch_size)
                data = data.to(device)
                labels = labels.to(device)
                
                if not self.conditional:
                    labels = None
                    
                for disc_step in range(self.disc_steps_per_gen_step):
                    disc_cost = self.disc_train_step(data, noise_batch, labels)
                    disc_loss += disc_cost

                gen_cost = self.gen_train_step(data, noise_batch, labels)
                gen_loss += gen_cost
            
            gen_losses[epoch] = gen_loss / len(dataloader)
            disc_losses[epoch] = disc_loss / len(dataloader)
           
            # Generate images
            generated_images = []
            for _ in range(10000 // self.batch_size):
                noise = self.noise_source(self.batch_size).to(device)
                with torch.no_grad():
                    generated = self.generator(self.batch_size, labels=None, noise=noise).detach().cpu()
                    generated = torch.stack([transform_image(img) for img in generated])
                    print(generated.shape)
                    generated_images.append(generated)
            generated_images = torch.cat(generated_images, 0)
            
            # Load real images
            real_images = []
            for _, (images, _) in zip(range(10000 // self.batch_size), dataloader):
                images = images.to(device)
                images = torch.stack([transform_image(img) for img in images])
                real_images.append(images)
            real_images = torch.cat(real_images, 0)

            fid = calculate_fid(real_images, generated_images)
            fids.append(fid)

            if save_frequency is not None and (epoch+1) % save_frequency == 0:
                print('Epoch:', epoch+1, 'Gen Loss:', gen_losses[epoch], 'Disc Loss:', disc_losses[epoch], 'FID:', fids[epoch])
                
                self.generate_image(self.generator, epoch+1, show=False)
                
                # if save_loss_estimates:
                #     loss_estimates.append(float(self.estimate_loss(data, noise_batch)))

        return generator_samples, loss_estimates, gen_losses, disc_losses
    
    def generate_image(self, generator, frame, n_samples=12, show=True, labels=None):
        
        if self.divergence.discriminator_penalty:
            if not os.path.exists(f'samples_{self.method}_GP_{self.dataset}/'):
                os.makedirs(f'samples_{self.method}_GP_{self.dataset}/')
        else:
            if not os.path.exists(f'samples_{self.method}_{self.dataset}/'):
                os.makedirs(f'samples_{self.method}_{self.dataset}/')

        """Generate a batch of images and save them to a grid."""
        generator.eval()
        fixed_noise = torch.randn(n_samples, DIM_G, device=device)
        fixed_labels = torch.tensor(np.array([0,1,2,3,4,5,6,7,8,9]*10), dtype=torch.long, device=device)
        
        # fixed_labels = torch.tensor(np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9] * n_samples), dtype=torch.long, device=device)
        
        with torch.no_grad():
            samples = generator(n_samples, labels=fixed_labels, noise=fixed_noise).detach().cpu()
            samples = ((samples + 1) * 127.5).clamp(0, 255).to(torch.uint8)
            if samples.shape[1] == 1:
                samples = samples.view(n_samples, 1, 28, 28)
            else:
                samples = samples.view(n_samples, 3, 32, 32)
            samples = samples.permute(0, 2, 3, 1)

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
            
            if self.divergence.discriminator_penalty:
                plt.savefig(f'samples_{self.method}_GP_{self.dataset}/gen_images_{frame}.png')
            else:
                plt.savefig(f'samples_{self.method}_{self.dataset}/gen_images_{frame}.png')
            
            if show:
                plt.show()
            plt.close(fig)

        generator.train()
        
    def save(self, path):
        torch.save({
            'generator_state_dict': self.generator.state_dict(),
            'gen_optimizer_state_dict': self.gen_optimizer.state_dict(),
            'discriminator_state_dict': self.divergence.discriminator.state_dict(),
            'discriminator_optimizer_state_dict': self.divergence.disc_optimizer.state_dict(),
        }, path)
        
    def load(self, path):
        checkpoint = torch.load(path)
        self.generator.load_state_dict(checkpoint['generator_state_dict'])
        self.gen_optimizer.load_state_dict(checkpoint['gen_optimizer_state_dict'])
        self.divergence.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        self.divergence.disc_optimizer.load_state_dict(checkpoint['discriminator_optimizer_state_dict'])
        