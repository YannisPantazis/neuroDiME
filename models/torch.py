from torch.nn import Module, Linear, ReLU, Sequential
from torch.nn.utils import spectral_norm
from Divergences_torch import *
from torchsummary import summary
from collections import OrderedDict as OrderedDict
from Divergences_torch import *


class BoundedActivation(Module):
    def __init__(self):
        super().__init__()

    def bounded_activation(x):
        M = 100.0
        return M * torch.tanh(x/M)

    def forward(self, input):
        return self.bounded_activation(input)
    
    
class Discriminator(Module):
    def __init__(self, input_dim, batch_size, spec_norm, bounded, layers_list):
        super(Discriminator, self).__init__()

        self.input_dim = input_dim
        self.spec_norm = spec_norm
        self.layers_list = layers_list
        self.bounded = bounded
        self.batch_size = batch_size
        model_dict = OrderedDict()

        if self.spec_norm:
            for i, h_dim in enumerate(self.layers_list):
                model_dict[f'Dense{i}'] = spectral_norm(Linear(input_dim, h_dim))
                model_dict[f'ReLU{i}'] = ReLU()
                input_dim = h_dim
            
            model_dict[f'Dense{i+1}'] = Linear(input_dim, 1)
        else:
            for i, h_dim in enumerate(self.layers_list):
                model_dict[f'Dense{i}'] = Linear(input_dim, h_dim)
                model_dict[f'ReLU{i}'] = ReLU()
                input_dim = h_dim

            model_dict[f'Dense{i+1}'] = Linear(input_dim, 1)

        if bounded:
            model_dict[f'Bounded_Activation'] = BoundedActivation()

        self.discriminator = Sequential(model_dict)
        summary(self.discriminator, (self.batch_size, self.input_dim))

    def forward(self, inputs):
        predicted = self.discriminator(inputs)
        return predicted
