from tensorflow_addons.layers import SpectralNormalization
from Divergences import *
from keras import backend as K  
from keras.layers import Dense, Input, Activation
from keras.models import Sequential, Model
from Divergences import *


class Discriminator(Model):
    def __init__(self, input_dim, spec_norm, bounded, layers_list):
        super(Discriminator, self).__init__()

        self.input_dim = input_dim
        self.spec_norm = spec_norm
        self.bounded = bounded
        self.layers_list = layers_list

        self.discriminator = Sequential()
        self.discriminator.add(Input(shape=(self.input_dim,)))

        if self.spec_norm:
            for h_dim in self.layers_list:
                self.discriminator.add(SpectralNormalization(Dense(units=h_dim, activation='relu')))
            self.discriminator.add(SpectralNormalization(Dense(units=1, activation='linear')))
        else:
            for h_dim in self.layers_list:
                self.discriminator.add(Dense(units=h_dim, activation='relu'))
            self.discriminator.add(Dense(units=1, activation='linear'))
        

        if bounded:
            self.discriminator.add(Activation(self.bounded_activation))

        self.discriminator.summary()
   
    def call(self, inputs):
        predicted = self.discriminator(inputs)
        return predicted

    def bounded_activation(x):
        M = 100.0
        return M * K.tanh(x/M)
