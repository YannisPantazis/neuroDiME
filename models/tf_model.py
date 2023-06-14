from tensorflow_addons.layers import SpectralNormalization
from keras import backend as K  
from keras.layers import Dense, Input, Activation
from keras.models import Sequential, Model


class Discriminator(Model):
    """Discriminator Class which is responsible of initializing the discriminator"""

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

        print()
        print('Discriminator Summary:')
        self.discriminator.summary()
   
    def call(self, inputs):
        predicted = self.discriminator(inputs)
        return predicted

    def bounded_activation(x):
        M = 100.0
        return M * K.tanh(x/M)


class Generator(Model):
    """Generator Class which is responsible of initializing the generator"""

    def __init__(self, X_dim, Z_dim, spec_norm, layers_list):
        super(Generator, self).__init__()

        self.X_dim = X_dim
        self.Z_dim = Z_dim
        self.spec_norm = spec_norm
        self.layers_list = layers_list

        self.generator = Sequential()
        self.generator.add(Input(shape=(self.Z_dim,)))

        if self.spec_norm:
            for h_dim in self.layers_list:
                self.generator.add(SpectralNormalization(Dense(units=h_dim, activation='relu')))
            self.generator.add(SpectralNormalization(Dense(units=X_dim, activation='linear')))
        else:
            for h_dim in self.layers_list:
                self.generator.add(Dense(units=h_dim, activation='relu'))
            self.generator.add(Dense(units=X_dim, activation='linear'))
        
        print()
        print('Generator Summary:')
        self.generator.summary()
   
    def call(self, inputs):
        predicted = self.generator(inputs)
        return predicted
