from keras.layers import Input, Dense, Activation, Dropout, Flatten, Reshape
from keras.models import Model
from keras.optimizers import Adam
from keras.initializers import RandomNormal

import os

class FC_Model():
    def __init__(self, name, input_shape, output_shape, lr = 0.0002):
        self.name = name
        self.lr = lr

        x = Input(shape = input_shape)
        hidden = Dense(10, activation = 'tanh', kernel_initializer=RandomNormal(mean=0., stddev=0.1)) (x)
        #hidden = Dropout(0.2) (hidden)
        hidden = Dense(10, activation = 'tanh', kernel_initializer=RandomNormal(mean=0., stddev=0.1)) (hidden)
        #hidden = Dropout(0.2) (hidden)
        out = Dense(output_shape, activation = 'tanh', kernel_initializer=RandomNormal(mean=0., stddev=0.1))(hidden)
        self.model = Model(x,out)
        self.model.compile(loss = 'mse', metrics = ['mae'], optimizer = Adam(lr = self.lr))
    
    def predict(self, x):
        return self.model.predict(x)

    def fit(self, x, y):
        self.model.fit(x,y)

    def train_on_batch(self, x_batch, y_batch):
        return self.model.train_on_batch(x_batch,y_batch)

    def save_weights(self, path = "trained_models/FC.h5"):
        self.model.save_weights(file_name)
    
    def load_weights(self, path = "trained_models/FC.h5"):
        self.model.load_weights(file_name)
    