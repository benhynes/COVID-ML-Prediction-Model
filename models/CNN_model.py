from keras.layers import Input, Dense, Activation, Dropout, Flatten, Reshape, Conv2D
from keras.models import Model
from keras.optimizers import Adam
from keras.initializers import RandomNormal

import os

class CNN_Model():
    def __init__(self, name, input_shape, output_shape, lr = 0.0002):
        self.name = name
        self.lr = lr

        x = Input(shape = input_shape)
        hidden = Conv2D(32,(3,3),strides = (1,1),padding = 'same') (x)
        #hidden = Dropout(0.2) (hidden)
        hidden = Conv2D(32,(3,3),strides = (1,1),padding = 'same') (hidden)
        #hidden = Dropout(0.2) (hidden)
        hidden = Conv2D(32,(3,3),strides = (2,2),padding = 'same') (hidden)
        out = Dense(output_shape, activation = 'tanh', kernel_initializer=RandomNormal(mean=0., stddev=0.1))(hidden)
        self.model = Model(x,out)
        self.model.compile(loss = 'mae', metrics = ['mean_absolute_percentage_error'], optimizer = Adam(lr = self.lr))
    
    def predict(self, x):
        return self.model.predict(x)

    def fit(self, x, y):
        self.model.fit(x,y)

    def train_on_batch(self, x_batch, y_batch):
        return self.model.train_on_batch(x_batch,y_batch)

    def save_weights(self, file_name = "FC_Model.h5"):
        self.model.save_weights(file_name)
    
    def load_weights(self, file_name = "FC_Model.h5"):
        self.model.load_weights(file_name)
    