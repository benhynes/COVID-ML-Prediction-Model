from keras.layers import Input, Dense, Activation, Dropout, Flatten, Reshape, Conv2D
from keras.models import Model
from keras.optimizers import Adam
from keras.initializers import RandomNormal

import os

import numpy as np

class CNN_Model():
    def __get_model(self):
        x = Input(shape = self.input_shape)
        hidden = Conv2D(30,(1,1),strides = (1,1),padding = 'same') (x)
        hidden = Dropout(0.2) (hidden)
        hidden = Conv2D(16,(3,3),strides = (1,1),padding = 'same') (hidden)
        hidden = Dropout(0.2) (hidden)
        hidden = Conv2D(16,(3,3),strides = (2,2),padding = 'same') (hidden)
        hidden = Dropout(0.2) (hidden)
        hidden = Conv2D(8,(3,3),strides = (2,2),padding = 'same') (hidden)
        hidden = Dropout(0.2) (hidden)
        hidden = Conv2D(4,(3,3),strides = (2,2),padding = 'same') (hidden)
        hidden = Dropout(0.2) (hidden)
        hidden = Flatten() (hidden)
        hidden = Dense(4096) (hidden)
        hidden = Dropout(0.2) (hidden)
        hidden = Dense(2048) (hidden)
        hidden = Dropout(0.2) (hidden)
        hidden = Dense(2048) (hidden)
        hidden = Dropout(0.2) (hidden)
        out = Dense(self.output_shape) (hidden)
        model = Model(x,out)
        model.compile(loss = 'mse', metrics = ['mae'], optimizer = Adam(lr = self.lr))
        model.summary()
        return model

    def __init__(self, name, input_shape, output_shape, lr = 0.00002):
        self.name = name
        self.lr = lr

        self.input_shape = input_shape
        self.output_shape = output_shape
        self.model = self.__get_model()

    def predict(self, x):
        return self.model.predict(x)

    def fit(self, x, y):
        self.model.fit(x,y)

    def train_on_batch(self, x_batch, y_batch):
        return self.model.train_on_batch(x_batch,y_batch)

    def save_weights(self, path = "trained_models/CNN.h5"):
        self.model.save_weights(path)
    
    def load_weights(self, path = "trained_models/CNN.h5"):
        self.model.load_weights(path)
    

class CNN_Data_Formatter():

    def __init__(self, input_shape, output_shape):
        self.input_shape = input_shape
        self.output_shape = output_shape
    
    def sampling(self, x_data, y_data):
        number_of_input_days = self.input_shape[2]

        r = np.random.randint(len(x_data)-number_of_input_days)
        input_start = r
        input_end = input_start + number_of_input_days

        x = x_data[input_start:input_end]
        y = y_data[input_end:input_end+1]

        return np.reshape(x,(self.input_shape[0],self.input_shape[1],self.input_shape[2])), np.reshape(y,(self.output_shape))

    def get_minibatch(self, x_data, y_data, batch_size):
        minibatch_x = []
        minibatch_y = []
        for _ in range(batch_size):
            x, y = self.sampling(x_data, y_data)
            minibatch_x.append(x)
            minibatch_y.append(y)
        return np.reshape(np.asarray(minibatch_x),(batch_size,self.input_shape[0],self.input_shape[1],self.input_shape[2])), np.reshape(np.asarray(minibatch_y),(batch_size, self.output_shape))