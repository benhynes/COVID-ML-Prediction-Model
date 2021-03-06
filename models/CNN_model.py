from keras.layers import Input, Dense, Activation, Dropout, Flatten, Reshape, Conv2D, BatchNormalization, Concatenate
from keras.models import Model
from keras.optimizers import Adam, schedules
from keras.initializers import RandomNormal
import tensorflow as tf

import os

import numpy as np

def custom_loss(mask):
    def loss(y_true, y_pred):
        res = tf.square(tf.boolean_mask(y_true, mask = mask) - tf.boolean_mask(y_pred, mask = mask))
        res = tf.reduce_mean(res)
        return res
    return loss

def Inception_Block(input, filter_7x7 = 0, filter_5x5 = 0, filter_3x3 = 0, filter_1x1 = 0):
    hidden_7 = Conv2D(filter_7x7, (7,7), (1,1), padding = 'same') (input)
    hidden_5 = Conv2D(filter_5x5, (5,5), (1,1), padding = 'same') (input)
    hidden_3 = Conv2D(filter_3x3, (3,3), (1,1), padding = 'same') (input)
    hidden_1 = Conv2D(filter_1x1, (1,1), (1,1), padding = 'same') (input)
    hidden = Concatenate() ([hidden_7,hidden_5,hidden_3,hidden_1])
    hidden = BatchNormalization() (hidden)
    hidden = Activation('relu') (hidden)
    return hidden

class CNN_Model():
    def __get_model(self):
        x = Input(shape = self.input_shape)
        hidden = Inception_Block(x, 32, 64, 128, 32)
        hidden = Inception_Block(hidden, 16, 32, 64, 16)
        hidden = Inception_Block(hidden, 8, 16, 32, 8)
        hidden = Inception_Block(hidden, 4, 8, 16, 4)
        hidden = Conv2D(16, (1,1), (1,1), padding = 'same') (hidden)
        hidden = BatchNormalization() (hidden)
        hidden = Activation('relu') (hidden)
        out = Conv2D(1,(1,1),strides = (1,1), padding = 'same') (hidden)
        model = Model(x,out)
        initial_learning_rate = 0.02
        lr_schedule = schedules.ExponentialDecay(
            initial_learning_rate,
            decay_steps=100,
            decay_rate=0.96,
            staircase=True)
        model.compile(loss = 'mse',metrics = ['mae'], optimizer = Adam(learning_rate = lr_schedule))
        model.summary()
        return model

    

    def __init__(self, input_shape, output_shape, mask, lr = 2e-3):
        self.lr = lr

        self.mask = mask
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.model = self.__get_model()

    def predict(self, x):
        if (len(x.shape) == 3):
            x = np.reshape(x,(1,self.input_shape[0],self.input_shape[1],self.input_shape[2]))
        y = self.model.predict(x)
        if (y.shape[0] == 1):
            y = np.reshape(y,self.output_shape)
        return y

    def fit(self, x, y):
        self.model.fit(x,y)

    def train_on_batch(self, x_batch, y_batch):
        return self.model.train_on_batch(x_batch,y_batch)

    def save_weights(self, path = "trained_models/CNN.h5"):
        os.makedirs('trained_models', exist_ok=True)
        self.model.save_weights(path)
    
    def load_weights(self, path = "trained_models/CNN.h5"):
        self.model.load_weights(path)
    

class CNN_Data_Formatter():

    def __init__(self, input_shape, output_shape):
        self.input_shape = input_shape
        self.output_shape = output_shape
    
    def robust_normalize(self,x,x_median,q1,q3):
        return (x-x_median)/(q3-q1)

    def robust_denormalize(self,x,x_median,q1,q3):
        return x*(q3-q1)+x_median

    def CNN_reshape(self, x):
        ans = np.zeros((x.shape[1],x.shape[2],x.shape[0]))
        for time in range(len(x)):
            for lat in range(len(x[0])):
                for lon in range(len(x[0][0])):
                    ans[lat][lon][time] = x[time][lat][lon]
        return ans

    def Original_reshape(self, x):
        ans = np.zeros((x.shape[2],x.shape[0],x.shape[1]))
        for lat in range(len(x)):
            for lon in range(len(x[0])):
                for time in range(len(x[0][0])):
                    ans[time][lat][lon] = x[lat][lon][time]
        return ans

    def sampling(self, x_sample_set, y_sample_set):
        r = np.random.randint(len(x_sample_set))
        return x_sample_set[r], y_sample_set[r]

    def interval(self, dataset, coordinates, start_day, n_days):
        x_data = np.zeros((180,360,n_days))
        y_data = np.zeros((180,360))
        end_day = start_day + n_days
        for country in range(len(coordinates)):
            for date in range(start_day, end_day):
                x_data[coordinates[country][0],coordinates[country][1],date-start_day] += dataset[country][date]
            y_data[coordinates[country][0],coordinates[country][1]] += dataset[country][end_day]
        return x_data, y_data

    def get_sample_set(self, dataset, coordinates, n_days):
        x_sample_set = []
        y_sample_set = []
        for date in range(len(dataset[0])-n_days-1):
            x_data, y_data = self.interval(dataset, coordinates, date, n_days)
            x_sample_set.append(x_data)
            y_sample_set.append(y_data)
        return x_sample_set, y_sample_set

    def get_minibatch(self, x_sample_set, y_sample_set, batch_size):
        minibatch_x = []
        minibatch_y = []
        for _ in range(batch_size):
            x, y = self.sampling(x_sample_set, y_sample_set)
            minibatch_x.append(x)
            minibatch_y.append(y)
        return np.reshape(np.asarray(minibatch_x),(batch_size,self.input_shape[0],self.input_shape[1],self.input_shape[2])), np.reshape(np.asarray(minibatch_y),(batch_size, self.output_shape[0],self.output_shape[1],1))