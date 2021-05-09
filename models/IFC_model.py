from keras.layers import Input, Dense, BatchNormalization
from keras.models import Model
from keras.optimizers import Adam
import numpy as np
import os

class FC_Model():
    def __get_model(self):
        x = Input(shape = (self.input_shape))
        hidden = Dense(48, activation = 'relu') (x)
        hidden = Dense(48, activation = 'relu') (hidden)
        hidden = Dense(48, activation = 'relu') (hidden)
        out = Dense(self.output_shape)(hidden)
        model = Model(x,out)
        model.compile(loss = 'mse', metrics = ['mae'], optimizer = Adam(lr = self.lr))
        return model

    def __init__(self, input_shape, output_shape, lr = 0.0002):
        self.lr = lr
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.model = self.__get_model()
            
        
    def predict(self, x):
        if (len(x.shape) == 1):
            x = np.reshape(x,(1,self.input_shape))
        y = self.model.predict(x)
        if (y.shape[0] == 1):
            y = np.reshape(y,self.output_shape)
        return y

    def fit(self, x, y):
        self.model.fit(x,y)

    def train_on_batch(self, x_batch, y_batch):
        return self.model.train_on_batch(x_batch,y_batch)

    def save_weights(self, path = "trained_models/IFC.h5"):
        os.makedirs('trained_models',exist_ok=True)
        self.model.save_weights(path)
        
    def load_weights(self, path = "trained_models/IFC.h5"):
        self.model.load_weights(path)


class FC_Data_Formatter():

    def __init__(self, input_shape, output_shape):
        self.input_shape = input_shape
        self.output_shape = output_shape
        
    def robust_normalize(self,x,x_median,q1,q3):
        return (x-x_median)/(q3-q1)

    def robust_denormalize(self,x,x_median,q1,q3):
        return x*(q3-q1)+x_median

    def lat_normalize(self, x):
        return x/180

    def long_normalize(self, x):
        return x/360

    def sampling(self, x, coordinates):
        n_days = self.input_shape - 2
        r1 = np.random.randint(len(x))
        r2 = np.random.randint(len(x[0])-n_days-1)
        x_sample = [self.lat_normalize(coordinates[r1][0]),self.long_normalize(coordinates[r1][1])]
        for date in range(n_days):
            x_sample.append(x[r1][r2+date])
        return np.array(x_sample), x[r1][r2+n_days]

    def get_minibatch(self, dataset, coordinates, batch_size):
        minibatch_x = []
        minibatch_y = []
        for _ in range(batch_size):
            x, y = self.sampling(dataset, coordinates)
            minibatch_x.append(x)
            minibatch_y.append(y)
        return np.reshape(np.asarray(minibatch_x),(batch_size,self.input_shape)), np.reshape(np.asarray(minibatch_y),(batch_size, self.output_shape))