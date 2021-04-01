from keras.layers import LSTM, Input, Dense, Activation, Dropout
from keras.models import Model
from keras.optimizers import Adam

class LSTM_Model():
    def __init__(self, name, input_shape, output_shape, lr):
        self.name = name
        self.lr = lr

        x = Input(shape = (input_shape,1))
        hidden = LSTM(units = 128, return_sequences = True) (x)
        hidden = Dropout(0.2) (hidden)
        hidden = LSTM(units = 128) (hidden)
        hidden = Dropout(0.2) (hidden)
        hidden = Dense(128) (hidden)
        hidden = Dropout(0.2) (hidden)
        out = Dense(output_shape, activation = 'tanh')(hidden)
        self.model = Model(x,out)
        self.model.compile(loss = 'mae', metrics = ['mean_absolute_percentage_error'], optimizer = Adam(lr = self.lr))
        self.model.summary()
    
    def predict(self, x):
        return self.model.predict(x)

    def fit(self, x, y):
        self.model.fit(x,y)

    def train_on_batch(self, x_batch, y_batch):
        return self.model.train_on_batch(x_batch,y_batch)

    def save_weights(self, path):
        self.model.save_weights(path)
    
    def load_weights(self, path):
        self.model.load_weights(path)
    