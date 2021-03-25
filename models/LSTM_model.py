from keras.layers import LSTM, Sequential


class LSTM_Model():
    def __init__(name, input_shape, output_shape):
        self.name = name
        self.model = self.get_LSTM_Model(input_shape,output_shape)
    
    def get_LSTM_Model(self, input_shape, output_shape):
        x = Input(shape = input_shape)
        hidden = LSTM(20) (x)
        hidden = Activation('sigmoid') (hidden)
        out = LSTM(20) (hidden)
        m = Model(x,out)
        m.compile(loss = 'mse', optimizer = adam(lr = 0.0002))
        return m
    
    def predict(self, x)
        return self.model.predict(x)

    def fit(self, x, y)
        self.model.fit(x,y)
    