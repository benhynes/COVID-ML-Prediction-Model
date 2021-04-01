import csv
import numpy as np
from sklearn.model_selection import train_test_split

def read_csv(file_path):
    raw_dataset = np.genfromtxt(file_path,delimiter = ',',names = True,usecols = np.arange(4,400),dtype = int)
    raw_dataset = np.array(raw_dataset.tolist())
    
    return raw_dataset

def preprocess(raw_dataset):
    dataset = np.zeros((len(raw_dataset),len(raw_dataset[0])))
    for i in range(len(raw_dataset)):
        dataset[i][0] = raw_dataset[i][0]
        for j in range(1,len(raw_dataset[0])):
            dataset[i][j] = raw_dataset[i][j] - raw_dataset[i][j-1]
    return np.asarray(dataset[0])

def split_data(dataset, ratio = [0.6,0.2,0.2]):

    x_train,x_valid = train_test_split(dataset,shuffle = False)
    return x_train, x_valid

def sampling(dataset, number_of_input_days, number_of_output_days):
    r = np.random.randint(len(dataset)-number_of_input_days-number_of_output_days+1)
    input_start = r
    input_end = input_start + number_of_input_days
    x = []
    y = []
    x = dataset[input_start:input_end]
    y = dataset[input_end:input_end+number_of_output_days]

    return np.reshape(x,(1,number_of_input_days)), np.reshape(y,(1,number_of_output_days))

def normialize(x, x_maximum):
    return (x/x_maximum)*2-1 + 0.5

def denormialize(x, x_maximum):
    return ((x+1-0.5)/2)*x_maximum

def get_minibatch(dataset, batch_size, input_shape, output_shape):
    minibatch_x = []
    minibatch_y = []
    for _ in range(batch_size):
        x, y = sampling(dataset,input_shape,output_shape)
        minibatch_x.append(x)
        minibatch_y.append(y)
    return np.reshape(np.asarray(minibatch_x),(batch_size,input_shape)), np.reshape(np.asarray(minibatch_y),(batch_size,output_shape))