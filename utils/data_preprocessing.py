import csv
import numpy as np

def read_csv(file_path):
    raw_dataset = np.genfromtxt(file_path,delimiter = ',',names = True,usecols = np.arange(4,400),dtype = int)
    raw_dataset = np.array(raw_dataset.tolist())
    dataset = raw_dataset.copy()
    for i in range(len(raw_dataset)):
        for j in range(1,len(raw_dataset[0])):
            dataset[i][j] = raw_dataset[i][j] - raw_dataset[i][j-1]
    return dataset

def data_preprocess(raw_dataset, ratio = [0.5,0.2,0.3]):
    dataset = raw_dataset
    x_train,x_valid,y_train,y_valid = train_test_split(raw_dataset,raw_dataset, test_size = ratio[1]+ratio[2],shuffle = False)

    return x_train,x_valid, y_train,y_valid

def sampling(dataset, input_size, output_size):
    r = np.random.randint(len(dataset)-input_size-output_size+1)
    input_start = r
    input_end = input_start + input_size
    x = dataset[input_start:input_end]
    y = dataset[input_end:input_end+output_size]
    return np.reshape(x,(1,input_size)), np.reshape(y,(1,output_size))
