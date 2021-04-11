import csv
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
import math

def read_csv(file_path):
    raw_dataset = pd.read_csv(file_path)
    return raw_dataset.to_numpy()

def preprocess(raw_dataset):
    dataset = []
    for i in range(len(raw_dataset)):
        if raw_dataset[i][1]=='Canada' or (raw_dataset[i][2] != 0 and raw_dataset[i][3]!=0):
            temp = [raw_dataset[i][4]]
            for j in range(5,len(raw_dataset[i])):
                temp.append(raw_dataset[i][j] - raw_dataset[i][j-1])
            if  i>=1 and raw_dataset[i][1] == 'Canada' and raw_dataset[i-1][1]=='Canada':
                dataset[-1] = dataset[-1] + temp
            else:
                dataset.append(np.asarray(temp))

    return np.asarray(dataset)

def extract_coordinates(raw_dataset):
    coordinates = []
    for i in range(len(raw_dataset)):
        if (raw_dataset[i][2] != 0 and raw_dataset[i][3] !=0):
            coordinates.append(np.array([raw_dataset[i][2]+90,raw_dataset[i][3]+180],dtype = int))
    return coordinates

def get_active_dataset(confirmed, deaths, recoveries):
    return confirmed - deaths - recoveries

def get_loc_map(coordinates, dataset):
    datamap = np.zeros((len(dataset[0]),180,360))
    for i in range(len(dataset)):
        for j in range(len(dataset[0])):
            datamap[j][coordinates[i][0]][coordinates[i][1]] += dataset[i][j]
    return datamap

def get_country_time_series(coordinate, loc_map):
    res = np.zeros((len(loc_map)))
    for i in range(len(loc_map)):
        res[i] = loc_map[i][coordinate[0]][coordinate[1]]
    return res

def split_data(datamap, dataset, ratio = [0.6,0.2,0.2]):

    x_train,x_valid, y_train, y_valid = train_test_split(datamap, dataset,train_size = ratio[0],shuffle = False)
    return x_train,x_valid, y_train, y_valid

def normalize(x,x_mean,x_std):
    return (x-x_mean)*7/x_std

def denormalize(x,x_mean,x_std):
    return (x*x_std/7)+x_mean
