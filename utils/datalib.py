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
    for country in range(len(raw_dataset)):
        if raw_dataset[country][1]=='Canada' or (raw_dataset[country][2] != 0 and raw_dataset[country][3] != 0):
            temp = [raw_dataset[country][4]]
            for date in range(5,len(raw_dataset[country])):
                temp.append(raw_dataset[country][date] - raw_dataset[country][date-1])
            if  country>=1 and raw_dataset[country][1] == 'Canada' and raw_dataset[country-1][1]=='Canada':
                dataset[-1] = dataset[-1] + temp
            else:
                dataset.append(np.asarray(temp))

    return np.asarray(dataset)

def extract_coordinates(raw_dataset):
    coordinates = []
    for country in range(len(raw_dataset)):
        if (raw_dataset[country][2] != 0 and raw_dataset[country][3] !=0):
            coordinates.append(np.array([raw_dataset[country][2]+90,raw_dataset[country][3]+180],dtype = int))
    return coordinates

def get_active_dataset(confirmed, deaths, recoveries):
    return confirmed - deaths - recoveries

def get_loc_map(coordinates, dataset):
    datamap = np.zeros((len(dataset[0]),180,360))
    for country in range(len(dataset)):
        for date in range(len(dataset[0])):
            datamap[date][coordinates[country][0]][coordinates[country][1]] += dataset[country][date]
    return datamap

def get_country_time_series(coordinate, country, loc_map):
    res = np.zeros((len(loc_map)))
    for date in range(len(loc_map)):
        res[date] = loc_map[date][coordinate[country][0]][coordinate[country][1]]
    return res

def split_data(datamap, dataset, ratio = [0.6,0.2,0.2]):

    x_train,x_valid, y_train, y_valid = train_test_split(datamap, dataset,train_size = ratio[0],shuffle = False)
    return x_train,x_valid, y_train, y_valid

def normalize(x,x_mean,x_std):
    return (x-x_mean)*7/x_std

def denormalize(x,x_mean,x_std):
    return (x*x_std/7)+x_mean
