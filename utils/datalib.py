import csv
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
import math

def read_csv(file_path):
    raw_dataset = pd.read_csv(file_path)
    return raw_dataset.to_numpy()

def export_csv(data, file_path):
    for row in range(len(data)):
        for col in range(len(data[0])):
            print(data[row][col],end = ',')

def preprocess(raw_dataset):
    dataset = []
    for country in range(len(raw_dataset)):
        if raw_dataset[country][1]=='Canada' or (not(math.isnan(raw_dataset[country][2])) and raw_dataset[country][2] != 0 and raw_dataset[country][3] != 0):
            temp = [max(0,raw_dataset[country][4])]
            for date in range(5,len(raw_dataset[country])):
                temp.append(max(0,raw_dataset[country][date] - raw_dataset[country][date-1]))
            if  country>=1 and raw_dataset[country][1] == 'Canada' and raw_dataset[country-1][1]=='Canada':
                dataset[-1] = dataset[-1] + temp
            else:
                dataset.append(np.asarray(temp))

    return np.asarray(dataset)

def moving_average(dataset, window):
    ans = []
    sum = 0
    for time in range(window-1):
        ans.append(dataset[time])
        sum += dataset[time]
    for time in range(window-1, len(dataset)):
        if time >= window:
            sum -= dataset[time-window]
        sum += dataset[time]
        ans.append(sum/window)
    return np.array(ans)

def extract_coordinates(raw_dataset):
    coordinates = []
    for country in range(len(raw_dataset)):
        if (not(math.isnan(raw_dataset[country][2])) and raw_dataset[country][2] != 0 and raw_dataset[country][3] !=0):
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

def get_mask(batch_size, coordinates):
    mask = np.zeros((180,360))
    batch_mask = []
    for country in range(len(coordinates)):
        mask[coordinates[country][0]][coordinates[country][1]] = 1
    for _ in range(batch_size):
        batch_mask.append(mask)
    return np.array(batch_mask)

def get_country_time_series(coordinate, country, loc_map):
    res = np.zeros((len(loc_map)))
    for date in range(len(loc_map)):
        res[date] = loc_map[date][coordinate[country][0]][coordinate[country][1]]
    return res

def split_data(dataset, ratio = [0.5,0.3,0.2]):

    x_train,x_valid = train_test_split(dataset.transpose(),train_size = ratio[0],shuffle = False)
    x_train = x_train.transpose()
    x_valid = x_valid.transpose()
    return x_train,x_valid

def robust_normalize(x, x_median, q1, q3):
    return (x-x_median)/(q3-q1)

def robust_denormalize(x, x_median, q1, q3):
    return x*(q3-q1)+x_median