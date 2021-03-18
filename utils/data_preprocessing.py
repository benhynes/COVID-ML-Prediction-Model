import numpy as np
import csv

def read_csv(file_path):
    raw_dataset = np.genfromtxt(file_path,delimiter = ',',names = True,usecols = np.arange(0,5),dtype = int)
    return raw_dataset



def data_preprocess(raw_dataset, ratio = [0.5,0.2,0.3]):
    rtrain = raw_dataset['b']
    return rtrain