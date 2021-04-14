
import argparse
from utils.datalib import *
from utils.plotting import *
from models.FC_model import FC_Model
from sklearn import preprocessing

import matplotlib.pyplot as plt 

def main(args):
    
    return 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', dest = 'file_path', default = "dataset/time_series_covid_19_confirmed.csv")
    args = parser.parse_args()

    main(args)
