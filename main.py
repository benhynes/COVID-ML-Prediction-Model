
import argparse
from utils.data_preprocessing import *
from utils.plotting import *
from models.FC_model import FC_Model
from sklearn import preprocessing

import matplotlib.pyplot as plt 

def main(args):
    dataset = preprocess(read_csv(args.file_path))
    x_mean = np.median(dataset)
    x_std = np.std(dataset)
    x_maximum = np.amax(dataset)
    x_minimum = np.amin(dataset)

    scaler = preprocessing.StandardScaler()

    #x = preprocessing.normalize(dataset, norm='l2')
    x = normialize(dataset,x_maximum)
    plot_vector(x)
    plt.show()
    return 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', dest = 'file_path', default = "dataset/time_series_covid_19_confirmed.csv")
    args = parser.parse_args()

    main(args)
