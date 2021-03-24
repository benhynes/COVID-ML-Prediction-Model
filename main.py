
import argparse
from utils.data_preprocessing import *
from utils.plotting import *

def main(args):
    x = [1,2,3,4,5,6,7,8,9]
    y = [1,5,4,2,6,8,5,6,6]
    loss = [1,5,4,3,2,1,1]
    plot(loss,x,y)
    
    
    return 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = 'Process some integers.')
    parser.add_argument('-f', dest = 'file_path', default = "dataset/test.csv")#time_series_covid_19_confirmed.csv")
    args = parser.parse_args()

    main(args)