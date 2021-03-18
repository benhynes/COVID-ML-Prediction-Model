
import argparse
from utils.data_preprocessing import *

def main(args):
    raw_dataset = read_csv(args.file_path)
    rtrain = data_preprocess(raw_dataset)
    print(rtrain)
    return 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = 'Process some integers.')
    parser.add_argument('-f', dest = 'file_path', default = "dataset/test.csv")#time_series_covid_19_confirmed.csv")
    args = parser.parse_args()

    main(args)