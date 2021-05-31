import argparse

from utils.datalib import *
from utils.plotting import *
from utils.csv_CNN import *

import matplotlib.pyplot as plt 

from Driver_CNN import driver_CNN
from Driver_IFC import driver_IFC


def export_past():
    data_urls = ['https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv', 
        'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv',
        'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_recovered_global.csv']

    confirmed_raw_dataset = read_csv(data_urls[0])
    #deaths_raw_dataset = read_csv(data_urls[1])
    recovered_raw_dataset = read_csv(data_urls[2])

    confirmed_dataset = preprocess(confirmed_raw_dataset)
    #deaths_dataset = preprocess(deaths_raw_dataset)
    #recovered_dataset = preprocess(recovered_raw_dataset)

    coordinates = extract_coordinates(recovered_raw_dataset)

    past_dataset = confirmed_dataset[:,len(confirmed_dataset[0])-9:len(confirmed_dataset[0])]

    past_map = get_loc_map(coordinates, past_dataset)

    past_map = list(past_map)

    parsePastToCSV(past_map)

    return past_map


def driver(args):

    data_urls = ['https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv', 
        'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv',
        'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_recovered_global.csv']

    confirmed_raw_dataset = read_csv(data_urls[0])
    #deaths_raw_dataset = read_csv(data_urls[1])
    recovered_raw_dataset = read_csv(data_urls[2])

    confirmed_dataset = preprocess(confirmed_raw_dataset)
    #deaths_dataset = preprocess(deaths_raw_dataset)
    #recovered_dataset = preprocess(recovered_raw_dataset)

    coordinates = extract_coordinates(recovered_raw_dataset)

    n_days = args.output_days

    ans = driver_CNN(args)
    ans_2 = driver_IFC(args)
    for country in range(len(confirmed_dataset)):
        for time in range(n_days):
                ans[time][coordinates[country][0]][coordinates[country][1]] = (ans[time][coordinates[country][0]][coordinates[country][1]] + ans_2[time][coordinates[country][0]][coordinates[country][1]])/2
    ans = np.around(ans)
    
    export_past()
    
    parseToCSV(ans)
    
    """
    #Demo Graph
    country_1 = []
    country_2 = []
    for i in range(len(ans)):
        country_1.append(ans[i][coordinates[0][0]][coordinates[0][1]])
        country_2.append(ans_2[i][coordinates[0][0]][coordinates[0][1]])

    plot_multiple_vectors([country_1,country_2,confirmed_dataset[0,len(confirmed_dataset[0])-n_days:len(confirmed_dataset[0])]],title = 'Honest graph', xlabel = 'ith day to the future', ylabel = 'number of new confirmed cases', legends=['predicted_CNN','predicted_IFC','expected'])
    """
    return ans


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', dest = 'output_days', default = 10)
    args = parser.parse_args()
    #cleanData()
    driver(args)
