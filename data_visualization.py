from utils.plotting import *
from utils.datalib import *
from models.CNN_model import *

import sklearn

if __name__ == "__main__":
    
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

    dataset = confirmed_dataset.copy()
    for country in range(len(confirmed_dataset)):
        dataset[country] = moving_average(confirmed_dataset[country], 5)

    plot_multiple_vectors([confirmed_dataset[1],dataset[1]], legends = ['raw','moving_average'])