import argparse

from utils.datalib import *
from utils.plotting import *
from utils.csv_CNN import *
from models.IFC_model import *
from sklearn import preprocessing

import matplotlib.pyplot as plt 

def driver_IFC(args):
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
    n_countries = len(coordinates)

    x_median = np.median(confirmed_dataset)
    q1 = np.quantile(confirmed_dataset,0.25)
    q3 = np.quantile(confirmed_dataset,0.75)

    
    #model preparing
    n_days = 10
    input_shape = n_days + 2
    output_shape = 1


    Data_Formatter = FC_Data_Formatter(input_shape = input_shape, output_shape = output_shape)
    normalized_dataset = Data_Formatter.robust_normalize(confirmed_dataset, x_median = x_median, q1 = q1, q3 = q3)

    model = FC_Model(input_shape = input_shape, output_shape = output_shape)
    model.load_weights()
    
    #Need only n latest days to predict the future
    normalized_dataset = normalized_dataset[:,len(normalized_dataset[0])-n_days:len(normalized_dataset[0])]
    
    x = []
    for country in range(n_countries):
        temp = [Data_Formatter.lat_normalize(coordinates[country][0]),Data_Formatter.long_normalize(coordinates[country][1])]
        for date in range(len(normalized_dataset[country])):
            temp.append(normalized_dataset[country][date])
        x.append(temp.copy())


    ans = []

    #Rolling and predicting
    for i in range(int(args.output_days)):
        y = np.reshape(model.predict(np.array(x)),(n_countries))
        ans.append(np.around(Data_Formatter.robust_denormalize(y,x_median = x_median, q1 = q1, q3 = q3)))
        for country in range(n_countries):
            for date in range(n_days-1):
                x[country][date+2] = x[country][date+3]
            x[country][11] = y[country]
    
    ans = get_loc_map(coordinates, np.array(ans).transpose())
    
    """
    #Demo Graph
    country_1 = []
    for i in range(len(ans)):
        country_1.append(ans[i][coordinates[0][0]][coordinates[0][1]])

    plot_multiple_vectors([country_1],title = 'Honest graph', xlabel = 'ith day to the future', ylabel = 'number of new confirmed cases', legends=['predicted'], f = 'preview')
    """
    return ans


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', dest = 'output_days', default = 10)
    args = parser.parse_args()

    driver_IFC(args)
