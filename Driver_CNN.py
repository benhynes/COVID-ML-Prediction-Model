import argparse

from utils.datalib import *
from utils.plotting import *
from utils.csv_CNN import *
from models.CNN_model import *
from sklearn import preprocessing

import matplotlib.pyplot as plt 

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
    n_countries = len(coordinates)

    x_median = np.median(confirmed_dataset)
    q1 = np.quantile(confirmed_dataset,0.25)
    q3 = np.quantile(confirmed_dataset,0.75)

    
    #model preparing
    n_days = 10
    input_shape = (180, 360, n_days)
    past_input_shape = (180, 360, n_days - 1)
    output_shape = (180,360)
    Batch_Size = 1
    loss = []
    val_loss = []

    Data_Formatter = CNN_Data_Formatter(input_shape = input_shape, output_shape = output_shape)
    Past_Data_Formatter = CNN_Data_Formatter(input_shape = past_input_shape, output_shape = output_shape)
    normalized_dataset = Data_Formatter.robust_normalize(confirmed_dataset, x_median = x_median, q1 = q1, q3 = q3)

    model = CNN_Model(input_shape = input_shape, output_shape = output_shape)
    model.load_weights()
    
    #Need only n latest days to predict the future
    normalized_dataset = normalized_dataset[:,len(normalized_dataset[0])-n_days:len(normalized_dataset[0])]
    past_dataset = confirmed_dataset[:,len(confirmed_dataset[0])-9:len(confirmed_dataset[0])]
    

    #Create map
    normalized_map = get_loc_map(coordinates, normalized_dataset)
    past_map = get_loc_map(coordinates, past_dataset)
    
    #CNN_reshape reshapes (time,lat,long) to (lat,long,time)
    normalized_map = Data_Formatter.CNN_reshape(normalized_map)

    ans = []

    #Rolling and predicting
    for i in range(int(args.output_days)):
        x = np.array(normalized_map)
        y = model.predict(x)
        normalized_map = np.roll(normalized_map,shift = -1, axis = 2)
        empty = np.zeros((180,360))
        for country in range(len(coordinates)):
            normalized_map[coordinates[country][0]][coordinates[country][1]][-1] = y[coordinates[country][0]][coordinates[country][1]]
            empty[coordinates[country][0]][coordinates[country][1]] = max(0,np.around(Data_Formatter.robust_denormalize(y[coordinates[country][0]][coordinates[country][1]],x_median = x_median, q1 = q1, q3 = q3)))
        ans.append(empty.copy())
    
    past = list(past_map)
    
    parseToCSV(ans)
    parsePastToCSV(past)
    
    #Demo Graph
    country_1 = []
    country_2 = []
    for i in range(len(ans)):
        country_1.append(ans[i][coordinates[0][0]][coordinates[0][1]])

    plot_multiple_vectors([country_1,confirmed_dataset[0,len(confirmed_dataset[0])-n_days:len(confirmed_dataset[0])]],title = 'Honest graph', xlabel = 'ith day to the future', ylabel = 'number of new confirmed cases', legends=['predicted','expected'])

    return ans


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', dest = 'output_days', default = 10)
    args = parser.parse_args()

    driver(args)
