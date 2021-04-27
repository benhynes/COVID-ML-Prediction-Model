import argparse

from utils.datalib import *
from utils.plotting import *
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

    

    
    #model preparing
    n_days = 10
    input_shape = (180, 360, n_days)
    output_shape = (180,360)
    Batch_Size = 1
    loss = []
    val_loss = []

    Data_Formatter = CNN_Data_Formatter(input_shape,output_shape)
    normalized_dataset = confirmed_dataset
    mask = get_mask(Batch_Size, coordinates)
    model = CNN_Model("CNN", input_shape, output_shape, mask = mask, lr = 2e-2)
    model.load_weights()
    
    #Need only n latest days to predict the future
    normalized_dataset = normalized_dataset[:,len(normalized_dataset[0])-n_days:len(normalized_dataset[0])]

    #Create map
    normalized_map = get_loc_map(coordinates, normalized_dataset)

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
            empty[coordinates[country][0]][coordinates[country][1]] = y[coordinates[country][0]][coordinates[country][1]]
        ans.append(empty.copy())
    
    #Demo Graph
    country_1 = []
    for i in range(len(ans)):
        country_1.append(ans[i][coordinates[0][0]][coordinates[0][1]])
    plot_vector(country_1)

    return ans


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', dest = 'output_days', default = 10)
    args = parser.parse_args()

    driver(args)
