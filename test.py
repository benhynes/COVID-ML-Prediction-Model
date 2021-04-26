from utils.plotting import *
from utils.datalib import *
from models.CNN_model import *
from sklearn.preprocessing import RobustScaler

import sklearn

if __name__ == "__main__":
    data_urls = ['https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv', 
        'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv',
        'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_recovered_global.csv']

    confirmed_raw_dataset = read_csv(data_urls[0])
    #deaths_raw_dataset = read_csv(data_urls[1])
    recovered_raw_dataset = read_csv(data_urls[2])

    confirmed_dataset = preprocess(np.reshape(confirmed_raw_dataset[0],(1,len(confirmed_raw_dataset[0]))))
    #deaths_dataset = preprocess(deaths_raw_dataset)
    #recovered_dataset = preprocess(recovered_raw_dataset)

    x_mean = np.mean(confirmed_dataset)
    x_std = np.std(confirmed_dataset)
    x_max = np.amax(confirmed_dataset)
    x_min = np.amin(confirmed_dataset)
    

    coordinates = extract_coordinates(np.reshape(recovered_raw_dataset[0],(1,len(recovered_raw_dataset[0]))))

    

    #x_train, x_valid, y_train, y_vaid = split_data(dataset)
    #x_plot_train, x_plot_valid = split_data(confirmed_dataset[0])

    n_days = 10
    n_countries = len(coordinates)
    input_shape = (180, 360, n_days)
    output_shape = (180,360)

    Data_Formatter = CNN_Data_Formatter(input_shape,output_shape)
    normalized_confirmed_dataset = confirmed_dataset#Data_Formatter.normalize(confirmed_dataset,x_max = x_max,x_min = x_min)
    mask = get_mask(1,coordinates)
    

    #Define and load weights for model
    model = CNN_Model("Hello", input_shape = input_shape, output_shape = output_shape, mask = mask, lr = 2e-3)
    model.load_weights("trained_models/CNN.h5")

    #Add the first 30 values into the vector since it will provide prediction from day 31
    res = []
    for i in range(n_days):
        res.append(normalized_confirmed_dataset[0][i])
    x_sample_set, y_sample_set = Data_Formatter.get_sample_set(normalized_confirmed_dataset, coordinates, n_days)
    #A slide interval will scan through each interval of the actual valid data and give prediction for the next day, then add to the result vector
    for i in range(len(x_sample_set)):
        x = np.reshape(x_sample_set[i],(1,input_shape[0],input_shape[1],input_shape[2]))
        y = model.predict(x)
        res.append(y[coordinates[0][0],coordinates[0][1]])
    res.append(0)

    
    
    #Plot the actual data
    
    res = np.array(res)
    print("MAE: ",sklearn.metrics.mean_absolute_error(normalized_confirmed_dataset[0],res))

    #plot result vector
    plot_multiple_vectors([normalized_confirmed_dataset[0],np.array(res)], 
                        title = "Time series plot and CNN prediction of new confirmed cases for Afghanistan",
                        xlabel = "day ith",
                        ylabel = "number of new confirmed cases",
                        legends = ['expected','predicted']
                        )