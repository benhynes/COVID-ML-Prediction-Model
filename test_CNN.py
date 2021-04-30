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

    dataset = confirmed_dataset

    x_median = np.median(dataset)

    q1 = np.quantile(dataset,0.25)
    q3 = np.quantile(dataset,0.75)

    #x_train, x_valid = split_data(dataset)

    n_days = 10
    n_countries = len(coordinates)
    input_shape = (180, 360, n_days)
    output_shape = (180,360)

    Data_Formatter = CNN_Data_Formatter(input_shape,output_shape)
    normalized_dataset = Data_Formatter.robust_normalize(dataset,x_median = x_median, q1 = q1, q3 = q3)
    

    #Define and load weights for model
    model = CNN_Model(input_shape = input_shape, output_shape = output_shape)
    model.load_weights()

    #Add the first 30 values into the vector since it will provide prediction from day 31
    res = []
    for i in range(n_days):
        res.append(dataset[0][i])
    x_sample_set, y_sample_set = Data_Formatter.get_sample_set(normalized_dataset, coordinates, n_days)
    #A slide interval will scan through each interval of the actual valid data and give prediction for the next day, then add to the result vector
    for i in range(len(x_sample_set)):
        x = np.reshape(x_sample_set[i],(1,input_shape[0],input_shape[1],input_shape[2]))
        y = model.predict(x)
        res.append(Data_Formatter.robust_denormalize(y[coordinates[0][0],coordinates[0][1]],x_median = x_median, q1 =q1, q3= q3))
    res.append(0)

    
    
    #Plot the actual data
    
    res = np.array(res)
    print("MAE: ",sklearn.metrics.mean_absolute_error(dataset[0],res))

    #plot result vector
    plot_multiple_vectors([dataset[0],np.array(res)], 
                        title = "Time series plot and CNN prediction of active cases for Afghanistan",
                        xlabel = "day ith",
                        ylabel = "number of active cases",
                        legends = ['expected','predicted'],
                        f = 'preview'
                        )