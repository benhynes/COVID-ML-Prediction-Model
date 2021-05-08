from utils.plotting import *
from utils.datalib import *
from models.IFC_model import *

import sklearn

if __name__ == "__main__":
    
    #Cloning dataset
    data_urls = ['https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv', 
        'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv',
        'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_recovered_global.csv']

    confirmed_raw_dataset = read_csv(data_urls[0])
    #deaths_raw_dataset = read_csv(data_urls[1])
    recovered_raw_dataset = read_csv(data_urls[2])

    confirmed_dataset = preprocess(confirmed_raw_dataset)
    #deaths_dataset = preprocess(deaths_raw_dataset)
    #recovered_dataset = preprocess(recovered_raw_dataset)

    dataset = confirmed_dataset
    coordinates = extract_coordinates(recovered_raw_dataset)
    #statistical data
    x_median = np.median(dataset)
    q1 = np.quantile(dataset,0.25)
    q3 = np.quantile(dataset,0.75)
    x_max = np.amax(dataset)
    #setup hyperparameters
    n_days = 10
    input_shape = n_days+2
    output_shape = 1
    loss = []
    val_loss = []

    #Define model
    model = FC_Model(input_shape = input_shape, output_shape = output_shape)

    Data_Formatter = FC_Data_Formatter(input_shape, output_shape)
    normalized_dataset = Data_Formatter.robust_normalize(dataset, x_median = x_median, q1 = q1, q3 = q3)
    

    #Define and load weights for model
    model = FC_Model(input_shape = input_shape, output_shape = output_shape)
    model.load_weights()

    #Add the first 30 values into the vector since it will provide prediction from day 31
    res = []
    for i in range(n_days):
        res.append(dataset[0][i])
    
    #A slide interval will scan through each interval of the actual valid data and give prediction for the next day, then add to the result vector
    for i in range(len(normalized_dataset[0])-n_days):
        x = [coordinates[0][0], coordinates[0][1]]
        for j in range(n_days):
            x.append(normalized_dataset[0][j+n_days])
        x = np.array(x)
        y = Data_Formatter.robust_denormalize(model.predict(x),x_max = x_max)
        res.append(max(0,y))    

    
    
    #Plot the actual data
    
    res = np.array(res)
    print(len(dataset[0]))

    print("MAE: ",sklearn.metrics.mean_absolute_error(dataset[0],res))

    #plot result vector
    plot_multiple_vectors([dataset[0],np.array(res)], 
                        title = "Time series plot and FC predictions of confirmed cases for Afghanistan",
                        figsize= (15,5),
                        xlabel = "day ith",
                        ylabel = "number of new confirmed cases",
                        legends = ['expected','predicted'],
                        f = 'preview'
                        )