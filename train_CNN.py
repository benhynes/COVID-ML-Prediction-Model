from models.CNN_model import *
from utils.datalib import *
from utils.plotting import *
import sklearn
import argparse


def train(args):

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

    coordinates = extract_coordinates(recovered_raw_dataset)

    n_countries = len(coordinates)

    dataset = confirmed_dataset.copy()

    for country in range(n_countries):
        dataset[country] = moving_average(confirmed_dataset[country],5)

    #statistical data
    x_median = np.median(dataset)
    q1 = np.quantile(dataset,0.25)
    q3 = np.quantile(dataset,0.75)
    
    #setup hyperparameters
    n_countries = len(coordinates)
    n_days = 10
    input_shape = (180, 360, n_days)
    output_shape = (180,360)
    Batch_Size = 8
    loss = []
    val_loss = []

    
    #Normalize
    Data_Formatter = CNN_Data_Formatter(input_shape,output_shape)
    normalized_dataset = Data_Formatter.robust_normalize(x = dataset, x_median = x_median, q1 = q1, q3 = q3)
    
    #spliting dataset
    x_train,x_valid = split_data(normalized_dataset)

    mask = get_mask(Batch_Size, coordinates)
    #Define model and load model
    model = CNN_Model(input_shape = input_shape,output_shape =  output_shape, mask = mask, lr = 2e-3)

    if args.weight_file!="":
        model.load_weights(args.weight_file)

    x_sample_set, y_sample_set = Data_Formatter.get_sample_set(x_train, coordinates, n_days)
    x_valid_set, y_valid_set = Data_Formatter.get_sample_set(x_valid, coordinates, n_days)

    full_x,_ = Data_Formatter.get_sample_set(normalized_dataset, coordinates, n_days)
    epoch = 0
    while True:
        epoch += 1

        #Get a mini batch
        minibatch_x, minibatch_y = Data_Formatter.get_minibatch(x_sample_set, y_sample_set, Batch_Size)

        #Train on batch
        batch_loss, _ = model.train_on_batch(minibatch_x,minibatch_y)

        #prepare loss vector to plot later
        loss.append(batch_loss)

        minibatch_x, minibatch_y = Data_Formatter.get_minibatch(x_valid_set, y_valid_set, Batch_Size)

        val_batch_loss, mae = model.model.test_on_batch(minibatch_x,minibatch_y)

        val_loss.append(val_batch_loss)

        #status
        if epoch%1==0:
            print("Epochs: ",epoch, "| loss: ", batch_loss, "| Val_loss: ",val_batch_loss, "| mae: ", mae)

        if epoch%100==0:
            #print example
            res = []
            for i in range(n_days):
                res.append(dataset[0][i])
            #A slide interval will scan through each interval of the actual valid data and give prediction for the next day, then add to the result vector
            for i in range(len(full_x)):
                x = np.reshape(full_x[i],(1,input_shape[0],input_shape[1],input_shape[2]))
                y = model.predict(x)
                res.append(Data_Formatter.robust_denormalize(y[coordinates[0][0],coordinates[0][1]],x_median = x_median, q1 =q1, q3= q3))
            res.append(0)

            
            
            #Plot the actual data
            
            res = np.array(res)
            print("MAE: ",sklearn.metrics.mean_absolute_error(dataset[0],res))

            #plot result vector
            plot_multiple_vectors([dataset[0],np.array(res)], 
                                title = "Time series plot and CNN prediction of confirmed cases for Afghanistan",
                                xlabel = "day ith",
                                ylabel = "number of confirmed cases",
                                legends = ['expected','predicted'],
                                f = 'preview'
                                )

            #plot
            plot_multiple_vectors(v = [loss], title = "Loss", xlabel = "epoch", legends = ['mse_loss'], f = "loss_graph")
            plot_multiple_vectors(v = [val_loss], title = "Validation Loss", xlabel = "epoch", legends = ['Val_mse_loss'], f = "Val_loss_graph")

            model.save_weights()
            print("Saving...")


    return 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-w', dest = 'weight_file', default = "")
    args = parser.parse_args()
    train(args)
