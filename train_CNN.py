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
    deaths_raw_dataset = read_csv(data_urls[1])
    recovered_raw_dataset = read_csv(data_urls[2])

    confirmed_dataset = preprocess(confirmed_raw_dataset)
    deaths_dataset = preprocess(deaths_raw_dataset)
    recovered_dataset = preprocess(recovered_raw_dataset)

    coordinates = extract_coordinates(recovered_raw_dataset)

    dataset = confirmed_dataset

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

    #Define model and load model
    model = CNN_Model(input_shape = input_shape,output_shape =  output_shape, lr = 2e-3)

    if args.weight_file!="":
        model.load_weights(args.weight_file)

    x_sample_set, y_sample_set = Data_Formatter.get_sample_set(x_train, coordinates, n_days)
    x_valid_set, y_valid_set = Data_Formatter.get_sample_set(x_valid, coordinates, n_days)

    epoch = 0
    for epoch in range(1000):

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
            x = model.predict(minibatch_x)[0,coordinates[0][0],coordinates[0][1],0]
            y = minibatch_y[0,coordinates[0][0],coordinates[0][1]]
            print(x," ",y)

            #plot
            plot_multiple_vectors(v = [loss], title = "Loss", xlabel = "epoch", legends = ['mse_loss'], f = "loss_graph")
            plot_multiple_vectors(v = [val_loss], figsize = (20,5), title = "Validation Loss", xlabel = "epoch", legends = ['Val_mse_loss'], f = "Val_loss_graph")

            model.save_weights()
            print("Saving...")


    return 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-w', dest = 'weight_file', default = "")
    args = parser.parse_args()
    train(args)
