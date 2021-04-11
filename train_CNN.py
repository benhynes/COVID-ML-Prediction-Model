from models.CNN_model import CNN_Model, CNN_Data_Formatter
from utils.datalib import *
from utils.plotting import *
import sklearn
import argparse

def train(args):
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
    number_of_countries = len(coordinates)


    x_mean = np.mean(confirmed_dataset)
    x_std = np.std(confirmed_dataset)
    confirmed_dataset = normalize(confirmed_dataset,x_mean,x_std)

    datamap = get_loc_map(coordinates, confirmed_dataset)
    

    x_train,x_valid, y_train, y_valid = split_data(datamap, confirmed_dataset.transpose())

    number_of_input_days = 30
    input_shape = (180, 360, number_of_input_days)
    output_shape = number_of_countries
    Batch_Size = 16
    loss = []
    val_loss = []

    model = CNN_Model("CNN", input_shape, output_shape, lr = 0.00002)
    #model.load_weights()
    Data_Formatter = CNN_Data_Formatter(input_shape,output_shape)

    for epoch in range(5000):

        #Get a mini batch
        minibatch_x, minibatch_y = Data_Formatter.get_minibatch(x_train,y_train, Batch_Size)

        #Train on batch
        batch_loss, mae = model.train_on_batch(minibatch_x,minibatch_y)

        #prepare loss vector to plot later
        loss.append(batch_loss)


        minibatch_x, minibatch_y = Data_Formatter.get_minibatch(x_valid,y_valid, Batch_Size)
        val_batch_loss,_ = model.model.test_on_batch(minibatch_x,minibatch_y)

        val_loss.append(val_batch_loss)

        #print loss and examine prediction (interval day [50..80] and get prediction of day 81) every 100 epoches
        if epoch%100==0:
            print("Epoches: ",epoch, "| loss: ", batch_loss, "| Val_loss: ",val_batch_loss, "| mae: ", mae)
            #y_bar = model.predict(np.reshape(x_valid[50:80],(1,input_shape[0], input_shape[1], input_shape[2])))

        if epoch%500==0:
            model.save_weights()
            print("Saving...")


    

    plot_multiple_vectors(v = [loss,val_loss], title = "Loss", xlabel = "epoch", legends = ['mse_loss','mse_val_loss'])

    return 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-w', dest = 'weight_file', default = "CNN_model.h5")
    parser.add_argument('-d', dest = 'data_path', default = "dataset/time_series_covid_19_confirmed.csv")
    args = parser.parse_args()
    train(args)
