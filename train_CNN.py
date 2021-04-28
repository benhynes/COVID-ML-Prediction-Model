from models.CNN_model import CNN_Model, CNN_Data_Formatter
from utils.datalib import *
from utils.plotting import *
import sklearn
import argparse
from sklearn.preprocessing import robust_scale


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
    n_countries = len(coordinates)


    x_mean = np.mean(confirmed_dataset)
    x_std = np.std(confirmed_dataset)
    x_max = np.amax(confirmed_dataset)
    x_min = np.amin(confirmed_dataset)
    x_median = np.median(confirmed_dataset)

    q1 = np.quantile(confirmed_dataset,0.25)
    q3 = np.quantile(confirmed_dataset,0.75)
    


    n_days = 10
    input_shape = (180, 360, n_days)
    output_shape = (180,360)
    Batch_Size = 64
    loss = []
    val_loss = []

    
    
    Data_Formatter = CNN_Data_Formatter(input_shape,output_shape)
    normalized_dataset = Data_Formatter.normalize(confirmed_dataset,x_median = x_median, q1 = q1, q3 = q3)
    mask = get_mask(Batch_Size, coordinates)
    epoch = 0

    x_train,x_valid = split_data(normalized_dataset.transpose())

    x_train = x_train.transpose()
    x_valid = x_valid.transpose()

    model = CNN_Model("CNN", input_shape, output_shape, mask = mask, lr = 2e-3)
    #model.load_weights()

    x_sample_set, y_sample_set = Data_Formatter.get_sample_set(x_train, coordinates, n_days)
    x_valid_set, y_valid_set = Data_Formatter.get_sample_set(x_valid, coordinates, n_days)
    for epoch in range(10000):

        epoch += 1

        #Get a mini batch
        minibatch_x, minibatch_y = Data_Formatter.get_minibatch(x_sample_set, y_sample_set, Batch_Size)

        #Train on batch
        batch_loss, mae = model.train_on_batch(minibatch_x,minibatch_y)

        #prepare loss vector to plot later
        loss.append(batch_loss)

        minibatch_x, minibatch_y = Data_Formatter.get_minibatch(x_valid_set, y_valid_set, Batch_Size)
        #minibatch_x, minibatch_y = Data_Formatter.get_minibatch(x_valid, Batch_Size)
        val_batch_loss, _ = model.model.test_on_batch(minibatch_x,minibatch_y)

        val_loss.append(val_batch_loss)

        #status
        if epoch%1==0:
            print("Epochs: ",epoch, "| loss: ", batch_loss, "| Val_loss: ",val_batch_loss, "| mae: ", mae)

        if epoch%100==0:
            x = model.predict(minibatch_x)[0,coordinates[0][0],coordinates[0][1],0]
            y = minibatch_y[0,coordinates[0][0],coordinates[0][1]]
            print(x," ",y)
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
