from models.FC_model import *
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
    #recovered_raw_dataset = read_csv(data_urls[2])

    confirmed_dataset = preprocess(confirmed_raw_dataset)
    #deaths_dataset = preprocess(deaths_raw_dataset)
    #recovered_dataset = preprocess(recovered_raw_dataset)

    dataset = confirmed_dataset[0]
    
    #statistical data
    x_median = np.median(dataset)
    q1 = np.quantile(dataset,0.25)
    q3 = np.quantile(dataset,0.75)
    
    #setup hyperparameters
    n_days = 10
    input_shape = n_days
    output_shape = 1
    Batch_Size = 128
    loss = []
    val_loss = []

    #Define model
    model = FC_Model(input_shape = input_shape, output_shape = output_shape, lr = 2e-3)
    if args.weight_file!="":
        model.load_weights(args.weight_file)

    Data_Formatter = FC_Data_Formatter(input_shape, output_shape)
    normalized_dataset = Data_Formatter.robust_normalize(dataset, x_median = x_median, q1 = q1, q3 = q3)

    x_train, x_valid = split_data(np.reshape(normalized_dataset,(1,len(normalized_dataset))))
    x_train = x_train[0]
    x_valid = x_valid[0]

    
    for epoch in range(1000):
        
        #Get a mini batch
        minibatch_x, minibatch_y = Data_Formatter.get_minibatch(dataset = x_train, batch_size = Batch_Size)

        #Train on batch
        batch_loss, _ = model.train_on_batch(minibatch_x,minibatch_y)

        minibatch_x, minibatch_y = Data_Formatter.get_minibatch(dataset = x_valid, batch_size = Batch_Size)
        val_batch_loss, mae = model.model.test_on_batch(minibatch_x,minibatch_y)

        

        if epoch%1==0:
            print("Epochs: ",epoch, "| loss: ", batch_loss, "| Val_loss: ",val_batch_loss, "| mae: ", mae)
            val_loss.append(val_batch_loss)
            loss.append(batch_loss)

        #save model every 1000 epoches
        if epoch%1000==0:
            model.save_weights()

    plot_multiple_vectors([loss,val_loss])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-w', dest = 'weight_file', default = "")
    args = parser.parse_args()
    train(args)
