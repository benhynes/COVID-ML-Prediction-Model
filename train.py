from models.FC_model import FC_Model
from utils.data_preprocessing import *
from utils.plotting import *
import argparse

def train(args):
    dataset = preprocess(read_csv(args.data_path))
    x_train, x_valid = split_data(dataset)

    Input_Shape = 30
    Output_Shape = 1
    Batch_Size = 32

    #Declare model, choosing between Fully Connected and LSTM
    model = []
    if args.type == "FC":
        model = FC_Model("FC", input_shape = Input_Shape, output_shape = Output_Shape, lr = 0.00002)
    else: 
        model = LSTM_Model("LSTM", input_shape = Input_Shape, output_shape = Output_Shape)
    
    #Load weights if it exists
    if args.weight_file != "":
        model.load_weights(args.file_name)
    
    
    #some statistical analysis
    x_mean = np.mean(dataset)
    x_std = np.std(dataset)
    x_maximum = np.amax(dataset)
    x_minimum = np.amin(dataset)

    loss = []
    
    # Scale data into smaller range
    #x_train = x_train/x_maximum
    #x_valid = x_valid/x_maximum

    # Standardize the dataset to get Z standard normal distribution
    x_train = normialize(x_train,x_maximum)
    x_valid = normialize(x_valid,x_maximum)
    print(x_train)
    
    for epoch in range(20000):
        
        #Get a mini batch
        minibatch_x, minibatch_y = get_minibatch(x_train, Batch_Size, Input_Shape, Output_Shape)

        #Train on batch
        batch_loss, accurate = model.train_on_batch(minibatch_x,minibatch_y)

        #prepare loss vector to plot later
        loss.append(batch_loss)

        #print loss and examine prediction (interval day [50..80] and get prediction of day 81) every 100 epoches
        if epoch%100==0:
            print("Epoches: ",epoch, "loss: ", batch_loss, "accuracy: ", accurate)
            y_bar = model.predict(np.reshape(x_valid[50:80],(1,30)))
            print(denormialize(y_bar,x_maximum), denormialize(x_valid[80],x_maximum))

        #save model every 1000 epoches
        if epoch%1000==0:
            model.save_weights("FC_model.h5")

    plot_vector(loss)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', dest = 'type', default = "FC")
    parser.add_argument('-w', dest = 'weight_file', default = "")
    parser.add_argument('-d', dest = 'data_path', default = "dataset/time_series_covid_19_confirmed.csv")
    args = parser.parse_args()
    train(args)
