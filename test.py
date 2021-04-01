from utils.plotting import *
from utils.data_preprocessing import *
from models.FC_model import FC_Model

if __name__ == "__main__":
    raw_dataset = read_csv('dataset/time_series_covid_19_confirmed.csv')
    dataset = preprocess(raw_dataset)

    x_mean = np.mean(dataset)
    x_std = np.std(dataset)
    x_maximum = np.amax(dataset)
    x_minimum = np.amin(dataset)

    _, x_valid = split_data(dataset)
    _, raw_x_valid = split_data(raw_dataset)

    #Add the first 30 values into the vector since it will provide prediction from day 31
    res = []
    for i in range(30):
        res.append(x_valid[i])

    #Plot the actual data
    plt.plot(x_valid)

    #Standardize
    x_valid = normialize(x_valid,x_maximum)

    #Define and load weights for model
    model = FC_Model("Hello", input_shape = 30, output_shape = 1)
    model.load_weights("FC_model.h5")
    
    #A slide interval will scan through each interval of the actual valid data and give prediction for the next day, then add to the result vector
    for i in range(len(x_valid)-30):
        x = np.reshape(x_valid[i:i+30],(1,30))
        y = denormialize(model.predict(x)[0][0],x_maximum)
        res.append(y)

    #plot result vector
    plt.plot(np.array(res),'r-')
    plt.show()