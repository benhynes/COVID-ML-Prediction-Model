
import argparse
from utils.data_preprocessing import *
from utils.plotting import *
from sklearn.model_selection import train_test_split
from models.FC_model import FC_Model
from sklearn.preprocessing import Normalizer

def main(args):
    dataset = read_csv(args.file_path)
    #plot_vector(dataset[0])
    IS = 30
    OS = 1
    batch_size = 32
    Predictor = FC_Model(name = "Hello_World",input_shape = IS, output_shape = OS, lr = 0.0002)
    loss = []
    data = np.array(dataset)[0]
    #maximum = np.amax(data)
    x_mean = np.mean(data)
    x_std = np.std(data)
    x_train,x_valid, _, _ = train_test_split(data,data,test_size = 0.3, shuffle = False)
    #x_train = x_train/maximum
    #x_valid = x_valid/maximum
    x_train = (x_train - x_mean) / x_std
    x_valid = (x_valid - x_mean) / x_std
    print(x_train)
    for epoch in range(10000):
        x_batch = []
        y_batch = []
        for _ in range(batch_size):
            x, y = sampling(x_train,IS,OS)
            x_batch.append(x)
            y_batch.append(y)
        x_batch = np.reshape(np.asarray(x_batch),(batch_size,IS))
        y_batch = np.reshape(np.asarray(y_batch),(batch_size,OS))
        res = Predictor.train_on_batch(x_batch,y_batch)
        loss.append(res)
        if epoch%100==0:
            print("Epoches: ",epoch, "loss: ", res)
            y_bar = Predictor.predict(np.reshape(x_valid[0:IS],(1,IS)))
            print((y_bar*x_std) + x_mean, (x_valid[IS]*x_std) + x_mean)
        #if epoch%1000==0:
            #FC_Model.save_weights("model.h5")

    #y_bar = Predictor.predict(np.reshape(data[0][100:100+IS],(1,IS)))
    #y_bar = scaler.inverse_transform(y_bar)
    #print(y_bar*maximum, dataset[0][100+IS])
    plot_vector(loss)
    return 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = 'Process some integers.')
    parser.add_argument('-f', dest = 'file_path', default = "dataset/time_series_covid_19_confirmed.csv")
    args = parser.parse_args()

    main(args)
