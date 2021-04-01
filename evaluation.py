from utils.data_preprocessing import *
import keras



if __name__ == "__main__":
    dataset = preprocess(read_csv('dataset/time_series_covid_19_confirmed.csv'))
    _, x_valid = split_data(dataset)

    Input_Shape = (len(dataset),30,)
    Output_Shape = (len(dataset),1)
    Batch_Size = 32

    Model = []
    # Initialize model and append to Model list
    model = FC_Model("Hello", input_shape = Input_Shape, output_shape = Output_Shape)
    model.load_weights("FC_model.h5")

    # Calculate score for each model following some metrics
    for m in Model:
        score.append(evaluate(m, r_test, label))

    # Print out score of each model
    for i in range(len(Model)):
        print('Model ',Model[i].name)
        print("Accuracy: ", score[i][0]," Number of parameters: ", score[i][1])
        print('')





