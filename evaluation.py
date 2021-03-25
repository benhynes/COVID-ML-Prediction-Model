from utils.data_preprocessing import *
import keras

def accuracy(prediction, label, threshold)
    number_of_success = 0
    for i in range(len(prediction)):
        if (MSE(prediction[i], label[i]) < threshold):
            number_of_success += 1
    return number_of_success/len(label)

def evaluate(Model, rtest, label)
    prediction = Model.predict(rtest)
    accuracy_metric = accuracy(prediction,label, 100)
    number_of_parameters = Model.count_params()
    return [accuracy_metric, number_of_parameters]



raw_dataset = read_csv('datax/applemobilitytrends-2021-03-15.csv')
_, rtest = data_preprocess(raw_dataset)

Model = []
# Initialize model and append to Model list

# Calculate score for each model following some metrics
for m in Model:
    score.append(evaluate(m, r_test, label))

# Print out score of each model
for i in range(len(Model)):
    print('Model ',Model[i].name)
    print("Accuracy: ",score[i][0], " Number of parameters: ", score[i][1])
    print('')





