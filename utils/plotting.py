import matplotlib.pyplot as plt 
import numpy as np


def plot_vector(v):
    plt.plot(v)
    plt.show()
    return

def plot_multiple_vectors(v):
    for vector in v:
        plt.plot(vector)
    plt.show()

def plot(loss, time, number_of_cases):
    fig = plt.figure(figsize = (15,5))
    plot1 = fig.add_subplot(1,2,1)
    plot2 = fig.add_subplot(1,2,2)
    
    # Loss plot
    plot1.title.set_text("Loss")
    plot1.set_xlabel("Epoch")
    plot1.plot(loss)

    # Prediction plot
    plot2.title.set_text("Prediction")
    plot2.set_xlabel("Date")
    plot2.set_ylabel("Number of confirmed cases")
    plot2.plot(time,number_of_cases)
    plt.show()
    return