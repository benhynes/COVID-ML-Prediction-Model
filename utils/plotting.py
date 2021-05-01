import matplotlib.pyplot as plt 
import numpy as np

def plot_multiple_vectors(v, figsize = (15,5), title = None, xlabel = None, ylabel = None, legends = None, f = None):
    plt.figure(figsize = figsize)
    for vector in v:
        plt.plot(vector)
    if title!=None:
        plt.title(title)
    if xlabel != None:
        plt.xlabel(xlabel)
    if ylabel != None:
        plt.ylabel(ylabel)
    if legends!= None:
        plt.legend(legends)
    if f != None:
        plt.savefig(f)
    plt.show()
