import numpy as np
import pylab as plt

def Nsamp_plot():
    
    


def prob_plot():


def loader(fyle):
    acc = np.array([])  ;  std = acc.copy()
    with open(fyle,'rb') as source:
        np.delete(pickle.load(source))

        acc = acc.append(pickle.load(obj))
        std = std.append(pickle.load(obj))
        return acc, std
        

def plot_w_err(x,y,xerr=None,yerr=None,xlabel="N",ylabel="Accuracy"):
    """Plot sample size variation results"""
    plt.clf()
    fig = pl.figure()
    ax = fig.add_subplot(1,1,1)

    if xerr or yerr: ax.errorbar(x,y,yerr=yerr,xerr=xerr)
    if x == x[0]**np.arange(len(x)): ax.set_xscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    plt.show()
