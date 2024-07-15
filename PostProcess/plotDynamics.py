import numpy as np
import matplotlib.pyplot as plt 
import parameters as param
import os

for imol in range(1, param.NMol+1):

    pBt_new = np.loadtxt(f"Data/1/pBt_1_0{imol}.txt")
    pBt_old = np.loadtxt(f"../Data/1/pBt_0{imol}_chebyshev.txt")
    os.makedirs(f"comparisonPlots/pBt_0{imol}", exist_ok=True)
    
    time_new = np.linspace(0, param.SimTime, pBt_new.shape[0])
    time_old = np.linspace(0, param.SimTime, pBt_old.shape[0])

    for iState in range(param.NStates):
        print(iState)
        plt.figure(iState)
        plt.plot(time_old, pBt_old[:, iState], lw=5, label='Old Chebyshev')
        plt.plot(time_new, pBt_new[:, iState], lw=3, label='New Chebyshev')
        plt.legend()
        plt.savefig(f"comparisonPlots/pBt_0{imol}/{iState}.png", dpi=300)
        plt.close()