import torch
import numpy as np
import matplotlib.pyplot as plt
from DLoader import *
from torch.utils.data import DataLoader
import os

def main():
    print("Loading Data")
    model = torch.load("FilterLocalizationNetwork-compressed.pt")
    val = TrainLoader('Validation_data.csv')
    validation_loader = DataLoader(val, batch_size=1, shuffle=False)
    cdf_results = []
    loss = torch.nn.L1Loss()
    os.system('clear')
    device = torch.device('cuda')
    with torch.no_grad():
        print("Using Validation Data ... ")
        for data, target, a1 in validation_loader:
            data, target = data.to(device), target.to(device)
            guess = model(data)
            er = loss(guess,target)
            cdf_results.append(er.item())


    count, bins_count = np.histogram(cdf_results, bins=1000)
    pdf = count / sum(count)
    cdf = np.cumsum(pdf)
    nf = 0
    diff = 100
    for i in range(len(cdf)):
        if cdf[i] >= 0.949 and cdf[i] <= 0.951:
            print(cdf[i],": ", bins_count[i+1])
            if abs(cdf[i]-0.95) < diff:
                nf = i
                diff = abs(cdf[i]-0.95)
            
    horizontalx = [0, bins_count[nf]]
    horizontaly = [cdf[nf], cdf[nf]]
    verticalx = [bins_count[nf],bins_count[nf]]
    verticaly = [0, cdf[nf]]
    plt.plot(bins_count[1:], cdf, color='black', label="Estimate using ANN")
    plt.xlabel("Mean Absolute Error [m]")
    plt.ylabel("CDF [%]")
    plt.plot(horizontalx, horizontaly, linestyle='dotted', color='black')
    plt.plot(verticalx, verticaly, linestyle='dotted', color='black')
    plt.annotate(text=str(bins_count[nf])[:5]+' m', xy=(bins_count[nf]+0.1, cdf[nf]-0.05))
    plt.ylim([0, 1])
    plt.xlim([0,10])
    plt.legend()
    plt.show()

    plt.plot(bins_count[1:], cdf, color='black', label="Estimate using ANN")
    plt.xlabel("Mean Absolute Error [m]")
    plt.ylabel("CDF [%]")
    plt.ylim([0, 1])
    plt.xlim([0,10])
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
