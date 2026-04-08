import pickle
import matplotlib.pyplot as plt
import torch
import pandas as pd
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
from NNLib import *
import math
import os
import pyprog

class TrainLoader(Dataset):
    def __init__(self, filename: str, num_anc: int):
        n_in = int(num_anc)*6 + 3*(int(num_anc)-1)
        file = pd.read_csv(filename)
        inputs = file.iloc[0:2500000, 0:n_in].values
        targets = file.iloc[0:2500000, n_in:n_in+3].values
        a1 = file.iloc[0:2500000, n_in+3:n_in+6].values
        self.inputs = torch.tensor(inputs, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32)
        self.a1 = torch.tensor(a1, dtype=torch.float32)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx], self.a1[idx]

def get_variance(arr):
    mean = sum(arr) / len(arr)
    summ = 0
    for x in arr:
        summ+=(x-mean)**2
    return mean, summ/len(arr)

def stdev(m):
    _, var = get_variance(m)
    return math.sqrt(var)

def main():
    v_means = []
    v_stdevs = []
    key = []
    loss = torch.nn.L1Loss()
    for i in range(8):
        os.system('clear')
        n_anc = i+3
        model = torch.load(str(n_anc)+' Anchors/FilterLocalizationNetwork-compressed.pt')
    
        open_file = open(str(n_anc)+" Anchors/training_results.pkl", "rb")
        t = pickle.load(open_file)
        open_file.close()   

        open_file = open(str(n_anc)+" Anchors/validation_results.pkl", "rb")
        v = pickle.load(open_file)
        open_file.close()

        plt.figure()
        plt.plot(v,color='black', label='Validation')
        plt.plot(t,color='gray', label = 'Training')
        plt.xlabel('Epoch [-]')
        plt.ylabel('Mean Absolute Error [m]')
        plt.legend()
        plt.title(str(n_anc)+" Anchors")
    plt.show()

if __name__ == '__main__':
    main()

        