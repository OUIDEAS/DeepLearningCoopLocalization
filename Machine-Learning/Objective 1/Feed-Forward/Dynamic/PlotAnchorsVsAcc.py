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
        print("Loading Neural Network...")
        model = torch.load(str(n_anc)+' Anchors/FilterLocalizationNetwork-compressed.pt')
        os.system('clear')
        print("Loading Validation Data Set")
        data = TrainLoader(str(n_anc)+' Anchors/Validation_data.csv', num_anc = n_anc)
        validation = DataLoader(data, batch_size = 1, shuffle=False)
        os.system('clear')
        
        V = []
        iteration = 0
        print("Testing Neural Network on " + str(n_anc)+" Anchors.")
        prog = pyprog.ProgressBar("-> ", " OK!", len(data))
        prog.update()
        with torch.no_grad():
            for inputs, target, _ in validation:
                inputs, target = inputs.to(torch.device('cuda')), target.to(torch.device('cuda'))
                pos = model(inputs)
                error = loss(target, pos)
                V.append(error.item())
                iteration += 1
                prog.set_stat(iteration)
                prog.update()
        
        v_means.append(sum(V)/len(V))
        v_stdevs.append(stdev(V))
        key.append(i+3)

    open_file = open("means.pkl", "wb")
    pickle.dump(v_means, open_file)
    open_file.close()   

    open_file = open("stdevs.pkl", "wb")
    pickle.dump(v_stdevs, open_file)
    open_file.close()   

    open_file = open("keys.pkl", "wb")
    pickle.dump(key, open_file)
    open_file.close()    


    plt.figure()
    plt.errorbar(key, v_means, yerr=v_stdevs)
    plt.xlabel('Number of Anchors [-]')
    plt.ylabel('Validation Error [m]')
    plt.show()

if __name__ == '__main__':
    main()

        