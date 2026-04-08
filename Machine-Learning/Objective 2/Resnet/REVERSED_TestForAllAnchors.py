import torch
import torch.nn as nn
import torch.optim as optim
from NNLib import *
import matplotlib.pyplot as plt
import os
import time
from torch.utils.data import DataLoader
import sys
from dadjokes import Dadjoke
import pickle
from tqdm import tqdm
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler


class TrainLoader(Dataset):
    def __init__(self, filename, num_anc):
        n_in = num_anc*6 + 3*(num_anc-1)
        file = pd.read_csv(filename)
        inputs = file.iloc[0:2500000, 0:n_in].values
        targets = file.iloc[0:2500000, n_in:n_in+3].values
        self.inputs = torch.tensor(inputs, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]

def main():
    os.system('clear')
    PATH = "ResNet-NEW.pt"
    dadjoke = Dadjoke()
    print(dadjoke.joke, '\n')
    print("Loading Data...")
    epochs = 392
    learning_rate = 1.335e-4

    for i in reversed(range(8)):
        os.system('clear')
        training = []
        valid = []
        num_anc = 3+i
        n_in = num_anc*6 + 3*(num_anc-1)
        neural_net = ResNet(n_in = n_in, size = 1220, res_layers = 3, residuals = 5, drop = 0.1067)

        System = Localization(neural_net, optim.RAdam(neural_net.parameters(), lr=learning_rate, weight_decay=1e-5), nn.L1Loss())
        print(System.Network)
        neural_net = None 
        train = TrainLoader('/home/rgeng98/BobGeng-Thesis/Machine-Learning/Objective 1/Feed-Forward/Dynamic/'+str(num_anc)+' Anchors/Training_data.csv', num_anc)
        val = TrainLoader('/home/rgeng98/BobGeng-Thesis/Machine-Learning/Objective 1/Feed-Forward/Dynamic/'+str(num_anc)+' Anchors/Validation_data.csv', num_anc)

        train_loader = DataLoader(train, batch_size=805, shuffle=True)
        validation_loader = DataLoader(val, batch_size=1000, shuffle=True)
        print("Training Residual Neural Network for ", str(num_anc), " Anchor UAVs.\n")
        for _ in tqdm(range(epochs)):
            accv = []
            acct = []
            for data, target in train_loader:
                _, loss = System.train(data, target)
                acct.append(loss)
            training.append(sum(acct)/len(acct))

            with torch.no_grad():
                for data, target in validation_loader:
                    _, loss = System.test(data, target)
                    accv.append(loss)
            valid.append(sum(accv)/len(accv))
        torch.save(System.Network, str(num_anc)+PATH)
        System = None
        open_file = open(str(num_anc)+"training_results.pkl", "wb")
        pickle.dump(training, open_file)
        open_file.close()

        file2 = open(str(num_anc)+"validation_results.pkl", "wb")
        pickle.dump(valid, file2)
        file2.close()
    
    exit()
    

if __name__ == "__main__":
    # Add save boolean too
    main()
