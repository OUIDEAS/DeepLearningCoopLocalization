import torch
import torch.nn as nn
import torch.optim as optim
from NNLib import *
import os
from torch.utils.data import DataLoader
import pickle
from tqdm import tqdm
import pandas as pd
import torch
from torch.utils.data import Dataset

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
    PATH = "ResNet_noci.pt"
    epochs = 392
    learning_rate = 1.335e-4
    for i in range(7):
        os.system('clear')
        print("Loading Data...")
        num_anc = 4+i
        device = torch.device('cuda') if torch.cuda.is_available() else torch.deivce('cpu')
        neural_net = ResNet(n_in = num_anc*6 + 3*(num_anc-1), size = 1220, res_layers = 3, residuals = 5, drop = 0).to(device)
        optimizer = optim.RAdam(neural_net.parameters(), lr = learning_rate, weight_decay=1e-5)
        loss = nn.L1Loss()
        # train = TrainLoader('DATA/'+str(num_anc)+'Training_data.csv', num_anc)
        # val = TrainLoader('DATA/'+str(num_anc)+'Validation_data.csv', num_anc)
        train = TrainLoader('/home/rgeng98/BobGeng-Thesis/Machine-Learning/Objective 1/Feed-Forward/Dynamic/'+str(num_anc)+" Anchors/Training_data.csv", num_anc)
        val = TrainLoader('/home/rgeng98/BobGeng-Thesis/Machine-Learning/Objective 1/Feed-Forward/Dynamic/'+str(num_anc)+" Anchors/Validation_data.csv", num_anc)
        methods = ["train", "validate"]
        loader = {
            methods[0]: DataLoader(train, batch_size=805, shuffle=True),
            methods[1]: DataLoader(val, batch_size=1000, shuffle=True)
        }
        errors = {
            methods[0]: [],
            methods[1]: []
        }
        print("Training Residual Neural Network for ", str(num_anc), " Anchor UAVs.\n")
        for _ in tqdm(range(epochs)):
            for method in methods:
                if method == "train":
                    neural_net.train()
                else:
                    neural_net.eval()
                acc = []
                for data, target in loader[method]:
                    data, target = data.to(device), target.to(device)
                    output = neural_net(data)
                    error = loss(output, target)
                    if method == "train":
                        optimizer.zero_grad()
                        error.backward()
                        optimizer.step()
                    acc.append(error.item())
                errors[method].append(sum(acc)/len(acc))
        torch.save(neural_net, "Networks/"+str(num_anc)+PATH)   
        open_file = open(str(num_anc)+"results.pkl", "wb")
        pickle.dump(errors, open_file)
        open_file.close()
    exit()

if __name__ == "__main__":
    main()