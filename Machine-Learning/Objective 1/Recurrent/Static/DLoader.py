import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler

class TriLatData(Dataset):
    def __init__(self, filename):
        file = pd.read_csv(filename)
        x = file.iloc[1:1000000,0:5].values
        y = file.iloc[1:1000000, 5:8].values
        z = file.iloc[1:1000000, 8:11].values
        drone = file.iloc[1:1000000, 11:14].values
        r1 = file.iloc[1:1000000, 14:15].values
        x_train = x
        y_train = y
        self.x_train = torch.tensor(x_train, dtype = torch.float32)
        self.y_train = torch.tensor(y_train, dtype = torch.float32)
        self.anc_train = torch.tensor(z, dtype = torch.float32)
        self.drone = torch.tensor(drone, dtype = torch.float32)
        self.r1 = torch.tensor(r1, dtype = torch.float32)
    
    def __len__(self):
        return len(self.y_train)
    
    def __getitem__(self, idx):
        return self.x_train[idx], self.y_train[idx], self.anc_train[idx], self.drone[idx], self.r1[idx]