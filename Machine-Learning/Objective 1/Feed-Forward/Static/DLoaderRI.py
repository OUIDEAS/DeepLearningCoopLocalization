import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler

class TrainLoader(Dataset):
    def __init__(self, filename):
        file = pd.read_csv(filename)
        ranges = file.iloc[1:1000000,0:12].values
        targets = file.iloc[1:1000000, 12:15].values
        a1 = file.iloc[1:1000000, 15:18].values
        a2 = file.iloc[1:1000000, 18:21].values
        a3 = file.iloc[1:1000000, 21:24].values
        a4 = file.iloc[1:1000000, 24:27].values
        a5 = file.iloc[1:1000000, 27:30].values
        drone = file.iloc[1:1000000, 30:33].values
        r1 = file.iloc[1:1000000, 33:34].values

        self.x_train = torch.tensor(ranges, dtype=torch.float32)
        self.y_train = torch.tensor(targets, dtype=torch.float32)
        self.a1 = torch.tensor(a1, dtype=torch.float32)
        self.a2 = torch.tensor(a2, dtype=torch.float32)
        self.a3 = torch.tensor(a3, dtype=torch.float32)
        self.a4 = torch.tensor(a4, dtype=torch.float32)
        self.a5 = torch.tensor(a5, dtype=torch.float32)
        self.drone = torch.tensor(drone, dtype=torch.float32)
        self.r1 = torch.tensor(r1, dtype=torch.float32)
    
    def __len__(self):
        return len(self.y_train)
    
    def __getitem__(self, idx):
        return self.x_train[idx], self.y_train[idx], self.a1[idx], self.a2[idx], self.a3[idx], self.a4[idx], self.a5[idx], self.drone[idx], self.r1[idx]

