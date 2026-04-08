import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler


class TrainLoader(Dataset):
    def __init__(self, filename):
        file = pd.read_csv(filename)
        inputs = file.iloc[0:1000000, 0:204].values
        targets = file.iloc[0:1000000, 204:207].values
        a1 = file.iloc[0:1000000, 207:210].values
        self.inputs = torch.tensor(inputs, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32)
        self.a1 = torch.tensor(a1, dtype=torch.float32)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx], self.a1[idx]
