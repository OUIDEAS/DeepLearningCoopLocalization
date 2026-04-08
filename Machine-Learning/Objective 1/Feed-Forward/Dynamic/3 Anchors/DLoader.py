import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler


class TrainLoader(Dataset):
    def __init__(self, filename):
        num_anc = 3
        n_in = num_anc*6 + 3*(num_anc-1)
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
