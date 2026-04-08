import torch
import torch.nn as nn
import math
import pandas as pd
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

class ResBlock(nn.Module):
    def __init__(self, res_layers, size, drop, residuals):
        super().__init__()
        self.layers = res_layers*residuals
        rb = []
        for i in range(res_layers):
            rb.append(nn.Linear(size, size))
            rb.append(nn.Dropout(0.1))
            rb.append(nn.PReLU(num_parameters=size))
        self.res = nn.Sequential(*rb)
        self.apply(self.rg_init)

    def forward(self, x):
        return x + self.res(x)

    def rg_init(self, m):
        if isinstance(m, torch.nn.Linear):
            (fan_in, fan_out) = nn.init._calculate_fan_in_and_fan_out(m.weight)
            c = 1
            var = c/(self.layers*fan_in)
            nn.init.normal_(m.weight, 0, math.sqrt(var))
            m.bias.data.zero_()

class ResNet(nn.Module):
    
    def __init__(self, n_in:int, size:int, res_layers:int, residuals:int, drop=0.1):
        super().__init__()
        self.in_layer = nn.Linear(n_in, size)
        self.activation = nn.PReLU(num_parameters=size)
        self.ResidualNetwork = nn.ModuleList([ResBlock(res_layers, size, drop, residuals) for i in range(residuals)])
        self.out = nn.Linear(size, 3)

    def forward(self, x):
        x = self.in_layer(x)
        x = self.activation(x)
        for layer in self.ResidualNetwork:
            x = layer(x)
        x = self.out(x)
        return x
    

