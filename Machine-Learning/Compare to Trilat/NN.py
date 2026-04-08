import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.autograd import Variable
from torch import Tensor
import math

class Node(nn.Module):
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
        return self.res(x)

    def rg_init(self, m):
        if isinstance(m, torch.nn.Linear):
            (fan_in, fan_out) = nn.init._calculate_fan_in_and_fan_out(m.weight)
            c = 1
            var = c/(self.layers*fan_in)
            nn.init.normal_(m.weight, 0, math.sqrt(var))
            m.bias.data.zero_()

class ResNeXt(nn.Module):
    def __init__(self, size:int, res_layers:int, residuals:int, drop=0.1):
        super().__init__()
        self.in_layer = nn.Linear(222, size)
        self.activation = nn.PReLU(num_parameters=size)
        self.ResidualNetwork = nn.ModuleList([ResBlock(res_layers, size, drop, residuals) for i in range(residuals)])
        self.out = nn.Linear(size, 3)
    def forward(self, x):
        x = self.in_layer(x)
        x = self.activation(x)
        resids = []
        for layer in self.ResidualNetwork:
            resids.append(layer(x))

        out=torch.zeros(x.size()).to(torch.device('cuda'))
        for r in resids:
            out = out + r
        x = self.out(x)
        return x

class NodalLayer(nn.Module):
    def __init__(self, nodes, size, layers, drop):
        super().__init__()
        self.Nodes = nn.ModuleList([Node(layers, size, drop, nodes) for i in range(nodes)])
    def forward(self, x):
        resids=[]
        for layer in self.Nodes:
            resids.append(layer(x))

        out=torch.zeros(x.size()).to(torch.device('cuda'))
        for r in resids:
            out = out + r
        return out+x

class RGNet(nn.Module):
    def __init__(self, size:int, res_layers:int, nodes:int, nodal_layers:int, drop=0.1):
        super().__init__()
        self.in_layer = nn.Linear(222, size)
        self.activation = nn.PReLU(num_parameters=size)
        self.Network = nn.ModuleList([NodalLayer(nodes, size, nodal_layers, drop) for i in range(nodal_layers)])
        self.out = nn.Linear(size, 3)

    def forward(self, x):
        x = self.in_layer(x)
        x = self.activation(x)
        resids = []
        # First Cluster of SubNetworks
        for layer in self.Network:
            x = layer(x)

        x = self.out(x)
        return x

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
    def __init__(self, size:int, res_layers:int, residuals:int, drop=0.1):
        super().__init__()
        self.in_layer = nn.Linear(222, size)
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
