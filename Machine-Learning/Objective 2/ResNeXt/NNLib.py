import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.autograd import Variable
from torch import Tensor
import math

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
        return self.res(x)+x

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
        self.Nodes = nn.ModuleList([ResBlock(layers, size, drop, nodes) for i in range(nodes)])
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

# Class that handles all neural network operations
class Localization(object):

    def __init__(self, network, optimizer, criterion1,
                 device=torch.device("cuda" if torch.cuda.is_available()
                                     else "cpu")):
        self.device = device
        self.Network = network.to(self.device)
        self.Optim = optimizer
        self.Loss = criterion1
        self.x = None
        self.y = None
        self.z = None

    def train(self, inputs, targets):
        inputs, t = inputs.to(self.device), targets.to(self.device)
        self.Optim.zero_grad()
        pos = self.Network(Variable(inputs))
        error = self.Loss(pos, Variable(t))
        error.backward()
        self.Optim.step()
        return pos, error.item()


    def test(self, inputs, targets):
        inputs, t = inputs.to(self.device), targets.to(self.device)
        with torch.no_grad():
            pos = self.Network(inputs)
            error1 = self.Loss(pos, t)
            error = error1
            return pos, error.item()

    def deploy(self, inputs, a1):
        inputs = inputs.to(self.device)
        with torch.no_grad():
            pos = self.Network(inputs)
            self.x = float(pos[0][0]) + float(a1[0][0])
            self.y = float(pos[0][1]) + float(a1[0][1])
            self.z = float(pos[0][2]) + float(a1[0][2])
        return pos


    @staticmethod
    def plot3ax(x, y, z, xnn, ynn, znn):
        fig = plt.figure(1)
        plt.clf()
        ax = plt.axes(projection='3d')
        ax.set_xlim3d(-10, 110)
        ax.set_ylim3d(-10, 110)
        ax.set_zlim3d(-10, 110)
        plt.title('Feed Forward Dynamic Anchor Test - Sliding Window with Noise')
        ax.plot3D(x, y, z, color='gray')
        ax.scatter(xnn, ynn, znn, color='green')
        plt.pause(0.001)
