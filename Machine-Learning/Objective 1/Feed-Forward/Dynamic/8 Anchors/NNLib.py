import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.autograd import Variable

# Initialize weights according to https://arxiv.org/abs/1502.01852
# Normal Distribution:
def initialize_weights_kn(m):
    if isinstance(m, torch.nn.Linear):
        nn.init.kaiming_normal_(m.weight,a = 0.2, mode='fan_in', nonlinearity='leaky_relu')#, gain=nn.init.calculate_gain('leaky_relu',0.2))

# Uniform distribution:
def initialize_weights_ku(m):
    if isinstance(m, torch.nn.Linear):
        nn.init.kaiming_uniform_(m.weight,a = 0.2, mode='fan_in', nonlinearity='leaky_relu')#, gain=nn.init.calculate_gain('leaky_relu',0.2))

# Initialize weights according to https://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf
# Normal Distribution:
def initialize_weights_xn(m):
    if isinstance(m, torch.nn.Linear):
        nn.init.xavier_normal_(m.weight, gain=nn.init.calculate_gain('leaky_relu',0.2))

# Uniform distribution
def initialize_weights_xu(m):
    if isinstance(m, torch.nn.Linear):
        nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('leaky_relu',0.2))

# Initialize weights according to https://openreview.net/forum?id=_wzZwKpTDF_9C
def initialize_weights_orth_(m):
    if isinstance(m, torch.nn.Linear):
        nn.init.orthogonal(m.weight)

class FeedForward(torch.nn.Module):
    def __init__(self, n_inputs: int, hidden_size: int, num_layers: int, drop: float):
        super(FeedForward, self).__init__()
        self.n_inputs = n_inputs
        self.hiddensize = hidden_size
        sl = [torch.nn.Linear(self.n_inputs, self.hiddensize),
              torch.nn.Dropout(drop)]
        for i in range(num_layers-1):
            sl.append(torch.nn.PReLU(num_parameters=self.hiddensize))
            sl.append(torch.nn.Linear(self.hiddensize, self.hiddensize))
            sl.append(torch.nn.Dropout(drop))
        sl.append(torch.nn.PReLU(num_parameters=self.hiddensize))
        sl.append(torch.nn.Linear(self.hiddensize, 3))
        self.network = torch.nn.Sequential(*sl)
        self.apply(initialize_weights_kn)

    def forward(self, x):
        return self.network(x)

# Class that handles all neural network operations
class Localization(object):

    def __init__(self, network, optimizer, criterion,
                 device=torch.device("cuda" if torch.cuda.is_available()
                                     else "cpu")):
        self.device = device
        self.Network = network.to(self.device)
        self.Optim = optimizer
        self.Loss = criterion
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
        ax.set_xlim3d(-10, 285)
        ax.set_ylim3d(-10, 285)
        ax.set_zlim3d(-10, 285)
        ax.set_xlabel('East [m]')
        ax.set_ylabel('North [m]')
        ax.set_zlabel('Up [m]')
        plt.title('Feed Forward Dynamic Anchor Test - Sliding Window with Noise')
        ax.plot3D(x, y, z, color='gray')
        ax.plot(xnn, ynn, znn, color='green')
        plt.pause(0.001)
