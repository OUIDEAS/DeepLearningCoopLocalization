import torch
import matplotlib.pyplot as plt
from torch.autograd import Variable


class NeuralNet(torch.nn.Module):

    def __init__(self, hidden_size, rnn_layers):
        super(NeuralNet,self).__init__()
        self.hidden_size = hidden_size
        self.rnn_layers = rnn_layers
        hl = [torch.nn.Linear(37, hidden_size), torch.nn.LeakyReLU(negative_slope=0.1)]#,
              #torch.nn.Linear(hidden_size, hidden_size), torch.nn.LeakyReLU(negative_slope=0.1),
              #torch.nn.Linear(hidden_size, hidden_size), torch.nn.LeakyReLU(negative_slope=0.1)]
        self.rnn = torch.nn.RNN(hidden_size, hidden_size, self.rnn_layers, nonlinearity='relu')
        hl2 = [torch.nn.Linear(hidden_size, hidden_size), torch.nn.LeakyReLU(negative_slope=0.1),
               #torch.nn.Linear(hidden_size, hidden_size), torch.nn.LeakyReLU(negative_slope=0.1),
               torch.nn.Linear(hidden_size,3)]
        self.in_layers = torch.nn.Sequential(*hl) 
        self.out_layers = torch.nn.Sequential(*hl2)  

    def forward(self, x, hidden):
        x = self.in_layers(x)
        x = x.reshape(1, 1, self.hidden_size)
        out, hn = self.rnn(x, hidden.detach())
        out = self.out_layers(out[:, -1, :])
        x = out.reshape(1, 3)
        return x, hn


class Localization(object):

    def __init__(self, network, optimizer, criterion1, criterion2,
                 device=torch.device("cuda" if torch.cuda.is_available()
                                     else "cpu")):
        self.device = device
        self.Network = network.to(self.device)
        self.Optim = optimizer
        self.Loss = criterion1
        self.rLoss = criterion2
        self.x = None
        self.y = None
        self.z = None

    def train(self, inputs, targets, hidden):
        inputs, targets, hidden = inputs.to(self.device), targets.to(self.device), hidden.to(self.device)
        self.Optim.zero_grad()
        pos, hidden = self.Network(inputs, hidden)
        error = self.Loss(pos, targets)
        error.backward()
        self.Optim.step()
        return pos, error.item(), hidden

    def test(self, inputs, targets, hidden):
        inputs, targets, hidden = inputs.to(self.device), targets.to(self.device), hidden.to(self.device)
        with torch.no_grad():
            pos, hidden = self.Network(inputs, hidden)
            error = self.Loss(pos, targets)
            return pos, error.item(), hidden

    def deploy(self, inputs, hidden, a1):
        inputs, hidden = inputs.to(self.device), hidden.to(self.device)
        with torch.no_grad():
            pos, hidden = self.Network(inputs, hidden)
            self.x = float(pos[0][0]) + float(a1[0][0])
            self.y = float(pos[0][1]) + float(a1[0][1])
            self.z = float(pos[0][2]) + float(a1[0][2])
        return pos, hidden

    @staticmethod
    def plot3ax(x, y, z, xnn, ynn, znn):
        fig = plt.figure(1)
        plt.clf()
        ax = plt.axes(projection='3d')
        ax.set_xlim3d(-110, 110)
        ax.set_ylim3d(-110, 110)
        ax.set_zlim3d(-110, 110)
        plt.title('RNN Dynamic Anchor Test')
        ax.plot3D(x, y, z, color='gray')
        ax.scatter(xnn, ynn, znn, color='green')
        plt.pause(0.001)
