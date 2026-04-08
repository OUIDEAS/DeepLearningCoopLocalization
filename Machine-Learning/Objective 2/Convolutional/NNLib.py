import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.autograd import Variable

# Class where neural network architecture is designed
class NeuralNetwork(torch.nn.Module):
    # PRELU ACTIVATION FUNCTION WORKS
    def __init__(self, n_inputs, hidden_size, num_layers):
        super(NeuralNetwork, self).__init__()
        self.cer = False
        self.n_inputs = n_inputs
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.channels = 63
        self.fc1 = torch.nn.Linear(self.n_inputs, hidden_size)
        self.conv1 = torch.nn.Conv1d(in_channels=hidden_size, out_channels=self.channels, kernel_size=1, stride=1, padding=1)
        self.conv2 = torch.nn.Conv1d(in_channels=self.channels, out_channels=self.channels, kernel_size=2, stride=1, padding=1)
        self.conv3 = torch.nn.Conv1d(in_channels=self.channels, out_channels=self.channels, kernel_size=2, stride=1, padding=1)
        self.conv4 = torch.nn.Conv1d(in_channels=self.channels, out_channels=self.channels, kernel_size=2, stride=1, padding=1)
        self.fc2 = torch.nn.Linear(378,3)

    def forward(self, x):
        x = torch.nn.functional.leaky_relu(self.fc1(x))
        x = torch.reshape(x, (1, self.hidden_size, 1))
        x = torch.nn.functional.leaky_relu(self.conv1(x))
        x = torch.nn.functional.leaky_relu(self.conv2(x))
        x = torch.nn.functional.leaky_relu(self.conv3(x))
        x = torch.nn.functional.leaky_relu(self.conv4(x))
        x = torch.reshape(x, (1, -1))
        x = self.fc2(x)
        return x


# Made Separate class for this so the forward function did not have to deal with an if statement
# shared layers followed by an individual hidden layer for each readout
class Net3(torch.nn.Module):
    def __init__(self, n_inputs, hidden_size, num_layers):
        super(Net3, self).__init__()
        self.cer = True
        self.n_inputs = n_inputs
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        hidden_layers = [torch.nn.Linear(self.n_inputs, self.hidden_size)]
        # First hidden layer is defined separately because number of inputs to that layer is different than the rest
        for i in range(self.num_layers - 1):
            # Add a fully connected layer and then a Hyperbolic
            # tangent activation function for each layer that is added
            hidden_layers.append(torch.nn.ELU())
            hidden_layers.append(torch.nn.Linear(self.hidden_size, self.hidden_size))
        hidden_layers.append(torch.nn.ELU())
        self.Network = torch.nn.Sequential(*hidden_layers)
        self.fc1 = nn.Linear(self.hidden_size, 900)
        self.fc2 = nn.Linear(self.hidden_size, 900)
        self.fc3 = nn.Linear(self.hidden_size, 900)
        self.oute = nn.Linear(900, 1)
        self.outn = nn.Linear(900, 1)
        self.outu = nn.Linear(900, 1)


    def forward(self, x):
        x = self.Network(x)

        e = torch.nn.functional.leaky_relu(self.fc1(x))
        e = self.oute(e)

        n = torch.nn.functional.leaky_relu(self.fc2(x))
        n = self.outn(n)

        u = torch.nn.functional.leaky_relu(self.fc1(x))
        u = self.oute(u)
        return e, n, u


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
        if self.Network.cer:
            indicese = torch.zeros([targets.size(dim=-2), 1], dtype=torch.int64)
            indicesn = torch.ones([targets.size(dim=-2), 1], dtype=torch.int64)
            indicesu = 2*torch.ones([targets.size(dim=-2), 1], dtype=torch.int64)
            te = targets.gather(1,indicese)
            tn = targets.gather(1,indicesn)
            tu = targets.gather(1,indicesu)
            inputs, te, tn, tu = inputs.to(self.device), te.to(self.device), tn.to(self.device), tu.to(self.device)
            self.Optim.zero_grad()
            east, north, up = self.Network(Variable(inputs))
            errore = self.Loss(east, Variable(te))
            errorn = self.Loss(north, Variable(tn))
            erroru = self.Loss(up, Variable(tu))
            error = errore + errorn + erroru
            error.backward()
            self.Optim.step()
            pos = torch.tensor([[east[0][0], north[0][0], up[0][0]]])
            return pos, error.item()
        else:
            inputs, t = inputs.to(self.device), targets.to(self.device)
            self.Optim.zero_grad()
            pos = self.Network(Variable(inputs))
            error = self.Loss(pos, Variable(t))
            error.backward()
            self.Optim.step()
            return pos, error.item()


    def test(self, inputs, targets):
        if self.Network.cer:
            indicese = torch.zeros([targets.size(dim=-2), 1], dtype=torch.int64)
            indicesn = torch.ones([targets.size(dim=-2), 1], dtype=torch.int64)
            indicesu = 2*torch.ones([targets.size(dim=-2), 1], dtype=torch.int64)
            te = targets.gather(1,indicese)
            tn = targets.gather(1,indicesn)
            tu = targets.gather(1,indicesu)
            inputs, te, tn, tu = inputs.to(self.device), te.to(self.device), tn.to(self.device), tu.to(self.device)
            east, north, up = self.Network(inputs)
            errore = self.Loss(east, Variable(te))
            errorn = self.Loss(north, Variable(tn))
            erroru = self.Loss(up, Variable(tu))
            error = errore + errorn + erroru
            pos = torch.tensor([[east[0][0], north[0][0], up[0][0]]])
            return pos, error.item()
        else:
            inputs, t = inputs.to(self.device), targets.to(self.device)
            with torch.no_grad():
                pos = self.Network(inputs)
                error1 = self.Loss(pos, t)
                error = error1
                return pos, error.item()

    def deploy(self, inputs, a1):
        if self.Network.cer:
            inputs = inputs.to(self.device)
            east, north, up = self.Network(Variable(inputs))
            self.x = float(east[0][0]) + float(a1[0][0])
            self.y = float(north[0][0]) + float(a1[0][1])
            self.z = float(up[0][0]) + float(a1[0][2])
            pos = torch.tensor([[east[0][0], north[0][0], up[0][0]]])
            return pos
        else:
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
        plt.title('Feed Forward Dynamic Anchor Test - Sliding Window with Noise')
        ax.plot3D(x, y, z, color='gray')
        ax.scatter(xnn, ynn, znn, color='green')
        plt.pause(0.001)
