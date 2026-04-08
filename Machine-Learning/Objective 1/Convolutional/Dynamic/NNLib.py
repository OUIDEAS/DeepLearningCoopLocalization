import torch
import matplotlib.pyplot as plt
from torch.autograd import Variable


# Class where neural network architecture is designed
class NeuralNetwork(torch.nn.Module):

    def __init__(self, n_inputs, hidden_size, num_layers):
        super(NeuralNetwork, self).__init__()
        self.n_inputs = n_inputs
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.fc1 = torch.nn.Linear(37, hidden_size)
        self.conv1 = torch.nn.Conv1d(in_channels=hidden_size, out_channels=1, kernel_size=1, stride=1, padding=1)
        self.conv2 = torch.nn.Conv1d(in_channels=1, out_channels=1, kernel_size=2, stride=1, padding=1)
        self.conv3 = torch.nn.Conv1d(in_channels=1, out_channels=1, kernel_size=2, stride=1, padding=1)
        self.conv4 = torch.nn.Conv1d(in_channels=1, out_channels=1, kernel_size=2, stride=1, padding=1)
        self.fc2 = torch.nn.Linear(6,3)

    def forward(self, x):
        x = torch.nn.functional.leaky_relu(self.fc1(x))
        x = torch.reshape(x, (1, self.hidden_size, 1))
        x = torch.nn.functional.leaky_relu(self.conv1(x))
        x = torch.nn.functional.leaky_relu(self.conv2(x))
        x = torch.nn.functional.leaky_relu(self.conv3(x))
        x = torch.nn.functional.leaky_relu(self.conv4(x))
        x = torch.reshape(x, (1, 6))
        x = self.fc2(x)
        return x


# Class that handles all neural network operations
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

    def train(self, inputs, targets, a1):
        inputs, t = inputs.to(self.device), targets.to(self.device)
        self.Optim.zero_grad()
        pos = self.Network(Variable(inputs))
        error = self.Loss(pos, Variable(t))
        error.backward()
        self.Optim.step()
        return pos, error.item()

    def test(self, inputs, targets, a1):
        t = torch.tensor([[targets[0][0], targets[0][1], targets[0][2]]])
        inputs, t = inputs.to(self.device), t.to(self.device)
        with torch.no_grad():
            pos = self.Network(Variable(inputs))
            error1 = self.Loss(pos, Variable(t))
            error = error1
            return pos, error.item()

    def deploy(self, inputs, a1):
        inputs = inputs.to(self.device)
        with torch.no_grad():
            pos = self.Network(inputs)
            self.x = float(pos[0][0]) + float(a1[0][0])
            self.y = float(pos[0][1]) + float(a1[0][1])
            self.z = float(pos[0][2]) + float(a1[0][2])
        return dir

    @staticmethod
    def prepare_data(a1, a2, a3, a4, a5, r):
        data = torch.tensor([[r[0], r[1], r[2], r[3], r[4], a2.x - a1.x, a2.y - a1.y, a2.z - a1.z, a3.x - a1.x, a3.y -
                              a1.y, a3.z - a1.z, a4.x - a1.x, a4.y - a4.y, a4.z - a1.z, a5.x - a1.x, a5.y - a1.y, a5.z -
                              a1.z]])
        return data

    @staticmethod
    def plot3ax(x, y, z, xnn, ynn, znn):
        fig = plt.figure(1)
        plt.clf()
        ax = plt.axes(projection='3d')
        ax.set_xlim3d(-32, 32)
        ax.set_ylim3d(-32, 32)
        ax.set_zlim3d(-32, 32)
        plt.title('Feed Forward Dynamic Anchor Test')
        ax.plot3D(x, y, z, color='gray')
        ax.scatter(xnn, ynn, znn, color='green')
        plt.pause(0.001)
