import torch
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.autograd import Variable
import math
import os

os.system('clear')

#Class where neural network architecture is designed
class NeuralNetwork(torch.nn.Module):
    
    def __init__(self, n_inputs, hidden_size, num_layers):
        super(NeuralNetwork, self).__init__()
        self.n_inputs = n_inputs
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        hidden_layers = [torch.nn.Linear(self.n_inputs, self.hidden_size)]
        # First hidden layer is defined separately because number of inputs to that layer is different than the rest
        for i in range(self.num_layers - 1):
            # Add a fully connected layer and then a Hyperbolic
            # tangent activation function for each layer that is added
            hidden_layers.append(torch.nn.ReLU())
            hidden_layers.append(torch.nn.Linear(self.hidden_size, self.hidden_size))
            
        hidden_layers.append(torch.nn.ReLU())
        hidden_layers.append(torch.nn.Linear(self.hidden_size, 3))
        self.Network = torch.nn.Sequential(*hidden_layers)

    def forward(self, x):
        x = self.Network(x)
        return x

class dummydrone():
    def __init__(self):
        self.x = 0
        self.y = 0
        self.z = 0

# Class that handles all neural network operations
class Localization(object):
    def __init__(self, nn, optimizer, criterion1, criterion2, device = torch.device("cuda" if torch.cuda.is_available() else "cpu")):

        # Initialize the class to store the neural network, the optimizer, the loss functions
        # and the device to store all tensors on
        self.reference_drone = 1
        self.device = device
        self.Network = nn.to(self.device)
        self.Optim = optimizer
        self.Loss = criterion1
        self.rLoss = criterion2

    def position_error(self, x, y, z, r, a1, d):

        # Take in the outputs of the neural network and determine the error of the predicted
        # location vs the actual location
        xd = float(x) * float(r) + float(a1[0][0]) - float(d[0][0])
        yd = float(y) * float(r) + float(a1[0][1]) - float(d[0][1])
        zd = float(z) * float(r) + float(a1[0][2]) - float(d[0][2])
        return math.sqrt(xd**2 + yd**2 + zd**2)

    def train(self, inputs, targets): # , r1):

        # Train the neural network
        # inputs = torch.cat([inputs, r1], 1)
        inputs, targets = inputs.to(self.device), targets.to(self.device)
        self.Optim.zero_grad()
        dir = self.Network(Variable(inputs))
        error = self.Loss(dir, Variable(targets))
        # error2 = self.rLoss(r, Variable(r1[0][0]).view(1, 1))
        # error = error1 + error2
        error.backward()
        self.Optim.step()
        return dir, error.item()

    def test(self, inputs, targets):
        inputs, targets = inputs.to(self.device), targets.to(self.device)
        with torch.no_grad():
            dir = self.Network(Variable(inputs))
            error1 = self.Loss(dir, Variable(targets))
            # error2 = self.rLoss(r, Variable(r1[0][0]).view(1,1))
            error = error1 #+ error2
            # r = r.to(self.device)
            return dir, error.item()
        
    def deploy(self, inputs):

        # Deploy the neural network to determine the position of the drone
        # inputs = inputs.to(self.device)
        #a1 = a1.to(self.device)
        # r1 = r1.to(self.device)
        # inputs = torch.cat([inputs, r1], 1)
        inputs = inputs.to(self.device)
        with torch.no_grad():
            dir = self.Network(inputs)
            self.x = float(dir[0][0]) # + float(a1[0])
            self.y = float(dir[0][1]) # + float(a1[1])
            self.z = float(dir[0][2]) # + float(a1[2])
        return dir

    def prepare_live(self, drone, a1, a2, a3, a4, a5):

        # Prepare the data in real time. Drone, a1, a2, a3, a4, a5 are classes. Drone has 5 attributes of r_n
        # Which represent the ranges to each anchor drone
        # a1, a2, a3, a4, a5 have an x, y, and z attributes
        self.reference_drone = torch.argmax(torch.tensor([drone.r1, drone.r2, drone.r3, drone.r4, drone.r5])) + 1

        if self.reference_drone == 1:
            data = torch.tensor([[drone.r1, drone.r2, drone.r3, drone.r4, drone.r5, a2.x - a1.x, a2.y - a1.y,
                                  a2.z - a1.z, a3.x - a1.x, a3.y - a1.y, a3.z - a1.z, a4.x - a1.x, a4.y - a1.y,
                                  a4.z - a1.z, a5.x - a1.x, a5.y - a1.y, a5.z - a1.z]])
            home = torch.tensor([a1.x, a1.y, a1.z])


        elif self.reference_drone == 2:
            data = torch.tensor([[drone.r2, drone.r1, drone.r3, drone.r4, drone.r5, a1.x - a2.x, a1.y - a2.y,
                                  a1.z - a2.z, a3.x - a2.x, a3.y - a2.y, a3.z - a2.z, a4.x - a2.x, a4.y - a2.y,
                                  a4.z - a2.z, a5.x - a2.x, a5.y - a2.y, a5.z - a2.z]])
            home = torch.tensor([a2.x, a2.y, a2.z])

        elif self.reference_drone == 3:
            data = torch.tensor([[drone.r3, drone.r1, drone.r2, drone.r4, drone.r5, a1.x - a3.x, a1.y - a3.y,
                                  a1.z - a3.z, a2.x - a3.x, a2.y - a3.y, a2.z - a3.z, a4.x - a3.x, a4.y - a3.y,
                                  a4.z - a3.z, a5.x - a3.x, a5.y - a3.y, a5.z - a3.z]])
            home = torch.tensor([a3.x, a3.y, a3.z])

        elif self.reference_drone == 4:
            data = torch.tensor([[drone.r4, drone.r1, drone.r2, drone.r3, drone.r5, a1.x - a4.x, a1.y - a4.y,
                                  a1.z - a4.z, a2.x - a4.x, a2.y - a4.y, a2.z - a4.z, a3.x - a4.x, a3.y - a4.y,
                                  a3.z - a4.z, a5.x - a4.x, a5.y - a4.y, a5.z - a4.z]])
            home = torch.tensor([a4.x, a4.y, a4.z])

        elif self.reference_drone == 5:
            data = torch.tensor([[drone.r5, drone.r1, drone.r2, drone.r3, drone.r4, a1.x - a5.x, a1.y - a5.y,
                                  a1.z - a5.z, a2.x - a5.x, a2.y - a5.y, a2.z - a5.z, a3.x - a5.x, a3.y - a5.y,
                                  a3.z - a5.z, a4.x - a5.x, a4.y - a5.y, a4.z - a5.z]])
            home = torch.tensor([a5.x, a5.y, a5.z])

        # Scale the input data before sending it back
        data = data/torch.max(torch.abs(data))

        return data, home

    def prepare_load(self, data, anchor1, anchor2, anchor3, anchor4, anchor5, d):

        a1, a2, a3, a4, a5 = dummydrone(), dummydrone(), dummydrone(), dummydrone(), dummydrone()
        a1.x, a1.y, a1.z = anchor1[0][0], anchor1[0][1], anchor1[0][2]
        a2.x, a2.y, a2.z = anchor2[0][0], anchor2[0][1], anchor2[0][2],
        a3.x, a3.y, a3.z = anchor3[0][0], anchor3[0][1], anchor3[0][2],
        a4.x, a4.y, a4.z = anchor4[0][0], anchor4[0][1], anchor4[0][2],
        a5.x, a5.y, a5.z = anchor5[0][0], anchor5[0][1], anchor5[0][2],
        ranges = torch.tensor([data[0][0], data[0][1], data[0][2], data[0][3], data[0][4]])
        self.reference_drone = torch.argmax(ranges) + 1
        if self.reference_drone == 1:
            data = torch.tensor([[data[0][0], data[0][1], data[0][2], data[0][3], data[0][4], a2.x - a1.x, a2.y - a1.y,
                                  a2.z - a1.z, a3.x - a1.x, a3.y - a1.y, a3.z - a1.z, a4.x - a1.x, a4.y - a1.y,
                                  a4.z - a1.z, a5.x - a1.x, a5.y - a1.y, a5.z - a1.z]])
            home = torch.tensor([a1.x, a1.y, a1.z])
            target = torch.tensor([[d[0][0] - a1.x, d[0][1] - a1.y, d[0][2] - a1.z]])

        elif self.reference_drone == 2:
            data = torch.tensor([[data[0][1], data[0][0], data[0][2], data[0][3], data[0][4], a1.x - a2.x, a1.y - a2.y,
                                  a1.z - a2.z, a3.x - a2.x, a3.y - a2.y, a3.z - a2.z, a4.x - a2.x, a4.y - a2.y,
                                  a4.z - a2.z, a5.x - a2.x, a5.y - a2.y, a5.z - a2.z]])
            home = torch.tensor([a2.x, a2.y, a2.z])
            target = torch.tensor([[d[0][0] - a2.x, d[0][1] - a2.y, d[0][2] - a2.z]])

        elif self.reference_drone == 3:
            data = torch.tensor([[data[0][2], data[0][0], data[0][1], data[0][3], data[0][4], a1.x - a3.x, a1.y - a3.y,
                                  a1.z - a3.z, a2.x - a3.x, a2.y - a3.y, a2.z - a3.z, a4.x - a3.x, a4.y - a3.y,
                                  a4.z - a3.z, a5.x - a3.x, a5.y - a3.y, a5.z - a3.z]])
            home = torch.tensor([a3.x, a3.y, a3.z])
            target = torch.tensor([[d[0][0] - a3.x, d[0][1] - a3.y, d[0][2] - a3.z]])

        elif self.reference_drone == 4:
            data = torch.tensor([[data[0][3], data[0][0], data[0][1], data[0][2], data[0][4], a1.x - a4.x, a1.y - a4.y,
                                  a1.z - a4.z, a2.x - a4.x, a2.y - a4.y, a2.z - a4.z, a3.x - a4.x, a3.y - a4.y,
                                  a3.z - a4.z, a5.x - a4.x, a5.y - a4.y, a5.z - a4.z]])
            home = torch.tensor([a4.x, a4.y, a4.z])
            target = torch.tensor([[d[0][0] - a4.x, d[0][1] - a4.y, data[0][2] - a4.z]])

        elif self.reference_drone == 5:
            data = torch.tensor([[data[0][4], data[0][0], data[0][1], data[0][2], data[0][3], a1.x - a5.x, a1.y - a5.y,
                                  a1.z - a5.z, a2.x - a5.x, a2.y - a5.y, a2.z - a5.z, a3.x - a5.x, a3.y - a5.y,
                                  a3.z - a5.z, a4.x - a5.x, a4.y - a5.y, a4.z - a5.z]])
            home = torch.tensor([a5.x, a5.y, a5.z])
            target = torch.tensor([[d[0][0] - a5.x, d[0][1] - a5.y, d[0][2] - a5.z]])

        mag = math.sqrt(float(target[0][0]) ** 2 + float(target[0][1]) ** 2 + float(target[0][2]) ** 2)
        target = target / mag
        data = data/torch.max(torch.abs(data))

        return data, target, home

    def plot3ax(self, x, y, z, xnn, ynn, znn):

        # Animate the test results
        fig = plt.figure(1)
        plt.clf()
        ax = plt.axes(projection='3d')
        ax.set_xlim3d(-30, 30)
        ax.set_ylim3d(-30, 30)
        ax.set_zlim3d(-30, 30)
        plt.title('Feed Forward Static Anchor Test')
        ax.plot3D(x, y, z, color='gray')
        ax.scatter(xnn, ynn, znn, color='green')
        # pause a bit so that plots are updated
        plt.pause(0.01)