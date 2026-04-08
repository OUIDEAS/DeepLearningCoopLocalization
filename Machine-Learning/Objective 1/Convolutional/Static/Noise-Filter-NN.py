import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import torch.optim as optim
from torch.serialization import load
from NNLib import *
import matplotlib.pyplot as plt
import os
import time
from DLoaderRI import *
from torch.utils.data import DataLoader
import sys


def main(load):

    PATH = "Convolutional-Position-NN.pt"

    os.system('clear')
    print("Loading Data...")
    start = time.ctime()

    # Load training, validation, and testing data
    train = TrainLoader('training_data.csv')
    test = TrainLoader('test_data.csv')
    val = TrainLoader('validation_data.csv')
    train_loader = DataLoader(train, batch_size=1, shuffle=True)
    test_loader = DataLoader(test, batch_size=1, shuffle=False)
    validation_loader = DataLoader(val, batch_size=1, shuffle=True)

    # 300 for overnight
    epochs = 1

    # 0.000005 for overnight
    learning_rate = 0.000005
    if load:
        neural_net = torch.load(PATH)
    else:
        neural_net = NeuralNetwork(300)

    System = Localization(neural_net, optim.Adam(neural_net.parameters(), lr=learning_rate), nn.L1Loss(), nn.L1Loss())

    acc = []
    averr = []
    N = 100
    i = 1

    print(start)
    for e in range(epochs):
        for data, target, a1, a2, a3, a4, a5, drone, r1 in train_loader:
            # Display to user how far along the simulation is
            i_with_commas = "{:,}".format(i)
            sys.stdout.write("\rTraining iteration: {0}".format(i_with_commas))
            sys.stdout.flush()

            # inputs, targets, home = System.prepare_load(data, a1, a2, a3, a4, a5, drone)

            # Push data through the neural network, calculate the loss and perform backpropagation
            dir, loss = System.train(data, drone)

            # Record the loss for plots
            acc.append(loss)
            if len(acc)>N:
                average_error = sum(acc[-N:])/N
            else:
                average_error = sum(acc)/len(acc)
            averr.append(average_error)
            i = i + 1

    os.system('clear')

    end = time.ctime()

    print("Start:       ", start)
    print("End:         ", end)

    plt.figure()
    plt.title('Average Training Error')
    plt.plot(averr)
    plt.show()

    acc = []
    averr = []
    
    with torch.no_grad():
        for data, target, a1, a2, a3, a4, a5, drone, r1 in validation_loader:
            # inputs, targets, home = System.prepare_load(data, a1, a2, a3, a4, a5, drone)
            dir, loss = System.test(data, drone)
            acc.append(loss)
            if len(acc)>N:
                average_error = sum(acc[-N:])/N
            else:
                average_error = sum(acc)/len(acc)
            averr.append(average_error)
            i = i + 1

    plt.figure()
    plt.title('Average Validation Error')
    plt.plot(acc)
    plt.show()
    
    acc, averr    = [], []
    x, y, z       = [], [], []
    xnn, ynn, znn = [], [], []
    startup = True
    
    with torch.no_grad():
        for data, target, a1, a2, a3, a4, a5, drone, r1 in test_loader:
            # Switch to one trilateration iteration to get an estimate of position
            # if startup:
            #    r = torch.tensor([[0]]).to(System.device)
            #    dir = torch.tensor([[0, 0, 0]]).to(System.device)
            #    startup = False
            # data = data.to(Sysqtem.device)
            # data = torch.cat([data, r, dir], 1)

            # inputs, targets, home = System.prepare_load(data, a1, a2, a3, a4, a5, drone)
            dir = System.deploy(data) # , home)
            xnn.append(System.x)
            ynn.append(System.y)
            znn.append(System.z)
            x.append(float(drone[0][0]))
            y.append(float(drone[0][1]))
            z.append(float(drone[0][2]))
            System.plot3ax(x, y, z, xnn, ynn, znn)

    plt.show()
    torch.save(System.Network, PATH)

    os.system('clear')
    print("Start:       ", start)
    print("End:         ", end)

if __name__ == "__main__":
    main(load=False)