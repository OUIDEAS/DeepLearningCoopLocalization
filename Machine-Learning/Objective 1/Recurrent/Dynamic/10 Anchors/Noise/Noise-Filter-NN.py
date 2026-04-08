import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from NNLib import *
import matplotlib.pyplot as plt
import os
import time
from DLoaderRI import *
from torch.utils.data import DataLoader
import sys


def main(load):
    PATH = "RNN10anc.pt"
    os.system('clear')
    print("Loading Data...")
    start = time.ctime()
    train = TrainLoader('training_data.csv')
    test = TrainLoader('test_data.csv')
    val = TrainLoader('validation_data.csv')
    train_loader = DataLoader(train, batch_size=1, shuffle=False)
    test_loader = DataLoader(test, batch_size=1, shuffle=False)
    validation_loader = DataLoader(val, batch_size=1, shuffle=True)
    epochs = 6
    learning_rate = 0.001


    layers = 4
    size = 800


    if load:
        neural_net = torch.load(PATH)
    else:
        neural_net = NeuralNet(hidden_size=size, rnn_layers=layers)

    System = Localization(neural_net, optim.Adam(neural_net.parameters(), lr=learning_rate), nn.L1Loss(), nn.L1Loss())
    acc = []
    averr = []
    N = 100
    i = 1
    os.system('clear')

    print(start, "\n")
    print(System.Network, "\n")
    hidden = torch.zeros(layers, 1, size).to(device="cuda")
    for e in range(epochs):
        i = 1
        hidden = torch.zeros(layers, 1, size).to(device="cuda")
        for data, target, a1 in train_loader:
            if i % 100:
                hidden = torch.zeros(layers, 1, size).to(device="cuda")
            i_with_commas = "{:,}".format(i)
            sys.stdout.write("\rTraining iteration: {0}     Epoch: {1}/{2}".format(i_with_commas, e+1, epochs))
            sys.stdout.flush()
            pos, loss, hidden = System.train(data, target, hidden)
            acc.append(loss)
            if len(acc) > N:
                average_error = sum(acc[-N:]) / N
            else:
                average_error = sum(acc) / len(acc)
            averr.append(average_error)
            i = i + 1

    os.system('clear')
    end = time.ctime()
    print("Start:       ", start)
    print("End:         ", end)
    plt.figure()
    plt.title('Training Mean Squared Error')
    plt.plot(averr)
    plt.show()
    acc = []
    averr = []
    # Validate the network
    with torch.no_grad():
        hidden = torch.zeros(layers, 1, size).to(device="cuda")
        for data, target, a1 in validation_loader:
            data = data/torch.max(torch.abs(data))
            pos, loss, hidden = System.test(data, target, hidden)
            acc.append(loss)
            if len(acc) > N:
                average_error = sum(acc[-N:]) / N
            else:
                average_error = sum(acc) / len(acc)
            averr.append(average_error)
            i = i + 1

    plt.figure()
    plt.title('Average Validation Error')
    plt.plot(acc)
    plt.show()
    x, y, z = [], [], []
    xnn, ynn, znn = [], [], []
    with torch.no_grad():
        hidden = torch.zeros(layers, 1, size).to(device="cuda")
        for data, target, a1 in test_loader:
            data = data/torch.max(torch.abs(data))
            pos, hidden = System.deploy(data, hidden, a1)
            xnn.append(System.x)
            ynn.append(System.y)
            znn.append(System.z)
            x.append(float(target[0][0]) + float(a1[0][0]))
            y.append(float(target[0][1]) + float(a1[0][1]))
            z.append(float(target[0][2]) + float(a1[0][2]))
            System.plot3ax(x, y, z, xnn, ynn, znn)

    plt.show()
    torch.save(System.Network, PATH)


if __name__ == "__main__":
    main(load=True)
