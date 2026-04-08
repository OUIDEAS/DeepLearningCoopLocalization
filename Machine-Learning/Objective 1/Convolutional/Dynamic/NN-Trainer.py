import torch
import torch.nn as nn
import torch.optim as optim
from NNLib import *
import matplotlib.pyplot as plt
import os
import time
from DLoaderRI import *
from torch.utils.data import DataLoader
import sys

def main(load):
    PATH = "Feed-Forward-Position-NN.pt"

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

    epochs = 10

    learning_rate = 0.001
    if load:
        neural_net = torch.load(PATH)
    else:
        neural_net = NeuralNetwork(n_inputs=17, hidden_size=95, num_layers=4)

    System = Localization(neural_net, optim.Adam(neural_net.parameters(), lr=learning_rate), nn.L1Loss(), nn.L1Loss())

    acc = []
    averr = []
    N = 100
    i = 1

    os.system('clear')
    print(start, "\n")

    # Stage 1 training. Noise free to learn the model
    # Train the network on noise free data
    print(System.Network, "\n")
    print('Noise-free training. ')
    for e in range(epochs):
        for data, target, a1 in train_loader:
            # Display to user how far along the simulation is
            i_with_commas = "{:,}".format(i)
            sys.stdout.write("\rTraining iteration: {0}".format(i_with_commas))
            sys.stdout.flush()
            # Push data through the neural network, calculate the loss and perform backpropagation
            pos, loss = System.train(data, target, a1)

            # Record the loss for plots
            acc.append(loss)
            if len(acc) > N:
                average_error = sum(acc[-N:]) / N
            else:
                average_error = sum(acc) / len(acc)
            averr.append(average_error)
            i = i + 1

    os.system('clear')
    print(start)
    print('Noise added. ')
    # Stage 2 training. Gaussian noise added to filter the noise
    # for e in range(epochs):
    # for data, target, a1 in train_loader:
    # Display to user how far along the simulation is
    #    i_with_commas = "{:,}".format(i)
    #    sys.stdout.write("\rTraining iteration: {0}".format(i_with_commas))
    #    sys.stdout.flush()
    # Push data through the neural network, calculate the loss and perform backpropagation
    #    pos, loss = System.train(data, target, a1)
    # Record the loss for plots
    #    acc.append(loss)
    #    if len(acc) > N:
    #        average_error = sum(acc[-N:])/N
    #    else:
    #        average_error = sum(acc)/len(acc)
    #    averr.append(average_error)
    #    i = i + 1

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
        for data, target, a1 in validation_loader:
            # inputs, targets, home = System.prepare_load(data, a1, a2, a3, a4, a5, drone)
            pos, loss = System.test(data, target, a1)
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
    x, y, z, xnn, ynn, znn = [], [], [], [], [], []

    # Test the network to generate a plot to illustrate the localization
    with torch.no_grad():
        for data, target, a1 in test_loader:
            pos = System.deploy(data, a1)
            xnn.append(System.x)
            ynn.append(System.y)
            znn.append(System.z)
            x.append(float(target[0][0]))
            y.append(float(target[0][1]))
            z.append(float(target[0][2]))
            System.plot3ax(x, y, z, xnn, ynn, znn)

    plt.show()
    torch.save(System.Network, PATH)

    os.system('clear')
    print("Start:       ", start)
    print("End:         ", end)


if __name__ == "__main__":
    main(load=False)
