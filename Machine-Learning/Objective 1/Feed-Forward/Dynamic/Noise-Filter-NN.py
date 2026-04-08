import torch
import torch.nn as nn
import torch.optim as optim
from NNLib import *
import matplotlib.pyplot as plt
import os
import time
from DLoader import TrainLoader
from torch.utils.data import DataLoader
import sys
import pickle
# 100% Necessary
from dadjokes import Dadjoke
import math



def main(load, line):

    PATH = "FilterLocalizationNetwork-compressed.pt"
    print('Dad Joke:')
    os.system('clear')
    dadjoke = Dadjoke()
    print(dadjoke.joke, '\n')
    print("Loading Data...")
    start = time.ctime()
    train = TrainLoader('Training_data.csv')
    test = TrainLoader('Test_data.csv')
    val = TrainLoader('Validation_data.csv')
    train_loader = DataLoader(train, batch_size=181, shuffle=False)
    test_loader = DataLoader(test, batch_size=1, shuffle=False)
    validation_loader = DataLoader(val, batch_size=1000, shuffle=True)
    epochs = 232
    learning_rate = 1.439e-4

    if load:
        neural_net = torch.load(PATH)
    else:
        # neural_net = ResNet(size=910, res_layers=3, residuals=3, drop=0.1)
        neural_net = FeedForward(n_inputs = 87, hidden_size = 910, num_layers = 5, drop = 8.652e-4)
        neural_net.apply(initialize_weights_ku)

    System = Localization(neural_net, optim.RAdam(neural_net.parameters(), lr=learning_rate), nn.L1Loss())
    acc = []
    averr = []
    N = 100
    os.system('clear')
    print(start)
    print(System.Network)
    print('Ultra-wideband ranging simulated using Cramer-Rao Lower Bound.')
    training = []
    valid = []
    for e in range(epochs):
        acct = []
        sys.stdout.write("\rTraining epoch: {0}/{1}".format(e,epochs))
        sys.stdout.flush()
        for data, target, a1 in train_loader:
            # Push data through the neural network, calculate the loss and perform backpropagation
            pos, loss = System.train(data, target)
            # Record the loss for plots
            acc.append(loss)
            acct.append(loss)
            if len(acc) > N:
                average_error = sum(acc[-N:]) / N
            else:
                average_error = sum(acc) / len(acc)
            averr.append(average_error)
        training.append(sum(acct)/len(acct))
        accv = []
        with torch.no_grad():
            for data, target, a1 in validation_loader:
                # inputs, targets, home = System.prepare_load(data, a1, a2, a3, a4, a5, drone)
                pos, loss = System.test(data, target)
                accv.append(loss)
        valid.append(sum(accv)/len(accv))

    torch.save(System.Network, PATH)
    os.system('clear')
    end = time.ctime()
    print("Start:       ", start)
    print("End:         ", end)
    print("Training Accuracy:   ", training[len(training)-1])
    print("Validation Accuracy: ", valid[len(valid)-1])
    plt.figure()
    plt.plot(training, label='Training Accuracy')
    plt.plot(valid, label='Validation')
    plt.xlabel('Epochs [-]')
    plt.ylabel('Error [m]')
    plt.legend()
    # Save Training and Validation accuracies
    open_file = open("training_results4.pkl", "wb")
    pickle.dump(training, open_file)
    open_file.close()

    file2 = open("validation_results4.pkl", "wb")
    pickle.dump(valid, file2)
    file2.close()
    exit()
    x, y, z, xnn, ynn, znn = [], [], [], [], [], []
    xt, yt, zt = [], [], []
    dop = []
    acc = []
    with torch.no_grad():
        for data, target, a1 in test_loader:
            pos = System.deploy(data, a1)
            _, loss = System.test(data, target)
            acc.append(loss)
            tlat, pdop = OLS_Trilat(data)
            dop.append(pdop)
            xt.append(float(tlat[0][0]) + float(a1[0][0]))
            yt.append(float(tlat[1][0]) + float(a1[0][1]))
            zt.append(float(tlat[2][0]) + float(a1[0][2]))
            xnn.append(System.x)
            ynn.append(System.y)
            znn.append(System.z)
            x.append(float(target[0][0]) + float(a1[0][0]))
            y.append(float(target[0][1]) + float(a1[0][1]))
            z.append(float(target[0][2]) + float(a1[0][2]))
            #System.plot3ax(x, y, z, xnn, ynn, znn)

#    plt.show()
    plt.figure()
    plt.plot(dop, label='Dilution of Precision [-]')
    plt.plot(acc, label='Accuracy [m]')
    plt.xlabel('Iteration [-]')
    plt.legend()
    plt.show()

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.set_xlim3d(0, 305)
    ax.set_ylim3d(0, 305)
    ax.set_zlim3d(0, 305)
    ax.plot3D(x, y, z, color='gray')
    ax.plot3D(xnn, ynn, znn, color='green')
    ax.plot(xt, yt, zt, color='black')
    plt.show()

    torch.save(System.Network, PATH)
    os.system('clear')
    # print('Dad Joke:')
    os.system('clear')
    dadjoke = Dadjoke()
    print(dadjoke.joke, '\n')
    print("Start:       ", start)
    print("End:         ", end)
    dadjoke = Dadjoke()
    print(dadjoke.joke)


if __name__ == "__main__":
    # Add save boolean too
    main(load=False, line = True)
