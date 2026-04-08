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
from dadjokes import Dadjoke
import pickle
import matplotlib.animation as animation
import time


def main(load, multi_out):
    # plt.rcParams.update({'font.size': 14})
    PATH = "FilterLocalizationNetwork-small.pt"
    os.system('clear')
    print('Dad Joke:')
    dadjoke = Dadjoke()
    print(dadjoke.joke, '\n')
    print("Loading Data...")
    start = time.ctime()
    train = TrainLoader('Training_data.csv')
    test = TrainLoader('Test_data.csv')
    val = TrainLoader('Validation_data.csv')
    train_loader = DataLoader(train, batch_size=1, shuffle=True)
    test_loader = DataLoader(test, batch_size=1, shuffle=False)
    validation_loader = DataLoader(val, batch_size=1, shuffle=True)
    epochs = 1
    learning_rate = 0.0001

    if load:
        neural_net = torch.load(PATH)
    else:
        layers = 15
        size = 1500
        if multi_out:
            neural_net = Net3(n_inputs=204, hidden_size=size, num_layers=layers)
        else:
            neural_net = NeuralNetwork(n_inputs=204, hidden_size=size, num_layers=layers)

    System = Localization(neural_net, optim.RAdam(neural_net.parameters(), lr=learning_rate), nn.L1Loss())
    acc = []
    averr = []
    N = 100
    os.system('clear')
    print('Dad Joke:')
    dadjoke = Dadjoke()
    print(dadjoke.joke, '\n')
    print(start)
    print(System.Network)
    print('Ultra-wideband ranging simulated.')
    val_acc = []
    train_acc = []
    i = 0
    for e in range(epochs):
        accv = []
        acct = []
        for data, target, a1 in train_loader:
            i = i+1
            sys.stdout.write("\rTraining iteration: {0}".format(i))
            sys.stdout.flush()
            pos, loss = System.train(data, target)
            acc.append(loss)
            acct.append(loss)
            if len(acc) > N:
                average_error = sum(acc[-N:]) / N
            else:
                average_error = sum(acc) / len(acc)
            averr.append(average_error)

        train_acc.append(sum(acct)/len(acct))

        with torch.no_grad():
            for data, target, a1 in validation_loader:
                pos, loss = System.test(data, target)
                accv.append(loss)

        val_acc.append(sum(accv)/len(accv))

    print(average_error)
    os.system('clear')
    print('Dad Joke:')
    dadjoke = Dadjoke()
    print(dadjoke.joke, '\n')
    print("Average error at the end of training:    ", average_error)
    print(start)
    print("Average error at the end of training:        ", average_error)
    end = time.ctime()
    print("Start:       ", start)
    print("End:         ", end)
    plt.figure(1, figsize=[5,5])
    plt.xlabel('Iteration')
    plt.ylabel('Error [m]')
    plt.plot(averr)
    plt.show()

    plt.figure(2, figsize=[5,5])
    plt.plot(val_acc, label='Validation Accuracy')
    plt.plot(train_acc, label = 'Training Accuracy')
    plt.xlabel('Epoch [-]')
    plt.ylabel('Average MAE [m]')
    plt.legend()
    plt.show()

    # Save Training and Validation accuracies
    open_file = open("training_results.pkl", "wb")
    pickle.dump(train_acc, open_file)
    open_file.close()

    file2 = open("validation_results.pkl", "wb")
    pickle.dump(val_acc, file2)
    file2.close()

    acc = []
    averr = []
    xnn = []
    ynn = []
    znn = []
    x = []
    y = []
    z = []
    with torch.no_grad():
        for data, target, a1 in test_loader:
            pos = System.deploy(data, a1)
            pos, loss = System.test(data, target)
            acc.append(loss)
            if len(acc) > N:
                average_error = sum(acc[-N:]) / N
            else:
                average_error = sum(acc) / len(acc)
            averr.append(average_error)
            xnn.append(System.x)
            ynn.append(System.y)
            znn.append(System.z)
            x.append(float(target[0][0]) + float(a1[0][0]))
            y.append(float(target[0][1]) + float(a1[0][1]))
            z.append(float(target[0][2]) + float(a1[0][2]))
            System.plot3ax(x, y, z, xnn, ynn, znn)

    plt.show()
    plt.figure(3, figsize = [5,5])
    plt.xlabel('Iteration [-]')
    plt.ylabel('Mean Absolute Error [m]')
    plt.plot(averr)
    plt.show()
    torch.save(System.Network, PATH)
    os.system('clear')
    print('Dad Joke:')
    dadjoke = Dadjoke()
    print(dadjoke.joke, '\n')
    print("Start:       ", start)
    print("End:         ", end)


if __name__ == "__main__":
    # Add save boolean too
    main(load=False, multi_out = False)
