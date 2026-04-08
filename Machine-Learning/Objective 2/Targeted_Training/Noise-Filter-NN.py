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

def main(load):
    # plt.rcParams.update({'font.size': 14})
    PATH = "L2-Reg-ResNeXt.pt"
    os.system('clear')
    print('Dad Joke:')
    dadjoke = Dadjoke()
    print(dadjoke.joke, '\n')
    print("Loading Data...")
    start = time.ctime()
    epochs = 100
    learning_rate = 0.0001

    if load:
        neural_net=torch.load(PATH)
        open_file = open("training_results.pkl", "rb")
        training = pickle.load(open_file)
        open_file.close()
        open_file = open("validation_results.pkl", "rb")
        valid = pickle.load(open_file)
        open_file.close()
        neural_net = torch.load(PATH)
    else:
        training = []
        valid = []
        #neural_net = ResNeXt(size = 1500, res_layers = 2, residuals = 3, drop = 0.1)
        neural_net = RGNet(size = 250, res_layers = 2, nodes = 10, drop = 0.1, nodal_layers=2)

    System = Localization(neural_net, optim.RAdam(neural_net.parameters(), lr=learning_rate, weight_decay=1e-5), nn.L1Loss())
    train = TrainLoader('Targeted_Training.csv')
    train_loader = DataLoader(train, batch_size=10000, shuffle=True)
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

    for e in range(epochs):
        sys.stdout.write("\rTraining Epoch: {0}/{1}".format(e+1, epochs))
        sys.stdout.flush()
        accv = []
        acct = []
        for data, target, a1 in train_loader:
            pos, loss = System.train(data, target)
            acc.append(loss)
            acct.append(loss)
            if len(acc) > N:
                average_error = sum(acc[-N:]) / N
            else:
                average_error = sum(acc) / len(acc)
            averr.append(average_error)
        training.append(sum(acct)/len(acct))


    print(average_error)
    os.system('clear')
    print('Dad Joke:')
    dadjoke = Dadjoke()
    print(dadjoke.joke, '\n')
    print("Average error at the end of training:    ", average_error)
    print(start)
    end = time.ctime()
    print("Start:       ", start)
    print("End:         ", end)
    plt.figure(1, figsize=[5,5])
    plt.xlabel('Iteration')
    plt.ylabel('Error [m]')
    plt.plot(averr)
    plt.show()

    plt.figure(2, figsize=[5,5])
    plt.plot(training, label = 'Training Accuracy')
    plt.xlabel('Epoch [-]')
    plt.ylabel('Average MAE [m]')
    plt.legend()
    plt.show()

    # Save Training and Validation accuracies
    open_file = open("training_results.pkl", "wb")
    pickle.dump(training, open_file)
    open_file.close()

    torch.save(System.Network, PATH)
    os.system('clear')
    print('Dad Joke:')
    dadjoke = Dadjoke()
    print(dadjoke.joke, '\n')
    print("Start:       ", start)
    print("End:         ", end)


if __name__ == "__main__":
    # Add save boolean too
    main(load=False)
