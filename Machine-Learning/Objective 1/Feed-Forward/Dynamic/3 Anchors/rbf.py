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
from tqdm import tqdm


def main(load, line):
    num_anc = 3
    PATH = "FilterLocalizationNetwork-compressed.pt"
    print('Dad Joke:')
    os.system('clear')
    dadjoke = Dadjoke()
    print(dadjoke.joke, '\n')
    print("Loading Data...")
    start = time.ctime()
    train = TrainLoader('Training_data.csv')
    val = TrainLoader('Validation_data.csv')
    
    train_loader = DataLoader(train, batch_size=400, shuffle=True)
    validation_loader = DataLoader(val, batch_size=1000, shuffle=True)
    epochs = 10
    learning_rate = 1e-3

    if load:
        neural_net = torch.load(PATH)
    else:
        # neural_net = ResNet(size=910, res_layers=3, residuals=3, drop=0.1)
        n_in = num_anc*6 + 3*(num_anc-1)
        neural_net = RBF(n_inputs = n_in, hidden_size = 910)

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
    for e in tqdm(range(epochs)):
        acct = []
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
    open_file = open("rbftraining_results.pkl", "wb")
    pickle.dump(training, open_file)
    open_file.close()

    file2 = open("rbfvalidation_results.pkl", "wb")
    pickle.dump(valid, file2)
    file2.close()
    plt.show()
    exit()
    


if __name__ == "__main__":
    # Add save boolean too
    main(load=False, line = True)
