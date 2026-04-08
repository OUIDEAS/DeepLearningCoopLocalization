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
    num_anc = 7
    os.system('clear')
    PATH = "FilterLocalizationNetwork-compressed.pt"
    print('Dad Joke:')
    dadjoke = Dadjoke()
    print(dadjoke.joke, '\n')
    print("Loading Data...")
    train = TrainLoader('Training_data.csv')
    val = TrainLoader('Validation_data.csv')
    
    train_loader = DataLoader(train, batch_size=181, shuffle=True)
    validation_loader = DataLoader(val, batch_size=1000, shuffle=True)
    epochs = 232
    learning_rate = 1.439e-4
    index = 0
    inits = [initialize_weights_kn, initialize_weights_ku, initialize_weights_xn, initialize_weights_xu, initialize_weights_orth_]
    title = ['kn','ku','xn','xu','orth']
    for initialization in inits:
        # neural_net = ResNet(size=910, res_layers=3, residuals=3, drop=0.1)
        n_in = num_anc*6 + 3*(num_anc-1)
        neural_net = FeedForward(n_inputs = n_in, hidden_size = 910, num_layers = 5, drop = 8.652e-4)
        neural_net.apply(initialization)

        System = Localization(neural_net, optim.RAdam(neural_net.parameters(), lr=learning_rate), nn.L1Loss())
        print(System.Network)
        print('Ultra-wideband ranging simulated using Cramer-Rao Lower Bound.')
        training = []
        valid = []
        for e in range(epochs):
            acct = []
            sys.stdout.write("\rTraining epoch: {0}/{1}".format(e,epochs))
            sys.stdout.flush()
            for data, target, a1 in train_loader:
                pos, loss = System.train(data, target)
                acct.append(loss)
                
            training.append(sum(acct)/len(acct))
            accv = []
            with torch.no_grad():
                for data, target, a1 in validation_loader:
                    # inputs, targets, home = System.prepare_load(data, a1, a2, a3, a4, a5, drone)
                    pos, loss = System.test(data, target)
                    accv.append(loss)
            valid.append(sum(accv)/len(accv))

        torch.save(System.Network, PATH)

        # Save Training and Validation accuracies
        open_file = open(title[index]+"_training_results.pkl", "wb")
        pickle.dump(training, open_file)
        open_file.close()

        file2 = open(title[index]+"_validation_results.pkl", "wb")
        pickle.dump(valid, file2)
        file2.close()

    exit()
    


if __name__ == "__main__":
    # Add save boolean too
    main(load=False, line = True)
