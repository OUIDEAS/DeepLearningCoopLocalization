import torch
import torch.nn as nn
import torch.optim as optim
from NNLib import *
import os
import time
from DLoader import TrainLoader
from torch.utils.data import DataLoader
import pickle
# 100% Necessary
from dadjokes import Dadjoke
from tqdm import tqdm


def main():
    os.system('clear')
    num_anc = 7
    PATH = "Objective_2_FeedForward.pt"
    dadjoke = Dadjoke()
    print(dadjoke.joke, '\n')
    print("Loading Data...")
    start = time.ctime()
    train = TrainLoader('Training_data.csv')
    val = TrainLoader('Validation_data.csv')
    
    train_loader = DataLoader(train, batch_size=171, shuffle=True)
    validation_loader = DataLoader(val, batch_size=1000, shuffle=True)
    epochs = 363
    learning_rate = 9.34e-5

    n_in = num_anc*6 + 3*(num_anc-1)
    neural_net = FeedForward(n_inputs = n_in, hidden_size = 1280, num_layers = 6, drop = 8.71e-4)
    neural_net.apply(initialize_weights_ku)

    System = Localization(neural_net, optim.RAdam(neural_net.parameters(), lr=learning_rate), nn.L1Loss())

    print(start)
    print(System.Network)
    print('Ultra-wideband ranging simulated using Cramer-Rao Lower Bound.')
    training = []
    valid = []
    for _ in tqdm(range(epochs)):
        acct = []
        for data, target in train_loader:
            _, loss = System.train(data, target)
            acct.append(loss)

        training.append(sum(acct)/len(acct))
        accv = []
        with torch.no_grad():
            for data, target in validation_loader:
                _, loss = System.test(data, target)
                accv.append(loss)

        valid.append(sum(accv)/len(accv))

    torch.save(System.Network, PATH)
    # Save Training and Validation accuracies
    open_file = open("training_results4.pkl", "wb")
    pickle.dump(training, open_file)
    open_file.close()
    file2 = open("validation_results4.pkl", "wb")
    pickle.dump(valid, file2)
    file2.close()
    exit()
    


if __name__ == "__main__":
    # Add save boolean too
    main()
