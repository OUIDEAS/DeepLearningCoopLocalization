import torch
import torch.nn as nn
import torch.optim as optim
from NNLib import *
import os
import time
from DLoader import TrainLoader
from torch.utils.data import DataLoader
import pickle
from tqdm import tqdm

def main():
    num_anc = 8
    PATH = "FilterLocalizationNetwork-compressed.pt"
    print("Loading Data...")
    start = time.ctime()
    train = TrainLoader('Training_data.csv')
    val = TrainLoader('Validation_data.csv')
    
    train_loader = DataLoader(train, batch_size=181, shuffle=True)
    validation_loader = DataLoader(val, batch_size=1000, shuffle=True)
    epochs = 232
    learning_rate = 1.439e-4

    n_in = num_anc*6 + 3*(num_anc-1)
    neural_net = FeedForward(n_inputs = n_in, hidden_size = 910, num_layers = 5, drop = 8.652e-4)
    neural_net.apply(initialize_weights_ku)

    System = Localization(neural_net, optim.RAdam(neural_net.parameters(), lr=learning_rate), nn.L1Loss())
    acc = []
    averr = []
    N = 100
    os.system('clear')
    print(System.Network)
    training = []
    valid = []
    for e in tqdm(range(epochs)):
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
    
    open_file = open("training_results.pkl", "wb")
    pickle.dump(training, open_file)
    open_file.close()

    file2 = open("validation_results.pkl", "wb")
    pickle.dump(valid, file2)
    file2.close()

    exit()
    


if __name__ == "__main__":
    main()
