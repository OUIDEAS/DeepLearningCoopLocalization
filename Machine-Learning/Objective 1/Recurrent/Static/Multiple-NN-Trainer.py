import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import os
import time
import sys
from DLoader import TriLatData
from torch.utils.data import DataLoader
import csv
import math

os.system('clear')
print("Loading Data...")
start = time.ctime()
epochs = 5
learning_rate = 0.0001
train = TriLatData('training_data.csv')
test = TriLatData('test_data.csv')
val = TriLatData('validation_data.csv')
train_loader = DataLoader(train, batch_size=1, shuffle=True)
test_loader = DataLoader(test, batch_size=1, shuffle=False)
validation_loader = DataLoader(val, batch_size=1, shuffle=True)
hidden_sizes = [50, 100, 150, 200]
os.system('clear')
layers = 2
accuracy = []
ae = [] # Location of each accuracy (notation: hidden_size.num_layers) [50.2, 50.3, 50.4, 50.5, 100.2, 100.3, 100.4, 100.5, 150.2, 150.3, 150.4, 150.5, 200.2, 200.3, 200.4, 200.5]
aev = []
for hidden_size in hidden_sizes:
    for i in range(4):
        print(hidden_size)
        class NN(nn.Module):
            def __init__(self, hidden_size, rnn_layers):
                super(NN,self).__init__()
                self.hidden_size = hidden_size
                self.rnn_layers = rnn_layers
                self.fc1 = nn.Linear(5, hidden_size)
                self.rnn = nn.RNN(hidden_size, hidden_size, self.rnn_layers, batch_first=False, nonlinearity='tanh')
                self.fc = nn.Linear(hidden_size,3)

            def forward(self,x):
                x = torch.tanh(self.fc1(x))
                x = x.reshape(1,1,self.hidden_size)
                h0 = torch.zeros(self.rnn_layers, x.size(0), self.hidden_size)
                out, hn = self.rnn(x, h0.detach())
                out = torch.tanh(self.fc(out[:, -1, :]))
                x = out.reshape(1,3)
                return x
        if i == 0:
            PATH = 'rnnsl2hl.pt'
        elif i == 1:
            PATH = 'rnnsl3hl.pt'
        elif i == 2:
            PATH = 'rnnsl4hl.pt'
        elif i == 3:
            PATH = 'rnnsl5hl.pt'

        # Print program details to terminal
        print("File:             Static-RNN-Trainer.py\n")
        print("NN:               " + PATH + "\n")

        device = torch.device("cpu")
        print("Start Time:      ", start + "\n")
        print("Learning Rate:   ", learning_rate , "\n")

        model = NN(hidden_size = hidden_size, rnn_layers = i+2).to(device)
        #model = torch.load(PATH)
        criterion = nn.L1Loss()

        optimizer = optim.Adam(model.parameters(), lr = learning_rate)

        model.train()
        acc = []
        i = 0
        averr = []
        N = 1000

        for e in range(epochs):
            for data, target, a1, drone, r1 in train_loader:
                data = data/torch.max(abs(data))
                data, target = data.to(device), target.to(device)
                optimizer.zero_grad()
                score = model(data)
                loss = criterion(score, target)
                loss.backward()
                optimizer.step()
                acc.append(loss.item())
                i_with_commas = "{:,}".format(i)
                sys.stdout.write("\rTraining iteration: {0}     Epoch: {1}/{2}".format(i_with_commas, e, epochs))
                sys.stdout.flush()
                if len(acc)>N:
                    average_error = sum(acc[-N:])/N
                else:
                    average_error = sum(acc)/len(acc)
                averr.append(average_error)
                i = i + 1

        ae.append(average_error)
        torch.save(model, PATH)
        plt.show()
        acc = []
        averr = []
        os.system('clear')
        print('Validating Network...')
        with torch.no_grad():
            for data, target, a1, drone, r1 in validation_loader:
                data = data/torch.max(abs(data))
                data, target = data.to(device), target.to(device)
                score = model(data)
                loss = criterion(score, target)
                acc.append(loss.item())
                average_error = sum(acc)/len(acc)
                averr.append(average_error)
        aev.append(average_error)

        os.system('clear')
#Simplify the print loop
forlist = [0, 4, 8, 12]
print("Recurrent Neural Networks\n")
for a in forlist:
    if a == 0:
        print("Hidden Size:             50")
    elif a == 4:
        print("Hidden Size:             100")
    elif a == 8:
        print("Hidden Size:             150")
    elif a == 12:
        print("Hidden Size:             200")
    
    print("2 Hidden Layers:         ", ae[a],  "    Validation error:       ", aev[a])
    print("3 Hidden Layers:         ", ae[a+1],"    Validation error:       ", aev[a+1])
    print("4 Hidden Layers:         ", ae[a+2],"    Validation error:       ", aev[a+2])
    print("5 Hidden Layers:         ", ae[a+3],"    Validation error:       ", aev[a+3],"\n")
