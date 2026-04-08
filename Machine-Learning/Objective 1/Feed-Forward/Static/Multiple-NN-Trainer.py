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
validation_loader = DataLoader(val, batch_size=100, shuffle=True)
hidden_sizes = [50, 100, 150, 200]
os.system('clear')
ae2 = []
ae3 = []
ae4 = []
ae5 = []
aev2 = []
aev3 = []
aev4 = []
aev5 = []
for hidden_size in hidden_sizes:
    class NN(nn.Module):
        def __init__(self, hidden_size):
            super(NN,self).__init__()
            self.fc1 = nn.Linear(5, hidden_size)
            self.fc2 = nn.Linear(hidden_size, hidden_size)
            #self.fc3 = nn.Linear(hidden_size, hidden_size)
            #self.fc4 = nn.Linear(hidden_size, hidden_size)
            #self.fc5 = nn.Linear(hidden_size, hidden_size)
            self.fc_out = nn.Linear(hidden_size,3)

        def forward(self,x):
            x = torch.tanh(self.fc1(x))
            x = torch.tanh(self.fc2(x))
            #x = torch.tanh(self.fc3(x))
            #x = torch.tanh(self.fc4(x))
            #x = torch.tanh(self.fc5(x))
            x = torch.tanh(self.fc_out(x))
            return x

    PATH = 'ffsl2hl.pt'
    # Print program details to terminal
    print("File:             Static-NN-Trainer.py\n")
    print("NN:               " + PATH + "\n")

    device = torch.device("cuda")
    print("Start Time:      ", start + "\n")
    print("Learning Rate:   ", learning_rate , "\n")

    model = NN(hidden_size = hidden_size).to(device)
    #model = torch.load(PATH)
    criterion = nn.L1Loss()

    optimizer = optim.Adam(model.parameters(), lr = learning_rate)
    

    model.train()
    acc2 = []
    i = 0
    averr2 = []
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
            acc2.append(loss.item())
            i_with_commas = "{:,}".format(i)
            sys.stdout.write("\rTraining iteration: {0}     Epoch: {1}/{2}".format(i_with_commas, e, epochs))
            sys.stdout.flush()
            if len(acc2)>N:
                average_error = sum(acc2[-N:])/N
            else:
                average_error = sum(acc2)/len(acc2)
            averr2.append(average_error)
            i = i + 1

    ae2.append(average_error)

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
            if len(acc)>N:
                average_error = sum(acc[-N:])/N
            else:
                average_error = sum(acc)/len(acc)
            averr.append(average_error)
    aev2.append(average_error)

    class NN(nn.Module):
        def __init__(self, hidden_size):
            super(NN,self).__init__()
            self.fc1 = nn.Linear(5, hidden_size)
            self.fc2 = nn.Linear(hidden_size, hidden_size)
            self.fc3 = nn.Linear(hidden_size, hidden_size)
            #self.fc4 = nn.Linear(hidden_size, hidden_size)
            #self.fc5 = nn.Linear(hidden_size, hidden_size)
            self.fc_out = nn.Linear(hidden_size,3)

        def forward(self,x):
            x = torch.tanh(self.fc1(x))
            x = torch.tanh(self.fc2(x))
            x = torch.tanh(self.fc3(x))
            #x = torch.tanh(self.fc4(x))
            #x = torch.tanh(self.fc5(x))
            x = torch.tanh(self.fc_out(x))
            return x

    PATH = 'ffsl3hl.pt'
    # Print program details to terminal
    print("File:             Static-NN-Trainer.py\n")
    print("NN:               " + PATH + "\n")

    device = torch.device("cuda")
    print("Start Time:      ", start + "\n")
    print("Learning Rate:   ", learning_rate , "\n")

    model = NN(hidden_size = hidden_size).to(device)
    criterion = nn.L1Loss()

    #optimizer = optim.Adam(model.parameters(), lr = learning_rate)
    optimizer = optim.Adadelta(model.parameters())
    
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
                average_error = sum(acc2)/len(acc2)
            averr.append(average_error)
            i = i + 1

    ae3.append(average_error)

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
            if len(acc)>N:
                average_error = sum(acc[-N:])/N
            else:
                average_error = sum(acc)/len(acc)
            averr.append(average_error)

    aev3.append(average_error)

    os.system('clear')


    class NN(nn.Module):
        def __init__(self, hidden_size):
            super(NN,self).__init__()
            self.fc1 = nn.Linear(5, hidden_size)
            self.fc2 = nn.Linear(hidden_size, hidden_size)
            self.fc3 = nn.Linear(hidden_size, hidden_size)
            self.fc4 = nn.Linear(hidden_size, hidden_size)
            #self.fc5 = nn.Linear(hidden_size, hidden_size)
            self.fc_out = nn.Linear(hidden_size,3)

        def forward(self,x):
            x = torch.tanh(self.fc1(x))
            x = torch.tanh(self.fc2(x))
            x = torch.tanh(self.fc3(x))
            x = torch.tanh(self.fc4(x))
            #x = torch.tanh(self.fc5(x))
            x = torch.tanh(self.fc_out(x))
            return x

    PATH = 'ffsl4hl.pt'
    # Print program details to terminal
    print("File:             Static-NN-Trainer.py\n")
    print("NN:               " + PATH + "\n")

    device = torch.device("cuda")
    print("Start Time:      ", start + "\n")
    print("Learning Rate:   ", learning_rate , "\n")

    model = NN(hidden_size = hidden_size).to(device)
    #model = torch.load(PATH)
    criterion = nn.L1Loss()

    #optimizer = optim.Adam(model.parameters(), lr = learning_rate)
    optimizer = optim.Adadelta(model.parameters())

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
                average_error = sum(acc2)/len(acc2)
            averr.append(average_error)
            i = i + 1

    ae4.append(average_error)

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
            if len(acc)>N:
                average_error = sum(acc[-N:])/N
            else:
                average_error = sum(acc)/len(acc)
            averr.append(average_error)

    aev4.append(average_error)


    class NN(nn.Module):
        def __init__(self, hidden_size):
            super(NN,self).__init__()
            self.fc1 = nn.Linear(5, hidden_size)
            self.fc2 = nn.Linear(hidden_size, hidden_size)
            self.fc3 = nn.Linear(hidden_size, hidden_size)
            self.fc4 = nn.Linear(hidden_size, hidden_size)
            self.fc5 = nn.Linear(hidden_size, hidden_size)
            self.fc_out = nn.Linear(hidden_size,3)

        def forward(self,x):
            x = torch.tanh(self.fc1(x))
            x = torch.tanh(self.fc2(x))
            x = torch.tanh(self.fc3(x))
            x = torch.tanh(self.fc4(x))
            x = torch.tanh(self.fc5(x))
            x = torch.tanh(self.fc_out(x))
            return x

    PATH = 'ffsl5hl.pt'
    # Print program details to terminal
    print("File:             Static-NN-Trainer.py\n")
    print("NN:               " + PATH + "\n")

    device = torch.device("cuda")
    print("Start Time:      ", start + "\n")
    print("Learning Rate:   ", learning_rate , "\n")

    model = NN(hidden_size = hidden_size).to(device)
    #model = torch.load(PATH)
    criterion = nn.L1Loss()

#    optimizer = optim.Adam(model.parameters(), lr = learning_rate)
    optimizer = optim.Adadelta(model.parameters())

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
                average_error = sum(acc2)/len(acc2)
            averr.append(average_error)
            i = i + 1

    ae5.append(average_error)

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
            if len(acc)>N:
                average_error = sum(acc[-N:])/N
            else:
                average_error = sum(acc)/len(acc)
            averr.append(average_error)

    aev5.append(average_error)

print("Feed Forward Nerual Network")
for i in range(len(hidden_sizes)):
    print("Hidden Size:             ", hidden_sizes[i])
    print("2 Hidden Layers:         ", ae2[i],"    Validation error:       ", aev2[i])
    print("3 Hidden Layers:         ", ae3[i],"    Validation error:       ", aev3[i])
    print("4 Hidden Layers:         ", ae4[i],"    Validation error:       ", aev4[i])
    print("5 Hidden Layers:         ", ae5[i],"    Validation error:       ", aev5[i])