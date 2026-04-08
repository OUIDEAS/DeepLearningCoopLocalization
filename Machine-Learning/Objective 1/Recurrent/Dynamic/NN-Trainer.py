import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import os
import time
import sys
from DLoader import TriLatData
from torch.utils.data import DataLoader

os.system('clear')
print("Loading Data...")
start = time.ctime()
epochs = 1
learning_rate = 0.000001
train = TriLatData('training_data.csv')
test  = TriLatData('test_data.csv')
val   = TriLatData('validation_data.csv')
train_loader = DataLoader(train, batch_size=1, shuffle=True)
test_loader = DataLoader(test, batch_size=1, shuffle=False)
validation_loader = DataLoader(val, batch_size=1, shuffle=True)
hidden_size = 200
os.system('clear')
layers = 5

class NN(nn.Module):
    def __init__(self, hidden_size, rnn_layers):
        super(NN,self).__init__()
        self.hidden_size = hidden_size
        self.rnn_layers = rnn_layers
        self.fc1 = nn.Linear(17, hidden_size)
        self.rnn = nn.RNN(hidden_size, hidden_size, self.rnn_layers, batch_first=False, nonlinearity='tanh')
        #self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc = nn.Linear(hidden_size,3)

    def forward(self,x):
        x = torch.tanh(self.fc1(x))
        x = x.reshape(1,1,self.hidden_size)
        h0 = torch.zeros(self.rnn_layers, x.size(0), self.hidden_size)
        out, hn = self.rnn(x, h0.detach())
        out = torch.tanh(self.fc(out[:, -1, :]))
        x = out.reshape(1,3)
        #x = torch.tanh(self.fc(x))
        return x

PATH = 'rnndl.pt'

#model = NN(hidden_size = hidden_size, rnn_layers = layers).to(device)
model = torch.load(PATH)

# Print program details to terminal
print("File:             Dynamic-RNN-Trainer.py\n")
print("NN:               " + PATH + "\n")

device = torch.device("cpu")
print("Start Time:      ", start + "\n")
print("Learning Rate:   ", learning_rate , "\n")


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
    torch.save(model, PATH)

torch.save(model, PATH)
print()

#plt.plot(acc)
#plt.plot(averr)
#plt.title("Recurrent Neural Network Static Anchor Point Training")
#plt.show()
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

plt.plot(acc)
plt.plot(averr)
plt.title("Recurrent Neural Network Validation")
plt.show()

x = []
y = []
z = []
x_tru = []
y_tru = []
z_tru = []
i = 0

os.system('clear')
print("Testing Neural Network...")
with torch.no_grad():
    for data, target, a1, drone, r1 in test_loader:
        data = data/torch.max(abs(data))
        data = data.to(device)
        score = model(data)
        x.append(float(a1[0][0] + r1[0][0] * score[0][0]))
        y.append(float(a1[0][1] + r1[0][0] * score[0][1]))
        z.append(float(a1[0][2] + r1[0][0] * score[0][2]))
        x_tru.append(float(drone[0][0]))
        y_tru.append(float(drone[0][1]))
        z_tru.append(float(drone[0][2]))

def plot3ax(x,y,z, xnn,ynn,znn):
    fig = plt.figure(1)
    plt.clf()
    ax = plt.axes(projection='3d')
    ax.set_xlim3d(-6, 6)
    ax.set_ylim3d(-6, 6)
    ax.set_zlim3d(-6, 6)
    plt.title('RNN Static Anchor Test')
    ax.plot3D(x,y,z,color='gray')
    ax.scatter(xnn,ynn,znn,color='green')
    plt.pause(0.01)  # pause a bit so that plots are updated

for i in range(len(x_tru)):
    plot3ax(x_tru[:i],y_tru[:i],z_tru[:i],x[:i],y[:i],z[:i])

plt.show()
