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
from OLSsolver import *

def main(load):
    # plt.rcParams.update({'font.size': 14})
    PATH = "L2-Reg-ResNeXt.pt"
    os.system('clear')
    print('Dad Joke:')
    dadjoke = Dadjoke()
    print(dadjoke.joke, '\n')
    print("Loading Data...")
    start = time.ctime()
    epochs = 20
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
        #neural_net = ResNeXt(size = 1250, res_layers = 2, residuals = 3, drop = 0.1)
        neural_net = RGNet(size = 200, res_layers = 3, nodes = 5, drop = 0.1, nodal_layers=3)

    System = Localization(neural_net, optim.RAdam(neural_net.parameters(), lr=learning_rate, weight_decay=1e-5), nn.L1Loss())
    train = TrainLoader('/home/rgeng98/BobGeng-Thesis/Machine-Learning/Objective 2/Resnet/MoreInterruptions/Training_data.csv')
    test = TrainLoader('/home/rgeng98/BobGeng-Thesis/Machine-Learning/Objective 2/Resnet/MoreInterruptions/Test_data.csv')
    val = TrainLoader('/home/rgeng98/BobGeng-Thesis/Machine-Learning/Objective 2/Resnet/MoreInterruptions/Validation_data.csv')
    train_loader = DataLoader(train, batch_size=10000, shuffle=True)
    test_loader = DataLoader(test, batch_size=1, shuffle=False)
    validation_loader = DataLoader(val, batch_size=100, shuffle=True)
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

        with torch.no_grad():
            for data, target, a1 in validation_loader:
                pos, loss = System.test(data, target)
                accv.append(loss)

        valid.append(sum(accv)/len(accv))

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
    plt.plot(valid, label='Validation Accuracy')
    plt.plot(training, label = 'Training Accuracy')
    plt.xlabel('Epoch [-]')
    plt.ylabel('Average MAE [m]')
    plt.legend()
    plt.show()

    # Save Training and Validation accuracies
    open_file = open("training_results.pkl", "wb")
    pickle.dump(training, open_file)
    open_file.close()

    file2 = open("validation_results.pkl", "wb")
    pickle.dump(valid, file2)
    file2.close()

    acc = []
    averr = []
    xnn = []
    ynn = []
    znn = []
    x, xt = [], []
    y, yt = [], []
    z, zt = [], []
    pdop = []
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
            (t, dop) = OLS_Trilat(data)
            xt.append(t[0])
            yt.append(t[1])
            zt.append(t[2])
            pdop.append(dop)

            #System.plot3ax(x, y, z, xnn, ynn, znn)
        #plt.show()
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.set_xlim3d(-10, 110)
    ax.set_ylim3d(-10, 110)
    ax.set_zlim3d(-10, 110)
    ax.set_xlabel('East [-]')
    ax.set_ylabel('North [-]')
    ax.set_zlabel('Up [-]')
    plt.title('ResNet Dynamic Anchor Test')
    ax.plot3D(x, y, z, color='gray')
    ax.plot3D(xnn, ynn, znn, color='green')
    plt.show()

    ax1 = plt.subplot()
    plt.xlabel('Iteration [-]')
    ax1.set_ylabel('Mean Absolute Error [m]')
    l1, = ax1.plot(averr, color='gray')
    ax2 = ax1.twinx()
    l2, = ax2.plot(pdop, color='black')
    ax2.set_ylabel("PDOP [-]")
    plt.legend([l1, l2],["Mean Average Error", "Positional Dilution of Precision"], frameon=True, shadow=True)
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
    main(load=False)
