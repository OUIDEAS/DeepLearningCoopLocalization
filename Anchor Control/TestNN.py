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

def main():
    # plt.rcParams.update({'font.size': 14})
    PATH = "Localization-Network-3-Out.pt"
    os.system('clear')
    start = time.ctime()
    test_loader = TrainLoader('TESTING_DATASET.csv')
    test = DataLoader(test_loader, batch_size=1, shuffle=False)
    neural_net = torch.load(PATH)
    System = Localization(neural_net, optim.RAdam(neural_net.parameters(), lr=0.00000001), nn.L1Loss())#, nn.L1Loss())
    acc = []
    averr = []
    N = 100
    for data, target, a1 in test:
        with torch.no_grad():
            pos, loss = System.test(data, target)
            acc.append(loss)

    nnx, nny, nnz = [],[],[]
    tx, ty, tz = [],[],[]
    x, y, z = [],[],[]
    with torch.no_grad():
        for data, target, a1 in test:
            pos = System.deploy(data, target)
            nnx.append(System.x)
            nny.append(System.y)
            nnz.append(System.z)
            try:
                t = OLS_Trilat(data)
                tx.append(float(t[0]) + float(a1[0][0]))
                ty.append(float(t[1]) + float(a1[0][1]))
                tz.append(float(t[2]) + float(a1[0][2]))
                x.append(float(target[0][0] + a1[0][0]))
                y.append(float(target[0][1] + a1[0][1]))
                z.append(float(target[0][2] + a1[0][2]))
            except:
                x.append(float(target[0][0] + a1[0][0]))
                y.append(float(target[0][1] + a1[0][1]))
                z.append(float(target[0][2] + a1[0][2]))

    fig = plt.figure(1)
    ax = plt.axes(projection='3d')
    ax.set_xlim3d(0, 305)
    ax.set_ylim3d(0, 305)
    ax.set_zlim3d(0, 305)
    ax.plot3D(x, y, z, color='gray')
    ax.plot(nnx, nny, nnz, color='green')
    #ax.plot(tx, ty, tz, color='red')

    plt.show()


if __name__ == "__main__":
    main()
