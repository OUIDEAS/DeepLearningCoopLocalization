import csv
import torch
import numpy as np
import random as rand
import csv
import os
import itertools
from SimFunctions import *
from NN import *
from NNLib import *
from OLSsolver import *
import pyprog
import random
import time
from Convergence import test_convergence
from ImportThisOne import *

class FeedForward(nn.Module):
    def __init__(self, size: int, layers: int, drop = 0.1):
        super().__init__()
        network = [nn.Linear(222, size), nn.Dropout(drop), nn.PReLU(num_parameters=size)]
        for i in range(layers-1):
            network.append(nn.Linear(size, size))
            network.append(nn.Dropout(drop))
            network.append(nn.PReLU(num_parameters=size))
        network.append(nn.Linear(size, 3))
        self.Network = nn.Sequential(*network)

    def forward(self, x):
        return self.Network(x)

def generate_unique_filename(mypath, mclen):
    file = str(mclen)+'_ANN_TRI.csv'
    from os.path import isfile, join
    onlyfiles = [f for f in os.listdir(mypath) if isfile(join(mypath, f))]
    newfile = str(random.randrange(0,9))+str(random.randrange(0,9))+str(random.randrange(0,9))+str(random.randrange(0,9))+"_"+file
    while newfile in onlyfiles:
        newfile = str(random.randrange(0,9))+str(random.randrange(0,9))+str(random.randrange(0,9))+str(random.randrange(0,9))+"_"+file
    return newfile

def main(waypoints):
    os.system('clear')
    PATH1="/home/rgeng98/BobGeng-Thesis/Machine-Learning/Objective 2/Resnet/Networks/Networks7ResNet.pt"
    model = torch.load(PATH1)
    mclength = 2**8
    loss = torch.nn.L1Loss()
    converged = False
    mclength *= 2
    os.system('clear')
    print("================================================================================")
    print("Starting Monte Carlo Simulation...")
    print("================================================================================")
    print("Simulation Length: ", mclength)
    print("================================================================================")
    print("Comparison Metric: ", loss)
    print("================================================================================")
    print("Methods Compared:")
    print("Neural Network")
    print("Trilateration")
    print("================================================================================")
    print("Update frequencies (Hz):")
    print("GPS:         1")
    print("UWB Ranging: 100")
    print("================================================================================")
    num_episodes_saved = 0
    num_anchors = 7
    prog = pyprog.ProgressBar("-> ", " OK!", mclength)
    prog.update()
    time.sleep(random.randrange(0, 100)/10)

    dopl, annl, tril = [], [], []
    while num_episodes_saved < mclength:
        w = 1
        drone_list = [Drone(waypoints[0][0])]
        # Create List of Drones
        for i in range(num_anchors):
            rp = [rand.randrange(0,100), rand.randrange(0,100), rand.randrange(70,100)]
            drone_list.append(Drone(rp))
        
        drone_list[0].range(drone_list)

        drone_list[0].set_accel(waypoints[0][1])


        end_sim = False
        write_all = True
        for drone in drone_list:
            drone.gps_noise()

        while not end_sim:
            if write_all:
                criterion = 0
                criterion = drone_list[0].reach_wp(waypoints[0][w])

                if criterion == 1:
                    w = w + 1

                if w == len(waypoints[0]):
                    w = len(waypoints[0])-1
                    end_sim = True
                drone_list[0].set_accel(waypoints[0][w])
                data = []

                for i in range(10):
                    drone_list[0].set_accel(waypoints[0][w])
                    drone_list[0].move(0.94/10)

                avail = []
                for drone in drone_list:
                    drone.gps_noise()
                # UWB ranging at 100hz freq
                for i in range(6):
                    avail.append([])
                    drone_list[0].range(drone_list)

                    for index in range(len(drone_list[0].r)):
                        avail[len(avail)-1].append(DroneData(drone_list[index+1].x, drone_list[index+1].y, drone_list[index+1].z, drone_list[0].r[index]))

                    drone_list[0].set_accel(waypoints[0][w])
                    drone_list[0].move(0.01)

                if len(avail) == 6:
                    for drone in drone_list:
                        drone.gps_noise()
                    for i in range(len(avail[1])):
                        for item in avail:
                            data.append(item[i].r)

                    c = 0
                    for drone in drone_list:
                        if c < 2:
                            x, y, z = [], [], []
                            c+=1
                        else:
                            x.append(drone.x - drone_list[1].x)
                            y.append(drone.y - drone_list[1].y)
                            z.append(drone.z - drone_list[1].z)

                    data = list(itertools.chain(data, x, y, z))
                    
                    try:
                        write_all = True
                        first = True
                        for drone in drone_list:
                            if first:
                                points = []
                                rho = []
                                c = 0
                                first = False
                            else:
                                points.append([drone.x, drone.y, drone.z])
                                rho.append(drone_list[0].r[c])
                                c+=1

                        tlat, dop = OLS_Trilat(points, rho)
                        
                        kf_t = torch.tensor([[float(tlat[0][0]), float(tlat[1][0]), float(tlat[2][0])]]).to(torch.device('cpu'))
                        target = torch.tensor([[drone_list[0].x_tru, drone_list[0].y_tru, drone_list[0].z_tru]])
                        loc_loss = loss(kf_t, target)
                        target = torch.tensor([[(drone_list[0].x_tru - drone_list[1].x),
                                                (drone_list[0].y_tru - drone_list[1].y),
                                                (drone_list[0].z_tru - drone_list[1].z)]]).to(torch.device("cuda"))
                        data = torch.tensor([data]).to(torch.device("cuda"))
                        with torch.no_grad():
                            pos = model(data)
                        ann_loss = loss(pos, target)
                            
                        tril.append(loc_loss.item())
                        annl.append(ann_loss.item())
                        dopl.append(dop)
                    except:
                        write_all = False
    tuples = []
    for (a, t, d) in zip(annl, tril, dopl):
        tuples.append((d, a, t))
    dop_sort = []
    ann_sort = []
    tri_sort = []
    tuples.sort()

    for t in tuples:
        dop_sort.append(t[0])
        tri_sort.append(t[1])
        ann_sort.append(t[2])

    bins = 100
    bin_size =(20 - tup[0][0])/bins

    tavg, eavg, aavg = [], [], []
    dbin = []
    conv_analysis = [[i for i in range(bins)]]
    for i in range(bins):
        min = i * bin_size
        max = (i+1) * bin_size
        a, trilateration, c = 0, 0, 0
        for t in tuples:
            # if t[0] < 1:
            if t[0] < max and t[0] > min:
                c+=1
                trilateration += t[1]
                a += t[2]
        if c > 0:
            dbin.append((min+max)/2)
            tavg.append(trilateration/c)
            aavg.append(a/c)
    
    plt.figure()
    plt.plot(dbin, tavg, linestyle = '-.', color='black', label="Trilateration")
    plt.plot(dbin, aavg, color='black', label="ResNet")
    plt.xlabel("PDOP [-]")
    plt.xlim([0,20])
    plt.ylabel("Mean Absolute Error [m]")
    plt.legend()

    counta, bins_counta = np.histogram(tril, bins=1000)
    pdf = counta / sum(counta)
    cdftrilat = np.cumsum(pdf)

    countc, bins_countc = np.histogram(annl, bins=1000)
    pdf = countc / sum(countc)
    cdfann = np.cumsum(pdf)
    plt.figure()
    plt.plot(bins_counta[1:], cdftrilat, linestyle = '-.', color='black', label="Trilateration")
    plt.plot(bins_countc[1:], cdfann, color='black', label="ResNet")
    plt.xlabel("Mean Absolute Error [m]")
    plt.ylabel("CDF [%]")
    plt.ylim([0, 1])
    plt.xlim([0, 10])
    plt.legend()

    mean, tvar = get_variance(tril)
    amean, avar = get_variance(annl)

    print("ANN MEAN:     ", amean)
    print("ANN STDEV:    ", math.sqrt(avar))

    print("TRI MEAN:     ", tmean)
    print("TRI STDEV:    ", math.sqrt(tvar))
    plt.show()


    

if __name__ == "__main__":
    wp = []
    anc_alt = 75
    max_alt = 100
    # Agents waypoints first
    wp.append([[0, 10, 10],[0, 80, 10],[10, 80, 5],[80, 10, 5]])

    # Anchors 1, 2, 3,..., 10
    wp.append([[0,0,max_alt],
               [0,10, anc_alt],
               [0,10, anc_alt],
               [0,10, anc_alt],
               [0, 10, max_alt],
               [0,10, max_alt]])

    wp.append([[10,0,max_alt],
               [max_alt/2,10, anc_alt-20],
               [max_alt/2,10, anc_alt-20],
               [max_alt/2,10, anc_alt-20],
               [30, 20, max_alt/3],
               [5, 50, max_alt/3]])

    wp.append([[20,0,max_alt],
               [max_alt,10, anc_alt+20],
               [max_alt,10, anc_alt+20],
               [max_alt,10, anc_alt+20],
               [60, 20, max_alt/3],
               [10, 50, max_alt/3]])

    wp.append([[0,10,max_alt],
               [0,max_alt/2, anc_alt-20],
               [0,max_alt/2, anc_alt-20],
               [0,max_alt/2, anc_alt-20],
               [0, max_alt/2, max_alt/3],
               [0,60, max_alt/3]])

    wp.append([[10,10, max_alt],
               [max_alt/2,max_alt/2, anc_alt+20],
               [max_alt/2,max_alt/2, anc_alt+20],
               [max_alt/2,max_alt/2, anc_alt+20],
               [30, max_alt/2, max_alt],
               [5,max_alt, max_alt]])

    wp.append([[20,10,max_alt],
               [max_alt,max_alt/2, anc_alt-20],
               [max_alt,max_alt/2, anc_alt-20],
               [max_alt,max_alt/2, anc_alt-20],
               [60, max_alt/2, max_alt],
               [10,60, max_alt]])

    wp.append([[0,4,max_alt],
               [0,max_alt, anc_alt+20],
               [0,max_alt, anc_alt+25],
               [0,max_alt, anc_alt+25],
               [0, max_alt, max_alt/3],
               [0,54, max_alt/3]])

    wp.append([[0,7,max_alt],
               [max_alt/2,max_alt, anc_alt-20],
               [max_alt/2,max_alt, anc_alt-25],
               [max_alt/2,max_alt, anc_alt-25],
               [max_alt/2, max_alt, max_alt/3],
               [0,max_alt/2, max_alt/3]])

    wp.append([[20,4,max_alt],
               [max_alt,max_alt, anc_alt+20],
               [max_alt,max_alt, anc_alt+20],
               [max_alt,max_alt, anc_alt+20],
               [max_alt,max_alt, max_alt/3],
               [10,max_alt/2, max_alt/3]])

    wp.append([[20,7,max_alt],
               [max_alt,max_alt, anc_alt-20],
               [max_alt/2,max_alt*0.75, anc_alt-20],
               [max_alt/2,max_alt*0.75, anc_alt-20],
               [60, max_alt*0.75, max_alt/3],
               [10,max_alt*0.75, max_alt/3]])
    main(wp)
