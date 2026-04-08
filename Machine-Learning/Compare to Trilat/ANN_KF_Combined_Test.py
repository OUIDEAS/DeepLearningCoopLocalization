import csv
import torch
import numpy as np
import random as rand
import csv
import os
import itertools
import math
from dadjokes import Dadjoke
from SimFunctions import *
from NN import *
from NNLib import *
from OLSsolver import *
import sys
from EKF import EKF, KalmanFilter
import pyprog
import random
import time

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
    file = str(mclen)+'_ANN_KF_Combine.csv'
    from os.path import isfile, join
    onlyfiles = [f for f in os.listdir(mypath) if isfile(join(mypath, f))]
    newfile = str(random.randrange(0,9))+str(random.randrange(0,9))+str(random.randrange(0,9))+str(random.randrange(0,9))+file
    while newfile in onlyfiles:
        newfile = str(random.randrange(0,9))+str(random.randrange(0,9))+str(random.randrange(0,9))+str(random.randrange(0,9))+"_"+file
    return newfile

def main(waypoints):
    os.system('clear')
    PATH1="/home/rgeng98/BobGeng-Thesis/Machine-Learning/Objective 2/Resnet/ResNet-7Anchors.pt"
    model = torch.load(PATH1)
    print("================================================================================")
    print("Starting Monte Carlo Simulation...")
    print("================================================================================")
    mclength = 25000
    print("Simulation Length: ", mclength)
    loss = torch.nn.L1Loss()
    print("================================================================================")
    print("Comparison Metric: ", loss)
    print("================================================================================")
    print("Methods Compared:")
    print("Kalman Filter on Intervehicle Ranging with OLS Trilateration")
    print("Neural Network")
    print("Trilateration w/o filtering")
    print("================================================================================")
    print("Update frequencies (Hz):")
    print("GPS:         1")
    print("UWB Ranging: 100")
    print("================================================================================")
    while True:
        num_episodes_saved = 0
        num_anchors = 7
        prog = pyprog.ProgressBar("-> ", " OK!", mclength)
        prog.update()
        time.sleep(random.randrange(0, 100)/10)
        with open('ANN_KF_Combine.csv', 'w', encoding='UTF8') as f:
            writer = csv.writer(f)
            writer.writerow(["KF","ANN","TRI", "PDOP"])
            while num_episodes_saved < mclength:
                w = 1
                drone_list = [Drone(waypoints[0][0])]
                # Create List of Drones
                for i in range(num_anchors):
                    rp = [rand.randrange(0,100), rand.randrange(0,100), rand.randrange(70,100)]
                    drone_list.append(Drone(rp))
                kf = []
                drone_list[0].range(drone_list)
                for i in drone_list[0].r:
                    kf.append(KalmanFilter(i))
                drone_list[0].set_accel(waypoints[0][1])

                dopl, kfl = [], []

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
                            w = 5
                            end_sim = True

                        drone_list[0].set_accel(waypoints[0][w])
                        data = []
                        r = []
                        for d in range(num_anchors):
                            r.append([])
                        for i in range(10):
                            drone_list[0].set_accel(waypoints[0][w])
                            drone_list[0].move(0.94/10)
                            drone_list[0].range(drone_list)

                            for i in range(len(drone_list[0].r)):
                                kf[i].predict()
                                kf[i].update(drone_list[0].r[i])

                        avail = []
                        # UWB ranging at 100hz freq
                        for i in range(6):
                            avail.append([])
                            drone_list[0].range(drone_list)

                            for i in range(len(drone_list[0].r)):
                                kf[i].predict()
                                kf[i].update(drone_list[0].r[i])

                            for index in range(len(drone_list[0].r)):
                                avail[len(avail)-1].append(DroneData(drone_list[index+1].x, drone_list[index+1].y, drone_list[index+1].z, drone_list[0].r[index]))

                            drone_list[0].set_accel(waypoints[0][w])
                            drone_list[0].move(0.01)

                        for drone in drone_list:
                            drone.gps_noise()

                        if len(avail) == 6:
                            for i in range(len(avail[1])):
                                for item in avail:
                                    data.append(item[i].r)

                            first = True
                            for item in avail[0]:
                                if first:
                                    x, y, z = [], [], []
                                    first = False
                                else:
                                    x.append(item.x - avail[0][0].x)
                                    y.append(item.y - avail[0][0].y)
                                    z.append(item.z - avail[0][0].z)

                            

                            data = list(itertools.chain(data, x, y, z))
                            
                            try:
                                write_all = True
                                
                                first = True
                                c = 0
                                for drone in drone_list:
                                    if first:
                                        points = []
                                        rho = []
                                        first = False
                                    else:
                                        points.append([drone.x, drone.y, drone.z])
                                        rho.append(kf[c].x_k)
                                        c+=1

                                kftlat, dop = OLS_Trilat(points, rho)
                                if dop > 2.5:
                                    kf_t = torch.tensor([[float(kftlat[0][0]), float(kftlat[1][0]), float(kftlat[2][0])]]).to(torch.device('cpu'))
                                    target = torch.tensor([[drone_list[0].x_tru, drone_list[0].y_tru, drone_list[0].z_tru]])
                                    loc_loss = loss(kf_t, target)
                                else:
                                    target = torch.tensor([[(drone_list[0].x_tru - drone_list[1].x),
                                                    (drone_list[0].y_tru - drone_list[1].y),
                                                    (drone_list[0].z_tru - drone_list[1].z)]]).to(torch.device("cuda"))
                                    data = torch.tensor([data]).to(torch.device("cuda"))
                                    with torch.no_grad():
                                        pos = model(data)
                                    loc_loss = loss(pos, target)
                                    
                                kfl.append(loc_loss.item())
                                dopl.append(PDOP(drone_list))
                            except:
                                write_all = False
                                end_sim=True


                if write_all:
                    num_episodes_saved = num_episodes_saved + 1
                    prog.set_stat(num_episodes_saved)
                    prog.update()
                    if sum(dopl)/len(dopl) < 20:
                        for (k, d) in zip(kfl, dopl):
                            writer.writerow([k, d])


if __name__ == "__main__":
    wp = []
    anc_alt = 75
    max_alt = 100
    # Agents waypoints first
    wp.append([[-5,0,max_alt],[0,10, max_alt],[0, 10, 10],[0, 40, 10],[0,40,max_alt], [-5,50,max_alt]])

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
