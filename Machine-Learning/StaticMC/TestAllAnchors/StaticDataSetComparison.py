import csv
import torch
import random as rand
import csv
import os
import itertools
from SimFunctions import *
from NNLib import *
from ImportThisOne import *
from OLSsolver import *
import random
from tqdm import tqdm
from plot_errors import test_convergence
import pickle
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import math

def generate_unique_filename(mypath, mclen):
    file = str(mclen)+'_MonteCarlo_Results_MAE.csv'
    from os.path import isfile, join
    onlyfiles = [f for f in os.listdir(mypath) if isfile(join(mypath, f))]
    newfile = str(random.randrange(0,9))+str(random.randrange(0,9))+str(random.randrange(0,9))+str(random.randrange(0,9))+file
    while newfile in onlyfiles:
        newfile = str(random.randrange(0,9))+str(random.randrange(0,9))+str(random.randrange(0,9))+str(random.randrange(0,9))+"_"+file
    return newfile

def MonteCarlo():
    os.system('clear')
    # PATH = "/home/rgeng98/BobGeng-Thesis/Machine-Learning/Objective 2/Resnet/"
    # PATH = "/home/rgeng98/BobGeng-Thesis/Machine-Learning/Objective 1/Feed-Forward/Dynamic/"
    PATH = "/home/rgeng98/BobGeng-Thesis/Machine-Learning/Objective 2/Resnet/Networks/Networks"
    loss = torch.nn.L1Loss()
    num_states = 0
    dt = 0.01
    key = []
    stdev = []
    mean = []
    tri_stdev = []
    tri_mean = []
    for i in range(7):
        num_anchors = i+4
        model = torch.load(PATH+str(num_anchors)+"ResNet.pt")
        
        key.append(num_anchors)
        mypath = "Static_MC_Results_"+str(num_anchors)+"/"
        Converge = False
        iter = 0
        os.system('clear')
        print('Monte Carlo Simulation on '+str(num_anchors)+' Anchors.')
        while not Converge:
            file_len = 2**(8+iter)
            file = generate_unique_filename(mypath, file_len)
            with open(mypath+file, 'w', encoding='UTF8') as f:
                writer = csv.writer(f)
                writer.writerow(["ANN","TRI","Powell"])
                for i in tqdm(range(file_len)):
                    write = True
                    drone_list = [Drone([rand.randrange(0,100), rand.randrange(0,100), rand.randrange(0,100)])]
                    for i in range(num_anchors):
                        drone_list.append(Drone([rand.randrange(0,100), rand.randrange(0,100), rand.randrange(0,100)]))

                    drone_list[0].range(drone_list)
                    avail = []
                    for drone in drone_list:
                        drone.gps_noise()

                    for i in range(6):
                        avail.append([])
                        drone_list[0].range(drone_list)

                        for index in range(len(drone_list[0].r)):
                            avail[len(avail)-1].append(DroneData(drone_list[index+1].x, drone_list[index+1].y, drone_list[index+1].z, drone_list[0].r[index]))
                    data = []
                    for i in range(len(avail[1])):
                        for item in avail:
                            data.append(item[i].r)  
                    first = True
                    for item in avail[len(avail)-1]:
                        if first:
                            x, y, z = [], [], []
                            first = False
                        else:
                            x.append(item.x - avail[len(avail)-1][0].x)
                            y.append(item.y - avail[len(avail)-1][0].y)
                            z.append(item.z - avail[len(avail)-1][0].z)
                    target = torch.tensor([[(drone_list[0].x_tru - drone_list[1].x), 
                                            (drone_list[0].y_tru - drone_list[1].y), 
                                            (drone_list[0].z_tru - drone_list[1].z)]]).to(torch.device("cuda"))
                                        
                    data = list(itertools.chain(data, x, y, z))
                    data = torch.tensor([data]).to(torch.device("cuda"))

                    pos = model(data)
                    ann_loss = loss(pos, target)
                    first = True

                    c = 0
                    for drone in drone_list:
                        if first:
                            points = []
                            rho = []
                            first = False
                        else:
                            points.append([drone.x, drone.y, drone.z])
                            rho.append(drone_list[0].r[c])
                            c+=1
                    try:
                        tlat = OLS_Trilat(points, rho)
                        target = torch.tensor([[drone_list[0].x_tru, drone_list[0].y_tru, drone_list[0].z_tru]]).to(torch.device('cpu'))
                        t = torch.tensor([[float(tlat[0][0]), float(tlat[1][0]), float(tlat[2][0])]]).to(torch.device('cpu'))
                        tloss = loss(t, target)

                        tlat = Powell_Trilat(points, rho)
                        target = torch.tensor([[drone_list[0].x_tru, drone_list[0].y_tru, drone_list[0].z_tru]]).to(torch.device('cpu'))
                        t = torch.tensor([[float(tlat[0][0]), float(tlat[1][0]), float(tlat[2][0])]]).to(torch.device('cpu'))
                        tloss_powell = loss(t, target)
                    except:
                        write = False

                    dop = PDOP(drone_list)
                    if write:
                        if dop < 20 and ann_loss.item() < 100 and tloss.item() < 100:
                            write = True
                        else:
                            write = False
                
                    if write:
                        num_states += 1
                        writer.writerow([num_anchors, float(ann_loss.item()), float(tloss.item()), float(tloss_powell.item())])
            
            Converge, avg, standard_dev, tri_avg, tri_standard_dev = test_convergence(num_anchors)
            if Converge:
                stdev.append(standard_dev)
                mean.append(avg)
                tri_stdev.append(tri_avg)
                tri_mean.append(tri_standard_dev)

            iter += 1
    results = {
        "Key": key,
        "ANN_StDev": stdev,
        "ANN_Mean": mean,
        "TRI_StDev":tri_stdev,
        "TRI_Mean": tri_mean
    }

    open_file = open("Results.pkl", "wb")
    pickle.dump(results, open_file)
    open_file.close()

    plt.figure()
    plt.errorbar(key, mean, yerr = stdev, color='black', label="ResNet", capsize = 8)
    plt.errorbar(key, tri_mean, yerr = tri_stdev, color='gray', label = "Trilateration", alpha = 0.5, capsize=8)
    plt.legend()
    plt.xlabel('Number of Anchors [-]')
    plt.ylabel('Error [m]')
    plt.savefig("Trilateration_vs_ResNet.png")
    plt.show()


if __name__ == "__main__":
    MonteCarlo()