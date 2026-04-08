import csv
import torch
import numpy as np
import random as rand
import csv
import os
import itertools
from SimFunctions import *
from NNLib import *
from OLSsolver import *
import random
from tqdm import tqdm
from converge import test_convergence

def generate_unique_filename(mypath, mclen):
    file = str(mclen)+'_MonteCarlo_Results_MAE.csv'
    from os.path import isfile, join
    onlyfiles = [f for f in os.listdir(mypath) if isfile(join(mypath, f))]
    newfile = str(random.randrange(0,9))+str(random.randrange(0,9))+str(random.randrange(0,9))+str(random.randrange(0,9))+"_"+file
    while newfile in onlyfiles:
        newfile = str(random.randrange(0,9))+str(random.randrange(0,9))+str(random.randrange(0,9))+str(random.randrange(0,9))+"_"+file
    return newfile

def MonteCarlo():
    PATH1="/home/rgeng98/BobGeng-Thesis/Machine-Learning/Objective 2/Resnet/ResNet-7Anchors.pt"
    model = torch.load(PATH1)
    loss = torch.nn.L1Loss()
    MClength = 2**15
    num_states = 0
    dt = 0.01
    num_anchors = 7
    mypath = "Static_MC_Results/"
    converged=False
    while not converged:# or True:
        file = generate_unique_filename(mypath, MClength)
        with open(mypath+file, 'w', encoding='UTF8') as f:
                writer = csv.writer(f)
                writer.writerow(["ANN","TRI", "PDOP"])
                for i in tqdm(range(MClength)):
                    drone_list = [Drone([rand.randrange(0,100), rand.randrange(0,100), rand.randrange(0,100)])]
                    for i in range(num_anchors):
                        drone_list.append(Drone([rand.randrange(0,100), rand.randrange(0,100), rand.randrange(0,100)]))

                    drone_list[0].range(drone_list)

                    avail = []
                    for i in range(6):
                        avail.append([])
                        for drone in drone_list:
                            drone.move(dt)
                            drone.gps_noise()
                        
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
                    with torch.no_grad():
                        pos = model(data)

                    ann_loss = loss(pos, target)                
                    try:
                        write, first = True, True                
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
                        tlat, _ = OLS_Trilat(points, rho)
                        t = torch.tensor([[float(tlat[0][0]), float(tlat[1][0]), float(tlat[2][0])]]).to(torch.device('cpu'))
                        target = torch.tensor([[float(drone_list[0].x_tru), float(drone_list[0].y_tru), float(drone_list[0].z_tru)]]).to(torch.device('cpu'))
                        tloss = loss(t, target)
                        
                        dop = PDOP(drone_list)
                    except:
                        write = False
                
                    if write:
                        num_states += 1
                        writer.writerow([0, float(ann_loss.item()), float(tloss.item()), float(dop)])
        converged, _, _, _, _ = test_convergence(mypath)
        MClength *= 2
if __name__ == "__main__":
    MonteCarlo()
