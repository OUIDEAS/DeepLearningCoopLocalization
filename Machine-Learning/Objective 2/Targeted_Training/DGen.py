import csv
import torch
import numpy as np
import random as rand
import csv
import os
from PosToGPS import *
import itertools
from haversine import inverse_haversine, Direction, Unit
import math
from dadjokes import Dadjoke
from SimFunctions import *

def main(waypoints):
    os.system('clear')
    window_size = 6
    w = 1
    num_anchors = 10
    with open('Targeted_Training.csv', 'w', encoding='UTF8') as f:
        writer = csv.writer(f)
        for i in range(10):
            drone_list = []
            # Create List of Drones
            for i in range(len(waypoints)):
                drone_list.append(Drone(waypoints[i][0]))

            # Set drone headings
            for i in range(len(waypoints)):
                drone_list[i].gps_noise()
                drone_list[i].set_accel(waypoints[i][1])

            print('Dad Joke:')
            dadjoke = Dadjoke()
            print(dadjoke.joke, '\n')
            dt = 0.1
            end_sim = False
            while not end_sim:
                criterion = 0
                for i in range(len(waypoints)):
                    criterion = criterion + drone_list[i].reach_wp(waypoints[i][w])

                if criterion == 11:
                    w = w + 1
                    print("WayPoint Reached")

                if w == len(waypoints[0]):
                    w = 5
                    end_sim = True

                for i in range(len(waypoints)):
                    drone_list[i].set_accel(waypoints[i][w])

                data = []
                avail = []
                r = []
                x = []
                y = []
                z = []
                i = 1
                r.append([])
                for drone in drone_list:
                    if i < 3:
                        i = i+1
                        pass
                    else:
                        r.append([])
                        x.append([])
                        y.append([])
                        z.append([])

                for drone in drone_list:
                    drone.move(dt)
                    drone.gps_noise()

                for b in range(6):
                    for drone in drone_list:
                        drone.gps_noise()
                    avail.append([])
                    drone_list[0].range(drone_list[1],drone_list[2],drone_list[3],drone_list[4],drone_list[5],drone_list[6],drone_list[7],drone_list[8],drone_list[9],drone_list[10])
                    for index in range(len(drone_list[0].r)):
                        if drone_list[0].r[index] < 300:
                            avail[b].append(DroneData(drone_list[index+1].x, drone_list[index+1].y, drone_list[index+1].z, drone_list[0].r[index]))
                    # 1st anchor drone is where others positions are referenced to, so start recycling inputs with the second one
                    i = 0
                    while len(avail[b]) < num_anchors:
                        avail[b].append(avail[b][i])
                        i = i+1

                # For each drone available in the list, save all timesteps of range readings
                for i in range(len(avail[1])):
                    for item in avail:
                        data.append(item[i].r)
                # For each drone in the list, save all time steps of ENU estimates
                for i in range(len(avail[1])-1):
                    for item in avail:
                        x[i].append((item[i+1].x - item[0].x))
                        y[i].append((item[i+1].y - item[0].y))
                        z[i].append((item[i+1].z - item[0].z))
                # Determine targets, and reference origin
                target = [(drone_list[0].x_tru - drone_list[1].x_tru), (drone_list[0].y_tru - drone_list[1].y_tru), (drone_list[0].z_tru - drone_list[1].z_tru)]
                for i in range(len(x)):
                    data = list(itertools.chain(data, x[i], y[i], z[i]))
                data = list(itertools.chain(data, target))
                writer.writerow(data)
                #writer.writerow([tL, nnL])


if __name__ == "__main__":
    wp = []
    anc_alt = 250
    # Agents waypoints first
    wp.append([[-10,0,300],[0,10, 300],[0, 10, 10],[0, 40, 10],[0,40,300], [-10,50,300]])

    # Anchors 1, 2, 3,..., 10
    wp.append([[0,0,300],[0,10, anc_alt],[0,10, anc_alt],[0,10, anc_alt],[0, 10, 300],[0,50, 300]])

    wp.append([[10,0,300],[30,10, anc_alt],[50,10, anc_alt],[50,10, anc_alt],[30, 20, 300],[5, 50, 300]])

    wp.append([[20,0,300],[60,10, anc_alt],[100,10, anc_alt],[100,10, anc_alt],[60, 20, 300],[10, 50, 300]])

    wp.append([[0,10,300],[0,50, anc_alt],[0,50, anc_alt],[0,50, anc_alt],[0, 50, 300],[0,60, 300]])

    wp.append([[10,10,300],[30,50, anc_alt],[50,50, anc_alt],[50,50, anc_alt],[30, 50, 300],[5,60, 300]])

    wp.append([[20,10,300],[60,50, anc_alt],[100,50, anc_alt],[100,50, anc_alt],[60, 50, 300],[10,60, 300]])

    wp.append([[0,4,300],[0,25, anc_alt],[0,25, anc_alt],[0,25, anc_alt],[0, 25, 300],[0,54, 300]])

    wp.append([[0,7,300],[0,35, anc_alt],[0,35, anc_alt],[0,35, anc_alt],[0, 35, 300],[0,57, 300]])

    wp.append([[20,4,300],[60,25, anc_alt],[60,25, anc_alt],[60,25, anc_alt],[60, 25, 300],[10,54, 300]])

    wp.append([[20,7,300],[60,35, anc_alt],[60,35, anc_alt],[60,35, anc_alt],[60, 35, 300],[10,57, 300]])

    main(wp)
