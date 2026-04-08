import csv
import numpy as np
import random as rand
import csv
import os
import itertools
from dadjokes import Dadjoke
from haversine import inverse_haversine, Direction, Unit
import math
import random
from tqdm import tqdm
from NNLib import DataStandardizer
# Sensor noise is modelled using a cramer rao lower bound for sensor error
def sensor_noise(r):
    # Bandwidth
    b = 3.1*10**9
    # Speed of signal
    c = 3e8
    # Calculate a signal to noise ratio based on distance
    snr = 10 - r*(10-0.01)/300
    sigsq = 1/(8*(math.pi**2)*(b**2)*snr)
    sig = 40*c*math.sqrt(sigsq)
    sensor = float(r + np.random.normal(0, sig,1))
    return sensor


class DroneData():
    def __init__(self, x, y, z, r):
        self.x = x
        self.y = y
        self.z = z
        self.r = r


class Drone(object):
    # Save the initial position of the drones
    def __init__(self, min_pos=0, max_pos=100):
        self.min = min_pos
        self.max = max_pos
        self.max_vel = 10
        self.min_vel = -10
        self.dt = 0.01
        self.x_tru = rand.randrange(self.min, self.max, 1) + rand.randrange(0, 100)/100
        self.y_tru = rand.randrange(self.min, self.max, 1) + rand.randrange(0, 100)/100
        self.z_tru = rand.randrange(self.min, self.max, 1) + rand.randrange(0, 100)/100
        self.vx = rand.randrange(self.min_vel, self.max_vel, 2)
        self.vy = rand.randrange(self.min_vel, self.max_vel, 2)
        self.vz = rand.randrange(self.min_vel, self.max_vel, 2)
        if self.vx == 0:
            self.vx = 10
        if self.vy == 0:
            self.vy = 10
        if self.vz == 0:
            self.vz = 10

        # Initialize instance variables
        self.r = None

    def noise(self):
        self.x = self.x_tru + float(np.random.normal(0,0.15,1))
        self.y = self.y_tru + float(np.random.normal(0,0.15,1))
        self.z = self.z_tru + float(np.random.normal(0,0.225,1))

    def move(self):

        if self.x_tru > self.max:
            self.vx = rand.randrange(self.min_vel, -1, 1)
        elif self.x_tru < self.min:
            self.vx = rand.randrange(1, self.max_vel, 1)
        if self.y_tru > self.max:
            self.vy = rand.randrange(self.min_vel, -1, 1)
        elif self.y_tru < self.min:
            self.vy = rand.randrange(1, self.max_vel, 1)
        if self.z_tru > self.max:
            self.vz = rand.randrange(self.min_vel, -1, 1)
        elif self.z_tru < self.min:
            self.vz = rand.randrange(1, self.max_vel, 1)
        self.x_tru = self.x_tru + self.vx*self.dt
        self.y_tru = self.y_tru + self.vy*self.dt
        self.z_tru = self.z_tru + self.vz*self.dt

    def range(self, dronelist):
        agent = True
        for drone in dronelist:
            if agent:
                self.r = []
                agent = False
            else:
                dx = self.x_tru - drone.x_tru
                dy = self.y_tru - drone.y_tru
                dz = self.z_tru - drone.z_tru
                r = np.sqrt(dx**2 + dy**2 + dz**2)
                self.r.append(sensor_noise(r))


def main():
    window_size = 6
    os.system('clear')
    check = []
    # for i in range(6):
    num_anchors = 5
    drone_list = [Drone()]

    for i in range(num_anchors):
        drone_list.append(Drone())


    dataset_lengths = [1000000, 10000]
    updates = ['Training', 'Validation']
    num_int = []
    aps = []
    DS = DataStandardizer()

    for q in range(len(dataset_lengths)):
        os.system('clear')
        dadjoke = Dadjoke()
        print(dadjoke.joke, '\n')
        if q > 0:
            print("Number of interruptions in previous dataset: ", count_int,"\n")
            num_int.append(count_int)
            print("Average Number of interruptions per state: ", count_int/dataset_lengths[q-1], "\n")
            aps.append(count_int/dataset_lengths[q-1])
        # Count uinterruptions
        count_int = 0
        print("Generating ", updates[q], " Data with "+str(num_anchors) + " Anchor UAVs ...")
        with open('StandardizedData/'+str(num_anchors)+updates[q]+'_data.csv', 'w', encoding='UTF8') as f:
            writer = csv.writer(f)
            for i in tqdm(range(dataset_lengths[q])):
                data = []
                avail = []
                r = []
                if q != 2:
                    drone_list = [Drone()]
                    for i in range(num_anchors):
                        drone_list.append(Drone())
                i = 1
                r.append([])
                for drone in drone_list:
                    if i < 3:
                        i = i+1
                        pass
                    else:
                        r.append([])

                    drone.move()
                for drone in drone_list:
                    drone.noise()
                remove_drones = random.randint(0,1)

                if remove_drones == 1:
                    remove_drones += random.randint(0,1)

                if num_anchors-remove_drones < 3:
                    remove_drones = num_anchors-3

                for b in range(6):
                    avail.append([])
                    drone_list[0].range(drone_list)

                    for index in range(len(drone_list[0].r)):
                        if len(drone_list[0].r) - (index) > remove_drones:
                            avail[b].append(DroneData(drone_list[index+1].x, drone_list[index+1].y, drone_list[index+1].z, drone_list[0].r[index]))
                    # 1st anchor drone is where others positions are referenced to, so start recycling inputs with the second one
                    i = 0
                    # Count the number of communication interruptions for each data set
                    while len(avail[b]) < num_anchors:
                        avail[b].append(avail[b][i])
                        i = i+1
                count_int += i

                # For each drone available in the list, save all timesteps of range readings
                for i in range(len(avail[1])):
                    for item in avail:
                        data.append(item[i].r)
                # For each drone in the list, save all time steps of ENU estimates
                first = True
                for item in avail[0]:
                    if first:
                        x, y, z = [], [], []
                        first = False
                    else:
                        x.append(item.x - avail[0][0].x)
                        y.append(item.y - avail[0][0].y)
                        z.append(item.z - avail[0][0].z)
                # Determine targets, and reference origin
                target = [drone_list[0].x_tru - drone_list[1].x,
                        drone_list[0].y_tru - drone_list[1].y,
                        drone_list[0].z_tru - drone_list[1].z,
                        drone_list[1].x,
                        drone_list[1].y,
                        drone_list[1].z]
                inputs = list(itertools.chain(data, x, y, z))
                inputs = DS.StandardizeInputs(inputs, List = True)
                target = DS.StandardizeTargets(target, List=True)
                data = list(itertools.chain(inputs, target))

                if q == 0:
                    check.append(data)
                    a=True
                elif q == 1:
                    if data in check:
                        a = False
                    else:
                        a = True

                if a:
                    writer.writerow(data)


if __name__ == "__main__":
    main()
