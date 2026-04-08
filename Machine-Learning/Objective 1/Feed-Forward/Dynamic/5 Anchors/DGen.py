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

    def range(self, Drone1, Drone2, Drone3, Drone4, Drone5):
        dx1 = self.x_tru - Drone1.x_tru
        dx2 = self.x_tru - Drone2.x_tru
        dx3 = self.x_tru - Drone3.x_tru
        dx4 = self.x_tru - Drone4.x_tru
        dx5 = self.x_tru - Drone5.x_tru
        
        dy1 = self.y_tru - Drone1.y_tru
        dy2 = self.y_tru - Drone2.y_tru
        dy3 = self.y_tru - Drone3.y_tru
        dy4 = self.y_tru - Drone4.y_tru
        dy5 = self.y_tru - Drone5.y_tru
        

        dz1 = self.z_tru - Drone1.z_tru
        dz2 = self.z_tru - Drone2.z_tru
        dz3 = self.z_tru - Drone3.z_tru
        dz4 = self.z_tru - Drone4.z_tru
        dz5 = self.z_tru - Drone5.z_tru
        

        r1 = np.sqrt(dx1 * dx1 + dy1 * dy1 + dz1 * dz1)
        r2 = np.sqrt(dx2 * dx2 + dy2 * dy2 + dz2 * dz2)
        r3 = np.sqrt(dx3 * dx3 + dy3 * dy3 + dz3 * dz3)
        r4 = np.sqrt(dx4 * dx4 + dy4 * dy4 + dz4 * dz4)
        r5 = np.sqrt(dx5 * dx5 + dy5 * dy5 + dz5 * dz5)
        ranges = []
        for r in [r1,r2,r3,r4,r5]:
            ranges.append(r)
        

        self.r = []
        for r in ranges:
            self.r.append(sensor_noise(r))


def main():
    window_size = 6
    os.system('clear')
    check = []
    num_anchors = 5
    drone_list = [Drone()]

    for i in range(num_anchors):
        drone_list.append(Drone())


    dataset_lengths = [2000000, 10000, 100]
    updates = ['Training', 'Validation', 'Test']
    num_int = []
    aps = []
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
        print("Generating ", updates[q], " Data ...")
        with open(updates[q]+'_data.csv', 'w', encoding='UTF8') as f:
            writer = csv.writer(f)
            for i in range(dataset_lengths[q]):
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

                for b in range(6):
                    avail.append([])
                    drone_list[0].range(drone_list[1],drone_list[2],drone_list[3],drone_list[4],drone_list[5])

                    

                    for index in range(len(drone_list[0].r)):
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

                data = list(itertools.chain(data, x, y, z, target))

                if q == 0:
                    check.append(data)
                if q == 1:
                    a = False
                    if data in check:
                        a = False
                    else:
                        a = True
                else:
                    a = True
                if a:
                    writer.writerow(data)

            
if __name__ == "__main__":
    main()
