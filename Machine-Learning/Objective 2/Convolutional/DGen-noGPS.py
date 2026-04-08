import csv
import numpy as np
import random as rand
import csv
import os
from PosToGPS import *
import itertools
from dadjokes import Dadjoke
from haversine import inverse_haversine, Direction, Unit


# Sensor noise is modelled using a cramer rao lower bound for sensor error
def sensor_noise(r):
    # Bandwidth
    b = 3.1*10**9
    # Speed of signal
    c = 3e8
    # Calculate a signal to noise ratio based on distance
    snr = 10 - r*(10-0.1)/300
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
    def __init__(self, min_pos=0, max_pos=240):
        self.min = min_pos
        self.max = max_pos
        self.max_vel = 10
        self.min_vel = -10
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
        self.x = float(self.x_tru + np.random.normal(0, 0.25,1))
        self.y = float(self.y_tru + np.random.normal(0, 0.25,1))
        self.z = float(self.z_tru + np.random.normal(0, 0.25,1))

    def move(self):

        if self.x > self.max:
            self.vx = rand.randrange(self.min_vel, -1, 1)
        elif self.x < self.min:
            self.vx = rand.randrange(1, self.max_vel, 1)
        if self.y > self.max:
            self.vy = rand.randrange(self.min_vel, -1, 1)
        elif self.y < self.min:
            self.vy = rand.randrange(1, self.max_vel, 1)
        if self.z > self.max:
            self.vz = rand.randrange(self.min_vel, -1, 1)
        elif self.z < self.min:
            self.vz = rand.randrange(1, self.max_vel, 1)
        self.x_tru = self.x_tru + self.vx*0.5
        self.y_tru = self.y_tru + self.vy*0.5
        self.z_tru = self.z_tru + self.vz*0.5

    def range(self, Drone1, Drone2, Drone3, Drone4, Drone5, Drone6, Drone7, Drone8, Drone9, Drone0):
        dx1 = self.x - Drone1.x
        dx2 = self.x - Drone2.x
        dx3 = self.x - Drone3.x
        dx4 = self.x - Drone4.x
        dx5 = self.x - Drone5.x
        dx6 = self.x - Drone6.x
        dx7 = self.x - Drone7.x
        dx8 = self.x - Drone8.x
        dx9 = self.x - Drone9.x
        dx0 = self.x - Drone0.x

        dy1 = self.y - Drone1.y
        dy2 = self.y - Drone2.y
        dy3 = self.y - Drone3.y
        dy4 = self.y - Drone4.y
        dy5 = self.y - Drone5.y
        dy6 = self.y - Drone6.y
        dy7 = self.y - Drone7.y
        dy8 = self.y - Drone8.y
        dy9 = self.y - Drone9.y
        dy0 = self.y - Drone0.y

        dz1 = self.z - Drone1.z
        dz2 = self.z - Drone2.z
        dz3 = self.z - Drone3.z
        dz4 = self.z - Drone4.z
        dz5 = self.z - Drone5.z
        dz6 = self.z - Drone6.z
        dz7 = self.z - Drone7.z
        dz8 = self.z - Drone8.z
        dz9 = self.z - Drone9.z
        dz0 = self.z - Drone0.z

        r1 = np.sqrt(dx1 * dx1 + dy1 * dy1 + dz1 * dz1)
        r2 = np.sqrt(dx2 * dx2 + dy2 * dy2 + dz2 * dz2)
        r3 = np.sqrt(dx3 * dx3 + dy3 * dy3 + dz3 * dz3)
        r4 = np.sqrt(dx4 * dx4 + dy4 * dy4 + dz4 * dz4)
        r5 = np.sqrt(dx5 * dx5 + dy5 * dy5 + dz5 * dz5)
        r6 = np.sqrt(dx6 * dx6 + dy6 * dy6 + dz6 * dz6)
        r7 = np.sqrt(dx7 * dx7 + dy7 * dy7 + dz7 * dz7)
        r8 = np.sqrt(dx8 * dx8 + dy8 * dy8 + dz8 * dz8)
        r9 = np.sqrt(dx9 * dx9 + dy9 * dy9 + dz9 * dz9)
        r0 = np.sqrt(dx0 * dx0 + dy0 * dy0 + dz0 * dz0)

        # Simulate sensor noise from ultra-wideband modules
        r1 = sensor_noise(r1)
        r2 = sensor_noise(r2)
        r3 = sensor_noise(r3)
        r4 = sensor_noise(r4)
        r5 = sensor_noise(r5)
        r6 = sensor_noise(r6)
        r7 = sensor_noise(r7)
        r8 = sensor_noise(r8)
        r9 = sensor_noise(r9)
        r0 = sensor_noise(r0)
        self.r = [r1,r2,r3,r4,r5,r6,r7,r8,r9,r0]

def main():
    window_size = 6
    os.system('clear')
    check = []
    num_anchors = 10
    drone_list = [Drone()]

    for i in range(num_anchors):
        drone_list.append(Drone())


    dataset_lengths = [500000, 10000, 100]
    updates = ['Training', 'Validation', 'Test']
    for q in range(len(dataset_lengths)):

        os.system('clear')
        print('Dad Joke:')
        dadjoke = Dadjoke()
        print(dadjoke.joke, '\n')
        if q > 0:
            print("Number of interruptions in previous dataset: ", count_int,"\n")
            print("Average Number of interruptions per state: ", count_int/dataset_lengths[q-1], "\n")
        # Count uinterruptions
        count_int = 0
        print("Generating ", updates[q], " Data ...")
        with open(updates[q]+'_data.csv', 'w', encoding='UTF8') as f:
            writer = csv.writer(f)
            for i in range(dataset_lengths[q]):
                data = []
                avail = []
                r = []
                x = []
                y = []
                z = []
                if q != 2:
                    drone_list = [Drone()]
                    for i in range(num_anchors):
                        drone_list.append(Drone())
                i = 1
                for drone in drone_list:
                    if i < 3:
                        i = i+1
                        pass
                    else:
                        r.append([])
                        x.append([])
                        y.append([])
                        z.append([])

                    drone.move()

                for b in range(6):
                    avail.append([])
                    drone_list[0].range(drone_list[1],drone_list[2],drone_list[3],drone_list[4],drone_list[5],drone_list[6],drone_list[7],drone_list[8],drone_list[9],drone_list[10])
                    for index in range(len(drone_list[0].r)):
                        if drone_list[0].r[index] < 300:
                            for drone in drone_list:
                                drone.noise()
                            avail[b].append(DroneData(drone_list[index+1].x, drone_list[index+1].y, drone_list[index+1].z, drone_list[0].r[index]))
                    # 1st anchor drone is where others positions are referenced to, so start recycling inputs with the second one
                    i = 0
                    # Count the number of communication interruptions for each data set
                    count_int = count_int + num_anchors - len(avail[b])
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
                        x[i].append(item[i+1].x - item[0].x)
                        y[i].append(item[i+1].y - item[0].y)
                        z[i].append(item[i+1].z - item[0].z)
                # Determine targets, and reference origin
                target = [drone_list[0].x_tru - drone_list[1].x_tru, drone_list[0].y_tru - drone_list[1].y_tru, drone_list[0].z_tru - drone_list[1].z_tru, drone_list[1].x_tru, drone_list[1].y_tru, drone_list[1].z_tru]
                for i in range(len(x)-1):
                    data = list(itertools.chain(data, x[i], y[i], z[i]))

                data = list(itertools.chain(data, target))

                if q ==0:
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
