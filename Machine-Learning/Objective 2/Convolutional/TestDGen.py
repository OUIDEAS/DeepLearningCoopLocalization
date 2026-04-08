import csv
import numpy as np
import random as rand
import csv
import os
from PosToGPS import *
import itertools
from dadjokes import Dadjoke
from haversine import inverse_haversine, Direction, Unit

def Simulate_GPS(x, y, z):
    # Used Ohio University's Geodetic Coordinates as a reference with height = 0
    reference_LLH= (39.324360, -82.101387)
    t1 = inverse_haversine(reference_LLH, y, Direction.NORTH, unit=Unit.METERS)
    t2 = inverse_haversine(t1, x, Direction.EAST, unit=Unit.METERS)
    pos = {
        "Latitude": t2[0] + float(np.random.normal(0, 1e-6,1)),
        "Longitude": t2[1] + float(np.random.normal(0, 1e-6,1)),
        "Height": z + float(np.random.normal(0, 1e-6,1))
    }
    return pos

class DroneData():
    def __init__(self, x, y, z, r):
        self.lat = x
        self.lon = y
        self.h = z
        self.r = r


class Drone(object):
    # Save the initial position of the drones
    def __init__(self, min_pos=0, max_pos=275):
        self.min = min_pos
        self.max = max_pos
        self.max_vel = 10
        self.min_vel = -10
        self.x = rand.randrange(self.min, self.max, 1) + rand.randrange(0, 100)/100
        self.y = rand.randrange(self.min, self.max, 1) + rand.randrange(0, 100)/100
        self.z = rand.randrange(self.min, self.max, 1) + rand.randrange(0, 100)/100
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
        self.x = self.x + self.vx*0.5
        self.y = self.y + self.vy*0.5
        self.z = self.z + self.vz*0.5

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

        r1 = dx1 * dx1 + dy1 * dy1 + dz1 * dz1
        r2 = dx2 * dx2 + dy2 * dy2 + dz2 * dz2
        r3 = dx3 * dx3 + dy3 * dy3 + dz3 * dz3
        r4 = dx4 * dx4 + dy4 * dy4 + dz4 * dz4
        r5 = dx5 * dx5 + dy5 * dy5 + dz5 * dz5
        r6 = dx6 * dx6 + dy6 * dy6 + dz6 * dz6
        r7 = dx7 * dx7 + dy7 * dy7 + dz7 * dz7
        r8 = dx8 * dx8 + dy8 * dy8 + dz8 * dz8
        r9 = dx9 * dx9 + dy9 * dy9 + dz9 * dz9
        r0 = dx0 * dx0 + dy0 * dy0 + dz0 * dz0

        # Simulate sensor noise from ultra-wideband modules
        r1 = float(np.sqrt(r1 + np.random.normal(0, r1*0.015,1)))
        r2 = float(np.sqrt(r2 + np.random.normal(0, r2*0.015,1)))
        r3 = float(np.sqrt(r3 + np.random.normal(0, r3*0.015,1)))
        r4 = float(np.sqrt(r4 + np.random.normal(0, r4*0.015,1)))
        r5 = float(np.sqrt(r5 + np.random.normal(0, r5*0.015,1)))
        r6 = float(np.sqrt(r6 + np.random.normal(0, r6*0.015,1)))
        r7 = float(np.sqrt(r7 + np.random.normal(0, r7*0.015,1)))
        r8 = float(np.sqrt(r8 + np.random.normal(0, r8*0.015,1)))
        r9 = float(np.sqrt(r9 + np.random.normal(0, r9*0.015,1)))
        r0 = float(np.sqrt(r0 + np.random.normal(0, r0*0.015,1)))
        self.r = [r1,r2,r3,r4,r5,r6,r7,r8,r9,r0]

def main():
    window_size = 6
    os.system('clear')
    check = []
    num_anchors = 10
    drone_list = [Drone()]

    for i in range(num_anchors):
        drone_list.append(Drone())


    os.system('clear')
    print('Dad Joke:')
    dadjoke = Dadjoke()
    print(dadjoke.joke, '\n')
    with open('Test_data.csv', 'w', encoding='UTF8') as f:
        writer = csv.writer(f)
        for i in range(100):
            data = []
            avail = []
            r = []
            x = []
            y = []
            z = []
            i = 1
            for drone in drone_list:
                drone.move()
                if i < 3:
                    i = i+1
                    pass
                else:
                    r.append([])
                    x.append([])
                    y.append([])
                    z.append([])

            for b in range(6):
                avail.append([])
                drone_list[0].range(drone_list[1],drone_list[2],drone_list[3],drone_list[4],drone_list[5],drone_list[6],drone_list[7],drone_list[8],drone_list[9],drone_list[10])
                for index in range(len(drone_list[0].r)):
                    if drone_list[0].r[index] < 300:
                        pos = Simulate_GPS(drone_list[index+1].x, drone_list[index+1].y, drone_list[index+1].z)
                        avail[b].append(DroneData(pos["Latitude"], pos["Longitude"], pos["Height"], drone_list[0].r[index]))
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
            for i in range(len(avail[1])-2):
                for item in avail:
                    e, n, u = GPS_to_ENU(item[1], item[i+2])
                    x[i].append(e)
                    y[i].append(n)
                    z[i].append(u)

            # Determine targets, and reference origin
            target = [drone_list[0].x - drone_list[1].x, drone_list[0].y - drone_list[1].y, drone_list[0].z - drone_list[1].z, drone_list[1].x, drone_list[1].y, drone_list[1].z]
            for i in range(len(x)):
                data = list(itertools.chain(data, x[i], y[i], z[i]))

            data = list(itertools.chain(data, target))
            writer.writerow(data)


if __name__ == "__main__":
    main()
