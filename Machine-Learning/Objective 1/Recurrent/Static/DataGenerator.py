import csv  
import math
import torch
import numpy as np
import random as rand
from torch.utils.data import Dataset, DataLoader
import csv
import os

class Drone():
    #Save the initial position of the drones
    def __init__(self, min, max):
        self.x = rand.randrange(min, max,1) + rand.randrange(0,99,1)/100
        self.y = rand.randrange(min, max,1) + rand.randrange(0,99,1)/100
        self.z = rand.randrange(min, max,1) + rand.randrange(0,99,1)/100
        self.vx = rand.randrange(-5,5,1)
        self.vy = rand.randrange(-5,5,1)
        self.vz = rand.randrange(-5,5,1)

    def move(self,x,y,z):
            if x > 5:
                self.vx = rand.randrange(-5,0,1)
            elif x < -5:
                self.vx = rand.randrange(0,5,1)
            if y > 5:
                self.vy = rand.randrange(-5,0,1)
            elif y < -5:
                self.vy = rand.randrange(0,5,1)
            if z > 5:
                self.vz = rand.randrange(-5,0,1)
            elif z < -5:
                self.vz = rand.randrange(0,5,1)
            self.x = x + self.vx*0.1
            self.y = y + self.vy*0.1
            self.z = z + self.vz*0.1

    def range(self, Drone1, Drone2, Drone3, Drone4, Drone5):
        dx1 = self.x - Drone1.x
        dx2 = self.x - Drone2.x
        dx3 = self.x - Drone3.x
        dx4 = self.x - Drone4.x
        dx5 = self.x - Drone5.x

        dy1 = self.y - Drone1.y
        dy2 = self.y - Drone2.y
        dy3 = self.y - Drone3.y
        dy4 = self.y - Drone4.y
        dy5 = self.y - Drone5.y

        dz1 = self.z - Drone1.z
        dz2 = self.z - Drone2.z
        dz3 = self.z - Drone3.z
        dz4 = self.z - Drone4.z
        dz5 = self.z - Drone5.z

        self.dx1 = dx1
        self.dy1 = dy1
        self.dz1 = dz1

        self.dx2 = dx2
        self.dy2 = dy2
        self.dz2 = dz2

        self.dx3 = dx3
        self.dy3 = dy3
        self.dz3 = dz3

        self.dx4 = dx4
        self.dy4 = dy4
        self.dz4 = dz4

        self.dx5 = dx5
        self.dy5 = dy5
        self.dz5 = dz5


        self.xa = dx1/math.sqrt(dx1**2 + dy1**2 + dz1**2)
        self.ya = dy1/math.sqrt(dx1**2 + dy1**2 + dz1**2)
        self.za = dz1/math.sqrt(dx1**2 + dy1**2 + dz1**2)

        self.r1 = math.sqrt(dx1*dx1 + dy1*dy1 + dz1*dz1)
        self.r2 = math.sqrt(dx2*dx2 + dy2*dy2 + dz2*dz2)
        self.r3 = math.sqrt(dx3*dx3 + dy3*dy3 + dz3*dz3)
        self.r4 = math.sqrt(dx4*dx4 + dy4*dy4 + dz4*dz4)
        self.r5 = math.sqrt(dx5*dx5 + dy5*dy5 + dz5*dz5)

os.system('clear')

anc_1 = Drone(-5,5)
anc_2 = Drone(-5,5)
anc_3 = Drone(-5,5)
anc_4 = Drone(-5,5)
anc_5 = Drone(-5,5)
drone = Drone(-5,5)

print("Generating Data...")

with open('test_data.csv', 'w', encoding='UTF8') as f:
    writer = csv.writer(f)
    for i in range(100):
        try:
            drone.move(drone.x, drone.y, drone.z)
            anc_1.range(drone, anc_2, anc_3, anc_4, anc_5)
            drone.range(anc_1, anc_2, anc_3, anc_4, anc_5)
            r1 = drone.r1 + rand.gauss(0,0.1)
            r2 = drone.r2 + rand.gauss(0,0.1)
            r3 = drone.r3 + rand.gauss(0,0.1)
            r4 = drone.r4 + rand.gauss(0,0.1)
            r5 = drone.r5 + rand.gauss(0,0.1)
            xa = drone.xa
            ya = drone.ya
            za = drone.za
            data = [r1, r2, r3, r4, r5, xa, ya, za, anc_1.x, anc_1.y, anc_1.z, drone.x, drone.y, drone.z, r1]
            # write the data
            writer.writerow(data)
            
        except:
            pass


with open('validation_data.csv', 'w', encoding='UTF8') as f:
    writer = csv.writer(f)
    for i in range(75000):
        try:
            anc_1.range(drone, anc_2, anc_3, anc_4, anc_5)
            drone.range(anc_1, anc_2, anc_3, anc_4, anc_5)
            r1 = drone.r1
            r2 = drone.r2
            r3 = drone.r3
            r4 = drone.r4
            r5 = drone.r5
            xa = drone.xa
            ya = drone.ya
            za = drone.za
            data = [r1, r2, r3, r4, r5, xa, ya, za, anc_1.x, anc_1.y, anc_1.z, drone.x, drone.y, drone.z, r1]
            # write the data
            writer.writerow(data)
        except:
            pass

with open('training_data.csv', 'w', encoding='UTF8') as f:
    writer = csv.writer(f)
    for i in range(50000):
        try:
            drone = Drone(-5,5)
            anc_1.range(drone, anc_2, anc_3, anc_4, anc_5)
            drone.range(anc_1, anc_2, anc_3, anc_4, anc_5)
            r1 = drone.r1
            r2 = drone.r2
            r3 = drone.r3
            r4 = drone.r4
            r5 = drone.r5
            xa = drone.xa
            ya = drone.ya
            za = drone.za
            data = [r1, r2, r3, r4, r5, xa, ya, za, anc_1.x, anc_1.y, anc_1.z, drone.x, drone.y, drone.z, r1]
            # write the data
            writer.writerow(data)
            
        except:
            pass



os.system('clear')
    
