import csv  
import math
import torch
import numpy as np
import random as rand
from torch.utils.data import Dataset, DataLoader
import csv
import os

#Sensor noise is modelled after the TeraBee TeraRanger Evo 60m. All drones spawn and remain within a 60m x 60m x 60m cubic area. Accuracy of these sensors are +/- 1.5% above 14m

class Drone():
    #Save the initial position of the drones
    def __init__(self, min = -30, max = 30):
        self.min = min
        self.max = max
        self.x = rand.randrange(min, max,1) + rand.randrange(0,100)/100
        self.y = rand.randrange(min, max,1) + rand.randrange(0,100)/100
        self.z = rand.randrange(min, max,1) + rand.randrange(0,100)/100
        self.vx = rand.randrange(-5,5,1)
        self.vy = rand.randrange(-5,5,1)
        self.vz = rand.randrange(-5,5,1)

    def move(self,x,y,z):
            if x > self.max:
                self.vx = rand.randrange(-5,0,1)
            elif x < self.min:
                self.vx = rand.randrange(0,5,1)
            if y > self.max:
                self.vy = rand.randrange(-5,0,1)
            elif y < self.min:
                self.vy = rand.randrange(0,5,1)
            if z > self.max:
                self.vz = rand.randrange(-5,0,1)
            elif z < self.min:
                self.vz = rand.randrange(0,5,1)
            self.x = x + self.vx*1
            self.y = y + self.vy*1
            self.z = z + self.vz*1

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

def main():
    os.system('clear')

    anc_1 = Drone()
    anc_2 = Drone()
    anc_3 = Drone()
    anc_4 = Drone()
    anc_5 = Drone()
    drone = Drone()

    print("Generating Test Data...")
    with open('test_data.csv', 'w', encoding='UTF8') as f:
        writer = csv.writer(f)
        for i in range(100):
            
            #Record information from previous states
            drone.range(anc_1, anc_2, anc_3, anc_4, anc_5)
            x_prev, y_prev, z_prev = drone.xa, drone.ya, drone.za
            #Vector for previous state information
            memory = [drone.r1, x_prev, y_prev, z_prev]

            #move the drone
            drone.move(drone.x, drone.y, drone.z)

            #Determine ranges
            anc_1.range(drone, anc_2, anc_3, anc_4, anc_5)
            drone.range(anc_1, anc_2, anc_3, anc_4, anc_5)

            #Add noise proportional to the distance squared
            r1 = drone.r1 + rand.gauss(0,1)*0.015*drone.r1
            r2 = drone.r2 + rand.gauss(0,1)*0.015*drone.r1
            r3 = drone.r3 + rand.gauss(0,1)*0.015*drone.r1
            r4 = drone.r4 + rand.gauss(0,1)*0.015*drone.r1
            r5 = drone.r5 + rand.gauss(0,1)*0.015*drone.r1

            #Targets
            xa = drone.xa
            ya = drone.ya
            za = drone.za

            #Position of the drone
            x = drone.x
            y = drone.y
            z = drone.z

            #Dummy vector to scale related inputs
            b = [r1, r2, r3, r4, r5, drone.vx, drone.vy, drone.vz]

            ranges = []
            m = max([abs(ele) for ele in b])
            #maxrange = [r1, r2, r3, r4, r5]
            for a in b:
                ranges.append((a/abs(a)) * math.sqrt(abs(a))/math.sqrt(m) if a!=0 else 0)

            # Data Returned is Range reading, maximum estimated error, ..., velocity_magnitude
            data = [ranges[0], ranges[1], ranges[2], ranges[3], ranges[4], ranges[5], ranges[6], ranges[7], memory[0], memory[1], memory[2], memory[3], xa, ya, za, anc_1.x, anc_1.y, anc_1.z, anc_2.x, anc_2.y, anc_2.z, anc_3.x, anc_3.y, anc_3.z, anc_4.x, anc_4.y, anc_4.z, anc_5.x, anc_5.y, anc_5.z, x, y, z, r1]
            # write the data
            writer.writerow(data)
    

    check = []
    os.system('clear')
    print("Generating Training Data...")
    with open('training_data.csv', 'w', encoding='UTF8') as f:
        writer = csv.writer(f)
        for i in range(50000):
            
            drone=Drone()
            #Record information from previous states
            drone.range(anc_1, anc_2, anc_3, anc_4, anc_5)
            x_prev, y_prev, z_prev = drone.xa, drone.ya, drone.za
            #Vector for previous state information
            memory = [drone.r1, x_prev, y_prev, z_prev]

            #move the drone
            drone.move(drone.x, drone.y, drone.z)

            #Determine ranges
            anc_1.range(drone, anc_2, anc_3, anc_4, anc_5)
            drone.range(anc_1, anc_2, anc_3, anc_4, anc_5)

            #Add noise proportional to the distance squared
            r1 = drone.r1 + rand.gauss(0,1)*0.015*drone.r1
            r2 = drone.r2 + rand.gauss(0,1)*0.015*drone.r1
            r3 = drone.r3 + rand.gauss(0,1)*0.015*drone.r1
            r4 = drone.r4 + rand.gauss(0,1)*0.015*drone.r1
            r5 = drone.r5 + rand.gauss(0,1)*0.015*drone.r1

            #Targets
            xa = drone.xa
            ya = drone.ya
            za = drone.za

            #Position of the drone
            x = drone.x
            y = drone.y
            z = drone.z

            #Dummy vector to scale related inputs
            b = [r1, r2, r3, r4, r5, drone.vx, drone.vy, drone.vz]

            ranges = []
            m = max([abs(ele) for ele in b])
            #maxrange = [r1, r2, r3, r4, r5]
            for a in b:
                ranges.append((a/abs(a)) * math.sqrt(abs(a))/math.sqrt(m) if a!=0 else 0)

            # Data Returned is Range reading, maximum estimated error, ..., velocity_magnitude
            data = [ranges[0], ranges[1], ranges[2], ranges[3], ranges[4], ranges[5], ranges[6], ranges[7], memory[0],
                    memory[1], memory[2], memory[3], xa, ya, za, anc_1.x, anc_1.y,
                    anc_1.z, anc_2.x, anc_2.y, anc_2.z, anc_3.x, anc_3.y, anc_3.z, anc_4.x, anc_4.y, anc_4.z, anc_5.x, anc_5.y, anc_5.z, x, y, z, r1]

            # write the data
            writer.writerow(data)


    os.system('clear')
    print("Generating Validation Data...")
    with open('validation_data.csv', 'w', encoding='UTF8') as f:
        writer = csv.writer(f)
        for i in range(20000):
            drone=Drone()
            #Record information from previous states
            drone.range(anc_1, anc_2, anc_3, anc_4, anc_5)
            x_prev, y_prev, z_prev = drone.xa, drone.ya, drone.za
            #Vector for previous state information
            memory = [drone.r1, x_prev, y_prev, z_prev]

            #move the drone
            drone.move(drone.x, drone.y, drone.z)

            #Determine ranges
            anc_1.range(drone, anc_2, anc_3, anc_4, anc_5)
            drone.range(anc_1, anc_2, anc_3, anc_4, anc_5)

            #Add noise proportional to the distance squared
            r1 = drone.r1 + rand.gauss(0,1)*0.015*drone.r1
            r2 = drone.r2 + rand.gauss(0,1)*0.015*drone.r1
            r3 = drone.r3 + rand.gauss(0,1)*0.015*drone.r1
            r4 = drone.r4 + rand.gauss(0,1)*0.015*drone.r1
            r5 = drone.r5 + rand.gauss(0,1)*0.015*drone.r1

            #Targets
            xa = drone.xa
            ya = drone.ya
            za = drone.za

            #Position of the drone
            x = drone.x
            y = drone.y
            z = drone.z

            #Dummy vector to scale related inputs
            b = [r1, r2, r3, r4, r5, drone.vx, drone.vy, drone.vz]

            ranges = []
            m = max([abs(ele) for ele in b])
            #maxrange = [r1, r2, r3, r4, r5]
            for a in b:
                ranges.append((a/abs(a)) * math.sqrt(abs(a))/math.sqrt(m) if a!=0 else 0)

            # Data Returned is Range reading, maximum estimated error, ..., velocity_magnitude
            data = [ranges[0], ranges[1], ranges[2], ranges[3], ranges[4], ranges[5], ranges[6], ranges[7], memory[0],
                    memory[1], memory[2], memory[3], xa, ya, za, anc_1.x, anc_1.y,
                    anc_1.z, anc_2.x, anc_2.y, anc_2.z, anc_3.x, anc_3.y, anc_3.z, anc_4.x, anc_4.y, anc_4.z, anc_5.x, anc_5.y, anc_5.z, x, y, z, r1]

            # Check to see if point generated for validation data has already been used or not.
            # If it has been used in training, set the boolean variable to False
            a=True
            for set in check:
                if data == set:
                    a=False
                else:
                    pass
                    
            if a:
                writer.writerow(data)


    os.system('clear')
    
if __name__ == "__main__":
    main()
