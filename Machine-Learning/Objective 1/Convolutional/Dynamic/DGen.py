import csv  
import math
import torch
import numpy as np
import random as rand
from torch.utils.data import Dataset, DataLoader
import csv
import os

# Sensor noise is modelled after the TeraBee TeraRanger Evo 60m. All drones spawn and remain within a 60m x 60m x 60m
# cubic area. Accuracy of these sensors are +/- 1.5% above 14m


class Drone(object):
    # Save the initial position of the drones
    def __init__(self, min_pos=-30, max_pos=30):
        self.min = min_pos
        self.max = max_pos
        self.max_vel = 5
        self.min_vel = -5
        self.x = rand.randrange(self.min, self.max, 1) + rand.randrange(0, 100)/100
        self.y = rand.randrange(self.min, self.max, 1) + rand.randrange(0, 100)/100
        self.z = rand.randrange(self.min, self.max, 1) + rand.randrange(0, 100)/100
        self.vx = rand.randrange(self.min_vel, self.max_vel, 2)
        self.vy = rand.randrange(self.min_vel, self.max_vel, 2)
        self.vz = rand.randrange(self.min_vel, self.max_vel, 2)
        if self.vx == 0:
            self.vx = 5
        if self.vy == 0:
            self.vy = 5
        if self.vz == 0:
            self.vz = 5

        # Initialize instance variables
        self.r1 = None
        self.r2 = None
        self.r3 = None
        self.r4 = None
        self.r5 = None
        self.r6 = None
        self.r7 = None
        self.r8 = None
        self.r9 = None
        self.r0 = None

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
        self.x = self.x + self.vx*0.1
        self.y = self.y + self.vy*0.1
        self.z = self.z + self.vz*0.1

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
        self.r1 = float(np.sqrt(r1 + np.random.normal(0, r1*0.015,1)))
        self.r2 = float(np.sqrt(r2 + np.random.normal(0, r2*0.015,1)))
        self.r3 = float(np.sqrt(r3 + np.random.normal(0, r3*0.015,1)))
        self.r4 = float(np.sqrt(r4 + np.random.normal(0, r4*0.015,1)))
        self.r5 = float(np.sqrt(r5 + np.random.normal(0, r5*0.015,1)))
        self.r6 = float(np.sqrt(r6 + np.random.normal(0, r6*0.015,1)))
        self.r7 = float(np.sqrt(r7 + np.random.normal(0, r7*0.015,1)))
        self.r8 = float(np.sqrt(r8 + np.random.normal(0, r8*0.015,1)))
        self.r9 = float(np.sqrt(r9 + np.random.normal(0, r9*0.015,1)))
        self.r0 = float(np.sqrt(r0 + np.random.normal(0, r0*0.015,1)))
       


def main():
    os.system('clear')

    anc_1 = Drone()
    anc_2 = Drone()
    anc_3 = Drone()
    anc_4 = Drone()
    anc_5 = Drone()
    anc_6 = Drone()
    anc_7 = Drone()
    anc_8 = Drone()
    anc_9 = Drone()
    anc_0 = Drone()
    drone = Drone()
    print("Generating Test Data...")
    with open('test_data.csv', 'w', encoding='UTF8') as f:
        writer = csv.writer(f)
        for i in range(100):
            # Move the drone
            drone.move()
            anc_1.move()
            anc_2.move()
            anc_3.move()
            anc_4.move()
            anc_5.move()
            anc_6.move()
            anc_7.move()
            anc_8.move()
            anc_9.move()
            anc_0.move()

            # Determine ranges
            anc_1.range(drone, anc_2, anc_3, anc_4, anc_5, anc_6, anc_7, anc_8, anc_9, anc_0)
            drone.range(anc_1, anc_2, anc_3, anc_4, anc_5, anc_6, anc_7, anc_8, anc_9, anc_0)

            r1 = drone.r1
            r2 = drone.r2
            r3 = drone.r3
            r4 = drone.r4
            r5 = drone.r5
            r6 = drone.r6
            r7 = drone.r7
            r8 = drone.r8
            r9 = drone.r9
            r0 = drone.r0

            # Data Returned is Range reading, maximum estimated error, ..., velocity_magnitude
            data = [r1, r2, r3, r4, r5, r6, r7, r8, r9, r0, anc_2.x - anc_1.x, anc_2.y - anc_1.y, anc_2.z - anc_1.z,
                    anc_3.x - anc_1.x, anc_3.y - anc_1.y, anc_3.z - anc_1.z, anc_4.x - anc_1.x, anc_4.y - anc_1.y,
                    anc_4.z - anc_1.z, anc_5.x - anc_1.x, anc_5.y - anc_1.y, anc_5.z - anc_1.z, anc_6.x - anc_1.x,
                    anc_6.y - anc_1.y, anc_6.z - anc_1.z, anc_7.x - anc_1.x, anc_7.y - anc_1.y, anc_7.z - anc_1.z,
                    anc_8.x - anc_1.x, anc_8.y - anc_1.y, anc_8.z - anc_1.z, anc_9.x - anc_1.x, anc_9.y - anc_1.y, 
                    anc_9.z - anc_1.z, anc_0.x - anc_1.x, anc_0.y - anc_1.y, anc_0.z - anc_1.z, drone.x - anc_1.x, 
                    drone.y - anc_1.y, drone.z - anc_1.z, anc_1.x, anc_1.y, anc_1.z]

            # write the data
            writer.writerow(data)

    check = []
    os.system('clear')
    print("Generating Training Data...")
    with open('training_data.csv', 'w', encoding='UTF8') as f:
        writer = csv.writer(f)
        for i in range(500):
            anc_1 = Drone()
            anc_2 = Drone()
            anc_3 = Drone()
            anc_4 = Drone()
            anc_5 = Drone()
            anc_6 = Drone()
            anc_7 = Drone()
            anc_8 = Drone()
            anc_9 = Drone()
            anc_0 = Drone()
            drone = Drone()

            for a in range(100):
                anc_1.move()
                anc_2.move()
                anc_3.move()
                anc_4.move()
                anc_5.move()
                anc_6.move()
                anc_7.move()
                anc_8.move()
                anc_9.move()
                anc_0.move()
                drone.move()
                # Determine ranges
                anc_1.range(drone, anc_2, anc_3, anc_4, anc_5, anc_6, anc_7, anc_8, anc_9, anc_0)
                drone.range(anc_1, anc_2, anc_3, anc_4, anc_5, anc_6, anc_7, anc_8, anc_9, anc_0)

                r1 = drone.r1
                r2 = drone.r2
                r3 = drone.r3
                r4 = drone.r4
                r5 = drone.r5
                r6 = drone.r6
                r7 = drone.r7
                r8 = drone.r8
                r9 = drone.r9
                r0 = drone.r0

                # Data Returned is Range reading, maximum estimated error, ..., velocity_magnitude
                data = [r1, r2, r3, r4, r5, r6, r7, r8, r9, r0, anc_2.x - anc_1.x, anc_2.y - anc_1.y, anc_2.z - anc_1.z,
                        anc_3.x - anc_1.x, anc_3.y - anc_1.y, anc_3.z - anc_1.z, anc_4.x - anc_1.x, anc_4.y - anc_1.y,
                        anc_4.z - anc_1.z, anc_5.x - anc_1.x, anc_5.y - anc_1.y, anc_5.z - anc_1.z, anc_6.x - anc_1.x,
                        anc_6.y - anc_1.y, anc_6.z - anc_1.z, anc_7.x - anc_1.x, anc_7.y - anc_1.y, anc_7.z - anc_1.z,
                        anc_8.x - anc_1.x, anc_8.y - anc_1.y, anc_8.z - anc_1.z, anc_9.x - anc_1.x, anc_9.y - anc_1.y, 
                        anc_9.z - anc_1.z, anc_0.x - anc_1.x, anc_0.y - anc_1.y, anc_0.z - anc_1.z, drone.x - anc_1.x, 
                        drone.y - anc_1.y, drone.z - anc_1.z, anc_1.x, anc_1.y, anc_1.z]

                # write the data
                writer.writerow(data)

    os.system('clear')
    print("Generating Validation Data...")
    with open('validation_data.csv', 'w', encoding='UTF8') as f:
        writer = csv.writer(f)
        for i in range(2000):
            anc_1 = Drone()
            anc_2 = Drone()
            anc_3 = Drone()
            anc_4 = Drone()
            anc_5 = Drone()
            anc_6 = Drone()
            anc_7 = Drone()
            anc_8 = Drone()
            anc_9 = Drone()
            anc_0 = Drone()
            drone = Drone()

            # Determine ranges
            anc_1.range(drone, anc_2, anc_3, anc_4, anc_5, anc_6, anc_7, anc_8, anc_9, anc_0)
            drone.range(anc_1, anc_2, anc_3, anc_4, anc_5, anc_6, anc_7, anc_8, anc_9, anc_0)

            r1 = drone.r1
            r2 = drone.r2
            r3 = drone.r3
            r4 = drone.r4
            r5 = drone.r5
            r6 = drone.r6
            r7 = drone.r7
            r8 = drone.r8
            r9 = drone.r9
            r0 = drone.r0

            # Data Returned is Range reading, maximum estimated error, ..., velocity_magnitude
            data = [r1, r2, r3, r4, r5, r6, r7, r8, r9, r0, anc_2.x - anc_1.x, anc_2.y - anc_1.y, anc_2.z - anc_1.z,
                    anc_3.x - anc_1.x, anc_3.y - anc_1.y, anc_3.z - anc_1.z, anc_4.x - anc_1.x, anc_4.y - anc_1.y,
                    anc_4.z - anc_1.z, anc_5.x - anc_1.x, anc_5.y - anc_1.y, anc_5.z - anc_1.z, anc_6.x - anc_1.x,
                    anc_6.y - anc_1.y, anc_6.z - anc_1.z, anc_7.x - anc_1.x, anc_7.y - anc_1.y, anc_7.z - anc_1.z,
                    anc_8.x - anc_1.x, anc_8.y - anc_1.y, anc_8.z - anc_1.z, anc_9.x - anc_1.x, anc_9.y - anc_1.y, 
                    anc_9.z - anc_1.z, anc_0.x - anc_1.x, anc_0.y - anc_1.y, anc_0.z - anc_1.z, drone.x - anc_1.x, 
                    drone.y - anc_1.y, drone.z - anc_1.z, anc_1.x, anc_1.y, anc_1.z]

            # Check to see if point generated for validation data has already been used or not.
            # If it has been used in training, set the boolean variable to False
            a = True
            for item in check:
                if data == item:
                    a = False
                else:
                    pass
                    
            if a:
                writer.writerow(data)

    os.system('clear')


if __name__ == "__main__":

    main()
