import csv
import numpy as np
import random as rand
import csv
import os
from PosToGPS import *
import itertools
from haversine import inverse_haversine, Direction, Unit
import math
from dadjokes import Dadjoke

class DroneData():
    def __init__(self, x, y, z, r):
        self.x = x
        self.y = y
        self.z = z
        self.r = r

def sensor_noise(r):
    # Bandwidth
    b = 3.1*10**9
    # Speed of signal
    c = 3e8
    # Calculate a signal to noise ratio based on distance
    snr = abs(10 - r*(10-0.1)/600)
    sigsq = 1/(8*(math.pi**2)*(b**2)*snr)
    sig = 40*c*math.sqrt(sigsq)
    sensor = float(r + np.random.normal(0, sig,1))
    return sensor

class Drone(object):
    # Save the initial position of the drones
    def __init__(self, initial_pos=[rand.randrange(0,100), rand.randrange(0,100), rand.randrange(75,100)], anchor = True):
        self.x_tru = initial_pos[0]
        self.y_tru = initial_pos[1]
        self.z_tru = initial_pos[2]
        # Initialize instance variables
        self.r = None
        self.vx, self.vy, self.vz = 0, 0, 0
        self.anchor = anchor
        if anchor:
            self.ax, self.ay, self.az = 0, 0, 0

    def set_x_velocity(self, x):
        self.vx = x

    def set_y_velocity(self, x):
        self.vy = x

    def set_z_velocity(self, x):
        self.vz = x

    def reach_wp(self, wp):
        dx = self.x_tru-wp[0]
        dy = self.y_tru-wp[1]
        dz=  self.z_tru-wp[2]
        d = float(np.sqrt(dx**2+dy**2+dz**2))
        if d < 10:
            return 1
        else:
            return 0

    def set_accel(self, waypoint):
        x_w = waypoint[0]
        y_w = waypoint[1]
        z_w = waypoint[2]

        dx = x_w - self.x_tru
        dy = y_w - self.y_tru
        dz = z_w - self.z_tru

        P, D = 1, 4

        self.ax = P * dx + D *(0-self.vx)
        self.ay = P * dy + D *(0-self.vy)
        self.az = P * dz + D *(0-self.vz)


    def gps_noise(self):
        dirx = float(np.random.uniform(-1, 1, 1))
        diry = float(np.random.uniform(-1, 1, 1))
        dirz = float(np.random.uniform(-1, 1, 1))
        dir_mag = float(np.sqrt(dirx**2 + diry**2 + dirz**2))
        unitx = dirx/dir_mag
        unity = diry/dir_mag
        unitz = dirz/dir_mag
        # mag = float(np.random.uniform(0,4.9,1))
        # Cranked down the noise to help AC algo learn
        mag = float(np.random.uniform(0,0.5,1))
        self.x = self.x_tru + unitx*mag
        self.y = self.y_tru + unity*mag
        self.z = self.z_tru + unitz*mag

    def move(self, dt):
        self.vx = self.vx + self.ax*dt
        self.vy = self.vy + self.ay*dt
        self.vz = self.vz + self.az*dt
        if self.anchor:
            mv = 5
        else:
            mv = 2
        if self.vx > mv:
            self.vx = mv
        if self.vx < -mv:
            self.vx = -mv
        if self.vy > mv:
            self.vy = mv
        if self.vy < -mv:
            self.vy = -mv
        if self.vz > mv:
            self.vz = mv
        if self.vz < -mv:
            self.vz = -mv
        self.x_tru = self.x_tru + self.vx*dt
        self.y_tru = self.y_tru + self.vy*dt
        self.z_tru = self.z_tru + self.vz*dt

    def clear_ranges(self):
        self.r = []

    def range(self, Drone1, Drone2, Drone3, Drone4, Drone5, Drone6, Drone7, Drone8, Drone9, Drone0):
        dx1 = self.x_tru - Drone1.x_tru
        dx2 = self.x_tru - Drone2.x_tru
        dx3 = self.x_tru - Drone3.x_tru
        dx4 = self.x_tru - Drone4.x_tru
        dx5 = self.x_tru - Drone5.x_tru
        dx6 = self.x_tru - Drone6.x_tru
        dx7 = self.x_tru - Drone7.x_tru
        dx8 = self.x_tru - Drone8.x_tru
        dx9 = self.x_tru - Drone9.x_tru
        dx0 = self.x_tru - Drone0.x_tru

        dy1 = self.y_tru - Drone1.y_tru
        dy2 = self.y_tru - Drone2.y_tru
        dy3 = self.y_tru - Drone3.y_tru
        dy4 = self.y_tru - Drone4.y_tru
        dy5 = self.y_tru - Drone5.y_tru
        dy6 = self.y_tru - Drone6.y_tru
        dy7 = self.y_tru - Drone7.y_tru
        dy8 = self.y_tru - Drone8.y_tru
        dy9 = self.y_tru - Drone9.y_tru
        dy0 = self.y_tru - Drone0.y_tru

        dz1 = self.z_tru - Drone1.z_tru
        dz2 = self.z_tru - Drone2.z_tru
        dz3 = self.z_tru - Drone3.z_tru
        dz4 = self.z_tru - Drone4.z_tru
        dz5 = self.z_tru - Drone5.z_tru
        dz6 = self.z_tru - Drone6.z_tru
        dz7 = self.z_tru - Drone7.z_tru
        dz8 = self.z_tru - Drone8.z_tru
        dz9 = self.z_tru - Drone9.z_tru
        dz0 = self.z_tru - Drone0.z_tru

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
        ranges = []
        for r in [r1,r2,r3,r4,r5,r6,r7,r8,r9,r0]:
            ranges.append(r)

        # Simulate sensor noise from ultra-wideband modules

        self.r = []
        for r in ranges:
            if r == 500:
                self.r.append(r)
            else:
                self.r.append(sensor_noise(r))

    def set_direction(self, command):
        self.vx, self.vy, self.vz = 0, 0, 0
        c = command

        if c==0:
            self.vx = -4
        elif c==1:
            self.vx = -3
        elif c==2:
            self.vx = -2
        elif c==3:
            self.vx = -1
        elif c==4:
            self.vx = 1
        elif c==5:
            self.vx = 2
        elif c==6:
            self.vx = 3
        elif c==7:
            self.vx = 4

        elif c==8:
            self.vy = -4
        elif c==9:
            self.vy = -3
        elif c==10:
            self.vy = -2
        elif c==11:
            self.vy = -1
        elif c==12:
            self.vy = 1
        elif c==13:
            self.vy = 2
        elif c==14:
            self.vy = 3
        elif c==15:
            self.vy = 4

        elif c==16:
            self.vz = -4
        elif c==17:
            self.vz = -3
        elif c==18:
            self.vz = -2
        elif c==19:
            self.vz = -1
        elif c==20:
            self.vz = 1
        elif c==21:
            self.vz = 2
        elif c==22:
            self.vz = 3
        elif c==23:
            self.vz = 4

        elif c==24:
            self.vx, self.vy, self.vz = 0, 0, 0


    def set_direction_ac(self, command):
        self.vx, self.vy, self.vz = 0, 0, 0
        c = command

        if c==0:
            self.vx = -5
        elif c==1:
            self.vx = 5
        elif c==2:
            self.vy = -5
        elif c==3:
            self.vy = 5
        elif c==4:
            self.vz = -5
        elif c==5:
            self.vz = 5
        elif c==6:
            self.vx, self.vy, self.vz = 0, 0, 0



    def set_direction_fc(self, command):
        self.vx, self.vy, self.vz = 0, 0, 0
        c = command

        if c[0]==0:
            self.vx = -4
        elif c[0]==1:
            self.vx = -3
        elif c[0]==2:
            self.vx = -2
        elif c[0]==3:
            self.vx = -1
        elif c[0]==4:
            self.vx = 0
        elif c[0]==5:
            self.vx = 1
        elif c[0]==6:
            self.vx = 2
        elif c[0]==7:
            self.vx = 3
        elif c[0]==8:
            self.vx = 4

        if c[1]==0:
            self.vy = -4
        elif c[1]==1:
            self.vy = -3
        elif c[1]==2:
            self.vy = -2
        elif c[1]==3:
            self.vy = -1
        elif c[1]==4:
            self.vy = 0
        elif c[1]==5:
            self.vy = 1
        elif c[1]==6:
            self.vy = 2
        elif c[1]==7:
            self.vy = 3
        elif c[1]==8:
            self.vy = 4

        if c[2]==0:
            self.vz = -4
        elif c[2]==1:
            self.vz = -3
        elif c[2]==2:
            self.vz = -2
        elif c[2]==3:
            self.vz = -1
        elif c[2]==4:
            self.vz = 0
        elif c[2]==5:
            self.vz = 1
        elif c[2]==6:
            self.vz = 2
        elif c[2]==7:
            self.vz = 3
        elif c[2]==8:
            self.vz = 4

    def set_direction_B(self, command):
        if command[0].item() == 0:
            self.vx = -5
        else:
            self.vx = 5
        if command[1].item() == 0:
            self.vy = -5
        else:
            self.vy = 5
        if command[2].item() == 0:
            self.vz = -5
        else:
            self.vz = 5

    def set_mask(self):
        mask = []
        if self.x_tru > 90:
            mask.append(0)
        elif self.x_tru < 10:
            mask.append(1)

        if self.y_tru > 90:
            mask.append(2)
        elif self.y_tru < 10:
            mask.append(3)

        if self.x_tru > 90:
            mask.append(4)
        elif self.x_tru < 50:
            mask.append(5)

        return mask
