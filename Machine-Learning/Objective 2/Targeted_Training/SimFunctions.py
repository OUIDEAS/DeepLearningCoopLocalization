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
    snr = 10 - r*(10-0.1)/300
    sigsq = 1/(8*(math.pi**2)*(b**2)*snr)
    sig = 40*c*math.sqrt(sigsq)
    sensor = float(r + np.random.normal(0, sig,1))
    return sensor

class Drone(object):
    # Save the initial position of the drones
    def __init__(self, initial_pos):
        self.x_tru = initial_pos[0]
        self.y_tru = initial_pos[1]
        self.z_tru = initial_pos[2]
        # Initialize instance variables
        self.r = None
        self.vx, self.vy, self.vz = 0, 0, 0

    def reach_wp(self, wp):
        dx = self.x-wp[0]
        dy = self.y-wp[1]
        dz=  self.z-wp[2]
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

        P, D = 3, 5

        self.ax = P * dx + D *(-1*self.vx)
        self.ay = P * dy + D *(-1*self.vy)
        self.az = P * dz + D *(-1*self.vz)


    def gps_noise(self):
        self.x = float(self.x_tru + np.random.normal(0, 0.25,1))
        self.y = float(self.y_tru + np.random.normal(0, 0.25,1))
        self.z = float(self.z_tru + np.random.normal(0, 0.25,1))

    def move(self, dt):
        self.vx = self.vx + self.ax * dt
        self.vy = self.vy + self.ay * dt
        self.vz = self.vz + self.az * dt
        self.x_tru = self.x_tru + self.vx*dt + float(np.random.normal(0,15,1))
        self.y_tru = self.y_tru + self.vy*dt + float(np.random.normal(0,15,1))
        self.z_tru = self.z_tru + self.vz*dt + float(np.random.normal(0,15,1))

    def clear_ranges(self):
        self.r = []

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
        ranges = []
        for r in [r1,r2,r3,r4,r5,r6,r7,r8,r9,r0]:
            if r < 300:
                ranges.append(r)
            else:
                ranges.append(500)
        # Simulate sensor noise from ultra-wideband modules

        self.r = []
        for r in ranges:
            if r == 500:
                self.r.append(r)
            else:
                self.r.append(sensor_noise(r))
