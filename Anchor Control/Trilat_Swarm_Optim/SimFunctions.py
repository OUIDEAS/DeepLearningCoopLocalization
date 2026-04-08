import numpy as np
import random as rand
import csv
import math

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
        self.x = self.x_tru + float(np.random.normal(0,0.15,1))
        self.y = self.y_tru + float(np.random.normal(0,0.15,1))
        self.z = self.z_tru + float(np.random.normal(0,0.225,1))

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

    def range_to_individual(self, Drone):
        dx = self.x_tru - Drone.x_tru
        dy = self.y_tru - Drone.y_tru
        dz = self.z_tru - Drone.z_tru

        r = np.sqrt(dx**2 + dy**2 + dz**2)

        self.r.append(sensor_noise(r))

    def range(self, Dronelist):
        self.r = []
        for drone in Dronelist:
            dx = self.x_tru - drone.x_tru
            dy = self.y_tru - drone.y_tru
            dz = self.z_tru - drone.z_tru
            r=np.sqrt(dx**2 + dy**2 + dz**2)
            self.r.append(sensor_noise(r))

    def set_direction(self, command):
        self.vx, self.vy, self.vz = 0, 0, 0
        c = command

        if c==0:
            self.vx = -3
        elif c==1:
            self.vx = 3
        elif c==2:
            self.vy = -3
        elif c==3:
            self.vy = 3
        elif c==4:
            self.vz = -3
        elif c==5:
            self.vz = 3
        elif c==6:
            self.vx, self.vy, self.vz = 0, 0, 0
