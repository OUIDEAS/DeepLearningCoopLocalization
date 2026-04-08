import numpy as np
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
    snr = 1 - r*(1-0.001)/175
    try:
        sigsq = 1/(8*(math.pi**2)*(b**2)*snr)
        sig = 65*c*math.sqrt(sigsq)
    except:
        snr = 0.0000000000000001
        sigsq = 1/(8*(math.pi**2)*(b**2)*snr)
        sig = 65*c*math.sqrt(sigsq)
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
        self.vxs, self.vys, self.vzs = 0, 0, 0
        self.ax, self.ay, self.az = 0, 0, 0
        self.x_drift = float(np.random.uniform(0.5,1,1))
        self.y_drift = float(np.random.uniform(0.5,1,1))
        self.z_drift = float(np.random.uniform(0.5,1,1))
        self.max_v = 3
        self.gps_noise()

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

        self.ax = P * dx + D *(-1*self.vx)
        self.ay = P * dy + D *(-1*self.vy)
        self.az = P * dz + D *(-1*self.vz)


    def gps_noise(self):
        self.x = self.x_tru + float(np.random.normal(0,0.15,1))
        self.y = self.y_tru + float(np.random.normal(0,0.15,1))
        self.z = self.z_tru + float(np.random.normal(0,0.225,1))

    def move(self, dt):
        
        self.vx = self.vx + self.ax*dt
        self.vy = self.vy + self.ay*dt
        self.vz = self.vz + self.az*dt

        for x in [self.vx, self.vy, self.vz]:
            if abs(x) > self.max_v:
                x = self.max_v * x/abs(x)

        self.x_tru = self.x_tru + self.vx*dt + 0.5*self.ax*dt**2
        self.y_tru = self.y_tru + self.vy*dt + 0.5*self.ay*dt**2
        self.z_tru = self.z_tru + self.vz*dt + 0.5*self.az*dt**2

    def clear_ranges(self):
        self.r = []

    def range(self, drone_list):
        self.r = []
        anchor = False
        for drone in drone_list:
            if anchor:
                dx = self.x_tru - drone.x_tru
                dy = self.y_tru - drone.y_tru
                dz = self.z_tru - drone.z_tru
                self.r.append(sensor_noise(np.sqrt(dx**2 + dy**2 + dz**2)))
            else:
                anchor = True
