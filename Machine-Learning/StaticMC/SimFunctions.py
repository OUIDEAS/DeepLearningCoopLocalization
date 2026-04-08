import numpy as np
import random as rand
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
    snr = 10 - r*(10-0.001)/175
    try:
        sigsq = 1/(8*(math.pi**2)*(b**2)*snr)
        sig = 50*c*math.sqrt(sigsq)
    except:
        snr = 0.0000001
        sigsq = 1/(8*(math.pi**2)*(b**2)*snr)
        sig = 50*c*math.sqrt(sigsq)
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
        self.vx, self.vy, self.vz = rand.randrange(-4,4), rand.randrange(-4,4), rand.randrange(-4,4)

    # Added so ranging operations can be performed using ">>" operator
    def __rshift__(self, drone_list):
        if isinstance(drone_list, list):
            self.r = []
            anchor = False
            for drone in drone_list:
                if isinstance(drone, Drone): 
                    if anchor:
                        dx = self.x_tru - drone.x_tru
                        dy = self.y_tru - drone.y_tru
                        dz = self.z_tru - drone.z_tru
                        r = sensor_noise(np.sqrt(dx**2 + dy**2 + dz**2))
                        self.r.append(r)
                    else:
                        anchor = True
                else:
                    raise TypeError('Expected a list of class type \'Drone\'.')
        elif isinstance(drone_list, Drone):
            dx = self.x_tru - drone_list.x_tru
            dy = self.y_tru - drone_list.y_tru
            dz = self.z_tru - drone_list.z_tru
            r = np.sqrt(dx**2 + dy**2 + dz**2)
            return r
        else:
            raise TypeError('Expected either a list of \'Drone\' classes, or a singular \'Drone\' class.')

    def gps_noise(self):
        self.x = self.x_tru + float(np.random.normal(0,1.5,1))/10
        self.y = self.y_tru + float(np.random.normal(0,1.5,1))/10
        self.z = self.z_tru + float(np.random.normal(0,2.25,1))/10

    def move(self, dt):

        self.vx = self.vx
        self.vy = self.vy
        self.vz = self.vz

        self.x_tru = self.x_tru + self.vx*dt
        self.y_tru = self.y_tru + self.vy*dt
        self.z_tru = self.z_tru + self.vz*dt
        
    def range(self, drone_list):
        self.r = []
        anchor = False
        for drone in drone_list:
            if anchor:
                dx = self.x_tru - drone.x_tru
                dy = self.y_tru - drone.y_tru
                dz = self.z_tru - drone.z_tru
                r = np.sqrt(dx**2 + dy**2 + dz**2)
                self.r.append(sensor_noise(r))
            else:
                anchor = True
