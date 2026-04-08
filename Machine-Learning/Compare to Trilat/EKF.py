import csv
import torch
import numpy as np
import random as rand
import csv
import os
import itertools
from haversine import inverse_haversine, Direction, Unit
import math
from dadjokes import Dadjoke
from SimFunctions import *
from OLSsolver import *
import pandas as pd
import matplotlib.pyplot as plt

def smooth(vals):
    unfiltered = pd.DataFrame(
        {'unfiltered': vals})
    return unfiltered.ewm(com=100).mean()

class EKF():
    def __init__(self, state, dt):
        # State = [x, y, z, vx, vy, vz]
        # Inputs = [ax, ay, az]
        self.x_k = state
        self.x = self.x_k
        self.dt = dt
        self.A = np.matrix([[1, 0, 0, dt, 0, 0, 1/2*dt**2, 0, 0],
                            [0, 1, 0, 0, dt, 0, 0, 1/2*dt**2, 0],
                            [0, 0, 1, 0, 0, dt, 0, 0, 1/2*dt**2],
                            [0, 0, 0, 1, 0, 0, dt, 0, 0],
                            [0, 0, 0, 0, 1, 0, 0, dt, 0],
                            [0, 0, 0, 0, 0, 1, 0, 0, dt],
                            [0, 0, 0, 0, 0, 0, 1, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 1, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 1]])
        
        Q = np.matrix([[dt**5/20, 0, 0, dt**4/8, 0, 0, dt**3/6, 0, 0],
                       [0, dt**5/20, 0, 0, dt**4/8, 0, 0, dt**3/6, 0],
                       [0, 0, dt**5/20, 0, 0, dt**4/8, 0, 0, dt**3/6],
                       [dt**4/8, 0, 0, dt**3/6, 0, 0, dt**2/2, 0, 0],
                       [0, dt**4/8, 0, 0, dt**3/6, 0, 0, dt**2/2, 0],
                       [0, 0, dt**4/8, 0, 0, dt**3/6, 0, 0, dt**2/2],
                       [dt**3/6, 0, 0, dt**2/2, 0, 0, dt, 0, 0],
                       [0, dt**3/6, 0, 0, dt**2/2, 0, 0, dt, 0],
                       [0, 0, dt**3/6, 0, 0, dt**2/2, 0, 0, dt]])
        self.Q = Q * 0.0018639 # Generic covariance matrix for Inertial Navigation * Standard Deviation of IMU
        self.P = np.eye(9)

    def predict(self, acc):
        self.x_prev = np.copy(self.x_k)
        self.P_prev = np.copy(self.P)
        x = self.x_k
        x[6] = acc[0]
        x[7] = acc[1]
        x[8] = acc[2]
        self.x = self.A*x
        self.P = self.A*self.P*self.A.T + self.Q
        self.x_k = self.x

    def format_data(self, ancs):
        rho = []
        points = [[0, 0, 0]]
        for p in ancs:
            rho.append(p[3])
        
        for i in range(len(ancs)-1):
            points.append([ancs[i+1][0] - ancs[0][0],
                           ancs[i+1][1] - ancs[0][1],
                           ancs[i+1][2] - ancs[0][2]])

        return np.array(points), np.array(rho)

    def observe(self, Ancs):
        self.SensorError(Ancs)
        Skip = True
        # Create the Jacobian Matrix for the Observation Step
        i = 1
        for p in Ancs:
            if Skip:
                r = float(np.sqrt((self.x[0] - Ancs[0][0])**2 + (self.x[1] - Ancs[0][1])**2 + (self.x[2] - Ancs[0][2])**2))
                pdx = -1*float((p[0] - self.x[0])/p[3])
                pdy = -1*float((p[1] - self.x[1])/p[3])
                pdz = -1*float((p[2] - self.x[2])/p[3])
                self.H = np.matrix([[float(pdx), float(pdy), float(pdz), 0, 0, 0, 0, 0, 0]], dtype = np.double)
                self.rho = np.array([[r]], dtype = np.double)
                self.z = np.array([p[3]], dtype = np.double)
                Skip = False
            else: 
                r = float(np.sqrt((self.x[0] - Ancs[i][0])**2 + (self.x[1] - Ancs[i][1])**2 + (self.x[2] - Ancs[i][2])**2))     
                i+=1         
                pdx = -1*float((p[0] - self.x[0])/p[3])
                pdy = -1*float((p[1] - self.x[1])/p[3])
                pdz = -1*float((p[2] - self.x[2])/p[3])
                self.H = np.append(self.H, [[float(pdx), float(pdy), float(pdz), 0, 0, 0, 0, 0, 0]], axis = 0)
                self.rho = np.append(self.rho, np.array([[r]]), axis = 0)
                self.z = np.append(self.z, np.array([p[3]]), axis = 0)
        self.z = np.transpose(np.array([self.z]))
                

    def update(self):
        self.Kalman = self.P*self.H.T*np.linalg.inv(self.H*self.P*self.H.T+self.R)
        self.x_k = self.x + self.Kalman * np.subtract(self.z, self.rho)
        self.P = (np.eye((self.Kalman*self.H).shape[1]) - self.Kalman*self.H)*self.P

    def SensorError(self, Ancs):
        b = 3.1*10**9
        c = 3e8
        self.R = np.eye(len(Ancs))
        for i in range(len(Ancs)):
            try:
                r = Ancs[i][2]
                snr = 10 - r*(10-0.1)/600
                sigsq = 1/(8*(math.pi**2)*(b**2)*snr)
                sig = 40*c*math.sqrt(sigsq)
                r_err = float(sig*sig)
                self.R[i][i] = r_err
            except:
                snr = 0.00000000000000000000000001
                sigsq = 1/(8*(math.pi**2)*(b**2)*snr)
                sig = 40*c*math.sqrt(sigsq)
                r_err = float(sig*sig)
                self.R[i][i] = r_err

class KalmanFilter():
    def __init__(self, x):
        self.x_k = x
        self.p = 0.01
        self.q = 0.005

    def predict(self):
        self.x = self.x_k
        self.p = self.p+self.q

    def update(self, y):
        r = self.SensorError(y)
        K = self.p * (self.p + r)**-1
        self.x_k = self.x + K*(y-self.x)
        self.p = (1-K)*self.p

    def SensorError(self, r):
        b = 3.1*10**9
        c = 3e8
        try:
            snr = 10 - r*(10-0.1)/300
            sigsq = 1/(8*(math.pi**2)*(b**2)*snr)
            sig = 40*c*math.sqrt(sigsq)
            r_err = float(sig*sig)
        except:
            snr = 0.00000000000000000000000001
            sigsq = 1/(8*(math.pi**2)*(b**2)*snr)
            sig = 40*c*math.sqrt(sigsq)
            r_err = float(sig*sig)
        return r_err


def gen_plots():
    wp = []
    anc_alt = 75
    max_alt = 100
    # Agents waypoints first
    wp.append([[-5,0,max_alt],[0,10, max_alt],[0, 10, 10],[0, 40, 10],[0,40,max_alt], [-5,50,max_alt]])

    # Anchors 1, 2, 3,..., 10
    wp.append([[0,0,max_alt],
               [0,10, anc_alt],
               [0,10, anc_alt],
               [0,10, anc_alt],
               [0, 10, max_alt],
               [0,10, max_alt]])

    wp.append([[10,0,max_alt],
               [max_alt/2,10, anc_alt-20],
               [max_alt/2,10, anc_alt-20],
               [max_alt/2,10, anc_alt-20],
               [30, 20, max_alt/3],
               [5, 50, max_alt/3]])

    wp.append([[20,0,max_alt],
               [max_alt,10, anc_alt+20],
               [max_alt,10, anc_alt+20],
               [max_alt,10, anc_alt+20],
               [60, 20, max_alt/3],
               [10, 50, max_alt/3]])

    wp.append([[0,10,max_alt],
               [0,max_alt/2, anc_alt-20],
               [0,max_alt/2, anc_alt-20],
               [0,max_alt/2, anc_alt-20],
               [0, max_alt/2, max_alt/3],
               [0,60, max_alt/3]])

    wp.append([[10,10, max_alt],
               [max_alt/2,max_alt/2, anc_alt+20],
               [max_alt/2,max_alt/2, anc_alt+20],
               [max_alt/2,max_alt/2, anc_alt+20],
               [30, max_alt/2, max_alt],
               [5,max_alt, max_alt]])

    wp.append([[20,10,max_alt],
               [max_alt,max_alt/2, anc_alt-20],
               [max_alt,max_alt/2, anc_alt-20],
               [max_alt,max_alt/2, anc_alt-20],
               [60, max_alt/2, max_alt],
               [10,60, max_alt]])

    wp.append([[0,4,max_alt],
               [0,max_alt, anc_alt+20],
               [0,max_alt, anc_alt+25],
               [0,max_alt, anc_alt+25],
               [0, max_alt, max_alt/3],
               [0,54, max_alt/3]])

    wp.append([[0,7,max_alt],
               [max_alt/2,max_alt, anc_alt-20],
               [max_alt/2,max_alt, anc_alt-25],
               [max_alt/2,max_alt, anc_alt-25],
               [max_alt/2, max_alt, max_alt/3],
               [0,max_alt/2, max_alt/3]])

    wp.append([[20,4,max_alt],
               [max_alt,max_alt, anc_alt+20],
               [max_alt,max_alt, anc_alt+20],
               [max_alt,max_alt, anc_alt+20],
               [max_alt,max_alt, max_alt/3],
               [10,max_alt/2, max_alt/3]])

    wp.append([[20,7,max_alt],
               [max_alt,max_alt, anc_alt-20],
               [max_alt/2,max_alt*0.75, anc_alt-20],
               [max_alt/2,max_alt*0.75, anc_alt-20],
               [60, max_alt*0.75, max_alt/3],
               [10,max_alt*0.75, max_alt/3]])

    loss = torch.nn.L1Loss()
    waypoints = wp
    drone_list = [Drone(waypoints[0][0])]
    
    # Create List of Drones
    for i in range(len(waypoints)-1):
        drone_list.append(Drone(waypoints[i+1][0]))

    # Set drone headings
    for i in range(len(waypoints)):
        drone_list[i].set_accel(waypoints[i][1])
        drone_list[i].gps_noise()

    ekfx, ekfy, ekfz = [], [], []
    tL = []
    trux, truy, truz = [], [], []
    dt = 0.1

    state = np.matrix([[drone_list[0].x], 
                       [drone_list[0].y], 
                       [drone_list[0].z], 
                       [drone_list[0].vx], 
                       [drone_list[0].vy], 
                       [drone_list[0].vz], 
                       [drone_list[0].ax], 
                       [drone_list[0].ay], 
                       [drone_list[0].az]])
    
    my_ekf = EKF(state, dt)
    end_sim = False
    w=1
    drone_list[0].range(drone_list[1],drone_list[2],drone_list[3],drone_list[4],drone_list[5],drone_list[6],drone_list[7],drone_list[8],drone_list[9],drone_list[10])
    kf = []
    for i in drone_list[0].r:
        kf.append(KalmanFilter(i))
    while not end_sim:
        criterion = 0
        for i in range(len(drone_list)):
            criterion = criterion + drone_list[i].reach_wp(waypoints[i][w])

        if criterion == 11:
            w = w + 1
            print("WayPoint Reached")

        if w == len(waypoints[0]):
            w = 5
            end_sim = True

        for i in range(len(waypoints)):
            drone_list[i].set_accel(waypoints[i][w])

        for drone in drone_list:
            drone.move(dt)
            drone.gps_noise()

        drone_list[0].range(drone_list[1],drone_list[2],drone_list[3],drone_list[4],drone_list[5],drone_list[6],drone_list[7],drone_list[8],drone_list[9],drone_list[10])

        vel_acc = np.matrix([[drone_list[0].ax_sens],[drone_list[0].ay_sens],[drone_list[0].az_sens]])
        ancs = []
        for i in range(len(drone_list)-1):
            ancs.append([drone_list[i+1].x, drone_list[i+1].y, drone_list[i+1].z, drone_list[0].r[i]])

        my_ekf.predict(vel_acc)
        my_ekf.observe(ancs)
        my_ekf.update()

        ekfx.append(float(my_ekf.x_k[0]))
        ekfy.append(float(my_ekf.x_k[1]))
        ekfz.append(float(my_ekf.x_k[2]*3))

        ekf_tensor = torch.tensor([[float(my_ekf.x_k[0][0]), float(my_ekf.x_k[1][0]), float(my_ekf.x_k[2][0])]]).to(torch.device('cuda'))
        ekf_target = torch.tensor([[drone_list[0].x_tru, drone_list[0].y_tru, drone_list[0].z_tru]]).to(torch.device("cuda"))
        ekfLoss = loss(ekf_tensor, ekf_target)
        tL.append(ekfLoss.item())
        trux.append(float(drone_list[0].x_tru))
        truy.append(float(drone_list[0].y_tru))
        truz.append(float(drone_list[0].z_tru*3))

    fig = plt.figure()
    ax1 = plt.axes(projection='3d')
    ax1.set_xlabel('East [m]')
    ax1.set_ylabel('North [m]')
    ax1.set_zlabel('Up [m]')
    ax1.set_xlim(-40,40)
    ax1.set_ylim(-10,60)
    ax1.plot3D(trux, truy, truz, color='black', label='True Position')
    ax1.plot3D(ekfx, ekfy, ekfz, color='green', label='Extended Kalman Filter')
    plt.legend()

    x = np.linspace(0, len(tL)*dt, len(tL))
    plt.figure()
    plt.plot(x, tL, label="EKF")
    plt.plot(x, smooth(tL), label="Smoothened EKF")
    plt.xlabel("Time [s]")
    plt.ylabel("Mean Absolute Error [m]")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    gen_plots()