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
            snr = 10 - r*(10-0.1)/175
            sigsq = 1/(8*(math.pi**2)*(b**2)*snr)
            sig = 40*c*math.sqrt(sigsq)
            r_err = float(sig*sig)
        except:
            snr = 0.00000000000000000000000001
            sigsq = 1/(8*(math.pi**2)*(b**2)*snr)
            sig = 40*c*math.sqrt(sigsq)
            r_err = float(sig*sig)
        return r_err


def test_kalman():
    r = 50
    rs = sensor_noise(r)
    a = KalmanFilter(rs)
    rk = []
    rtru = []
    rsensor = []
    for i in range(6):
        if i >= 49:
            r += 0.05
        a.predict()
        rs = sensor_noise(r)
        a.update(rs)
        rsensor.append(rs)
        rk.append(a.x_k)
        rtru.append(r)
    
    plt.figure()
    plt.plot(rsensor)
    plt.plot(rk)
    plt.plot(rtru)
    plt.show()


if __name__ == "__main__":
    test_kalman()