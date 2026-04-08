#!/usr/bin/env python import roslib
###################################################################
#
# Mobile Anchor Point Machine Learning Cooperative Localization
# For Multiagent Multirotor Systems
#  
# Bob Geng - Ohio University
#
# rg383015@ohio.edu
#
####################################################################

from re import X
import rospy
import sys
import scipy
from scipy.optimize import minimize
from geometry_msgs.msg import PoseStamped
import os
import math
import matplotlib.pyplot as plt

os.system('clear')

#Classes to subscribe and log the data
class drone1():
    def __init__(self):
        self.x = None
        self.y = None
        self.z = None
        self.sub = rospy.Subscriber("/uav0/mavros/local_position/pose", PoseStamped, self.callback)
        print("Connection established with Anchor Drone 1")

    def callback(self,data):
        self.x = float(data.pose.position.x)
        self.y = float(data.pose.position.y)
        self.z = float(data.pose.position.z)

class drone2():
    def __init__(self):
        self.x = None
        self.y = None
        self.z = None
        self.sub = rospy.Subscriber("/uav1/mavros/local_position/pose", PoseStamped, self.callback)
        print("Connection established with Anchor Drone 2")

    def callback(self,data):
        self.x = float(data.pose.position.x)
        self.y = float(data.pose.position.y)
        self.z = float(data.pose.position.z)

class drone3():
    def __init__(self):
        self.x = None
        self.y = None
        self.z = None
        self.sub = rospy.Subscriber("/uav2/mavros/local_position/pose", PoseStamped, self.callback)
        print("Connection established with Anchor Drone 3")

    def callback(self,data):
        self.x = float(data.pose.position.x)
        self.y = float(data.pose.position.y)
        self.z = float(data.pose.position.z)

class drone4():
    def __init__(self):
        self.x = None
        self.y = None
        self.z = None
        self.sub = rospy.Subscriber("/uav3/mavros/local_position/pose", PoseStamped, self.callback)
        print("Connection established with Anchor Drone 4")

    def callback(self,data):
        self.x = float(data.pose.position.x)
        self.y = float(data.pose.position.y)
        self.z = float(data.pose.position.z)

class drone5():
    def __init__(self):
        self.x = None
        self.y = None
        self.z = None
        self.sub = rospy.Subscriber("/uav4/mavros/local_position/pose", PoseStamped, self.callback)
        print("Connection established with Anchor Drone 5")

    def callback(self,data):
        self.x = float(data.pose.position.x)
        self.y = float(data.pose.position.y)
        self.z = float(data.pose.position.z)

class drone6():
    def __init__(self):
        self.x = None
        self.y = None
        self.z = None
        self.sub = rospy.Subscriber("/uav5/mavros/local_position/pose", PoseStamped, self.callback)
        print("Connection established with Agent")

    def callback(self,data):
        self.x = float(data.pose.position.x)
        self.y = float(data.pose.position.y)
        self.z = float(data.pose.position.z)

def Trilat(P, anchors):
    lse = 0
    for anchor in anchors:
        x = anchor['X_Position']
        y = anchor['Y_Position']
        z = anchor['Z_Position']
        r = anchor['range']
        error = math.sqrt((x-P[0])**2 + (y-P[1])**2 + (z-P[2])**2)
        lse += (r - error)**2

    return lse


def plot3ax(x,y,z, xnn,ynn,znn):
    fig = plt.figure(1)
    plt.clf()
    ax = plt.axes(projection='3d')
    ax.set_xlim3d(-10, 10)
    ax.set_ylim3d(-10, 10)
    ax.set_zlim3d(-10, 10)
    plt.title('Feed Forward Dynamic Anchor Test')
    ax.plot3D(x,y,z,color='gray')
    ax.scatter(xnn,ynn,znn,color='green')
    plt.pause(0.01)  # pause a bit so that plots are updated

def  main(args, d1, d2, d3, d4, d5, d6):
    if d1.x is not None and d2.x is not None and d3.x is not None and d4.x is not None and d5.x is not None and agent.x is not None:
        a1 = {"X_Position": d1.x,
              "Y_Position": d1.y,
              "Z_Position": d1.z,
              "range":  math.sqrt((d1.x-d6.x)**2 + (d1.y-d6.y)**2 + (d1.z-d6.z)**2)}
        a2 = {"X_Position": d2.x,
              "Y_Position": d2.y,
              "Z_Position": d2.z,
              "range":  math.sqrt((d2.x-d6.x)**2 + (d2.y-d6.y)**2 + (d2.z-d6.z)**2)}
        a3 = {"X_Position": d3.x,
              "Y_Position": d3.y,
              "Z_Position": d3.z,
              "range":  math.sqrt((d3.x-d6.x)**2 + (d3.y-d6.y)**2 + (d3.z-d6.z)**2)}
        a4 = {"X_Position": d4.x,
              "Y_Position": d4.y,
              "Z_Position": d4.z,
              "range":  math.sqrt((d4.x-d6.x)**2 + (d4.y-d6.y)**2 + (d4.z-d6.z)**2)}
        a5 = {"X_Position": d5.x,
              "Y_Position": d5.y,
              "Z_Position": d5.z,
              "range":  math.sqrt((d5.x-d6.x)**2 + (d5.y-d6.y)**2 + (d5.z-d6.z)**2)}
        data = [a1,a2,a3,a4,a5]
        result = minimize(Trilat,[0,0,0], method='Powell', args=(data),tol=0.1)
        return result.x[0], result.x[1], result.x[2]
        
        
        
 
if __name__ == "__main__":
    rospy.init_node('Localization', anonymous = True)
    rate = rospy.Rate(10)
    d1 = drone1()
    d2 = drone2()
    d3 = drone3()
    d4 = drone4()
    d5 = drone5()
    agent = drone6()
    rospy.sleep(5)
    ax = []
    ay = []
    az = []
    xnn = []
    ynn = []
    znn = []
    N = 100
    while not rospy.is_shutdown():
        x, y, z = main(sys.argv, d1, d2, d3, d4, d5, agent)
        ax.append(float(agent.x))
        ay.append(float(agent.y))
        az.append(float(agent.z))
        xnn.append(x)
        ynn.append(y)
        znn.append(z)
        plot3ax(ax[-N:], ay[-N:], az[-N:], xnn[-N:], ynn[-N:], znn[-N:])
        rate.sleep()