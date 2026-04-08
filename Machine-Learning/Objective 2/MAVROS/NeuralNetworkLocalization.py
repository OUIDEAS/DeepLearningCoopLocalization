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
import torch
import torch.nn as nn
from geometry_msgs.msg import PoseStamped
import os
import math
import matplotlib.pyplot as plt

class NN(nn.Module):
    def __init__(self, hidden_size, rnn_layers):
        super(NN,self).__init__()
        self.hidden_size = hidden_size
        self.rnn_layers = rnn_layers
        self.fc1 = nn.Linear(17, hidden_size)
        self.rnn = nn.RNN(hidden_size, hidden_size, self.rnn_layers, batch_first=False, nonlinearity='tanh')
        #self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc = nn.Linear(hidden_size,3)

    def forward(self,x):
        x = torch.tanh(self.fc1(x))
        x = x.reshape(1,1,self.hidden_size)
        h0 = torch.zeros(self.rnn_layers, x.size(0), self.hidden_size)
        out, hn = self.rnn(x, h0.detach())
        out = torch.tanh(self.fc(out[:, -1, :]))
        x = out.reshape(1,3)
        #x = torch.tanh(self.fc(x))
        return x



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

def prepare(a1,a2,a3,a4,a5,agent):
    r1 = math.sqrt((a1.x-agent.x)**2 + (a1.y-agent.y)**2 + (a1.z-agent.z)**2)
    r2 = math.sqrt((a2.x-agent.x)**2 + (a2.y-agent.y)**2 + (a2.z-agent.z)**2)
    r3 = math.sqrt((a3.x-agent.x)**2 + (a3.y-agent.y)**2 + (a3.z-agent.z)**2)
    r4 = math.sqrt((a4.x-agent.x)**2 + (a4.y-agent.y)**2 + (a4.z-agent.z)**2)
    r5 = math.sqrt((a5.x-agent.x)**2 + (a5.y-agent.y)**2 + (a5.z-agent.z)**2)
    d12x = a2.x - a1.x
    d12y = a2.y - a1.y
    d12z = a2.z - a1.z
    d13x = a3.x - a1.x
    d13y = a3.y - a1.y
    d13z = a3.z - a1.z
    d14x = a4.x - a1.x
    d14y = a4.y - a1.y
    d14z = a4.z - a1.z
    d15x = a5.x - a1.x
    d15y = a5.y - a1.y
    d15z = a5.z - a1.z
    data = torch.tensor([r1,r2,r3,r4,r5,d12x,d12y,d12z,d13x,d13y,d13z,d14x,d14y,d14z,d15x,d15y,d15z])
    return data/torch.max(abs(data))

def eval(score, a1, agent):
    r = math.sqrt((a1.x-agent.x)**2 + (a1.y-agent.y)**2 + (a1.z-agent.z)**2)
    guessx = a1.x + score[0][0] * r
    guessy = a1.y + score[0][1] * r
    guessz = a1.z + score[0][2] * r
    #print(float(guessx - agent.x))
    #print(float(guessy - agent.y))
    #print(float(guessz - agent.z))
    return float(guessx), float(guessy), float(guessz)

def plot3ax(x,y,z, xnn,ynn,znn):
    fig = plt.figure(1)
    plt.clf()
    ax = plt.axes(projection='3d')
    ax.set_xlim3d(-6, 6)
    ax.set_ylim3d(-6, 6)
    ax.set_zlim3d(-6, 6)
    plt.title('Feed Forward Dynamic Anchor Test')
    ax.plot3D(x,y,z,color='gray')
    ax.scatter(xnn,ynn,znn,color='green')
    plt.pause(0.01)  # pause a bit so that plots are updated

def  main(args, model, d1, d2, d3, d4, d5, agent):
    if d1.x is not None and d2.x is not None and d3.x is not None and d4.x is not None and d5.x is not None and agent.x is not None:
        data = prepare(d1,d2,d3,d4,d5,agent).to(device="cpu")
        with torch.no_grad():
            output = model(data)
        x, y, z = eval(output, d1, agent)
        return x, y, z
        
 
if __name__ == "__main__":
    rospy.init_node('Localization', anonymous = True)
    PATH = 'rnndl.pt'
    model = torch.load(PATH)
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
        x, y, z = main(sys.argv, model, d1, d2, d3, d4, d5, agent)
        ax.append(float(agent.x))
        ay.append(float(agent.y))
        az.append(float(agent.z))
        xnn.append(x)
        ynn.append(y)
        znn.append(z)
        plot3ax(ax[-N:], ay[-N:], az[-N:], xnn[-N:], ynn[-N:], znn[-N:])
        rate.sleep()