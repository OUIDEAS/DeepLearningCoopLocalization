import csv
import torch
import numpy as np
import random as rand
import torch.nn as nn
import torch.nn.functional as F
import csv
import os
import itertools
from haversine import inverse_haversine, Direction, Unit
import math
from dadjokes import Dadjoke
from SimFunctions import *
from OLSsolver import *
from torch.distributions import Categorical
import sys
import matplotlib.pyplot as plt
import random

class ResBlock(nn.Module):
    def __init__(self, res_layers, size, drop, residuals):
        super().__init__()
        self.layers = res_layers*residuals
        rb = []
        for i in range(res_layers):
            rb.append(nn.Linear(size, size))
            rb.append(nn.Dropout(0.1))
            rb.append(nn.PReLU(num_parameters=size))
        self.res = nn.Sequential(*rb)
        self.apply(self.rg_init)

    def forward(self, x):
        return x + self.res(x)

    def rg_init(self, m):
        if isinstance(m, torch.nn.Linear):
            (fan_in, fan_out) = nn.init._calculate_fan_in_and_fan_out(m.weight)
            c = 1
            var = c/(self.layers*fan_in)
            nn.init.normal_(m.weight, 0, math.sqrt(var))
            m.bias.data.zero_()

class ResNet(nn.Module):
    def __init__(self, size:int, res_layers:int, residuals:int, drop=0.1):
        super().__init__()
        self.cer = False
        self.in_layer = nn.Linear(185, size)
        self.activation = nn.PReLU(num_parameters=size)
        self.ResidualNetwork = nn.ModuleList([ResBlock(res_layers, size, drop, residuals) for i in range(residuals)])
        self.out = nn.Linear(size, 3)
    def forward(self, x):
        x = self.in_layer(x)
        x = self.activation(x)
        for layer in self.ResidualNetwork:
            x = layer(x)
        x = self.out(x)
        return x


class env():
    def __init__(self, num_anchors):#drone_list, waypoints):
        self.drone_list = []
        self.n_anchors = num_anchors
        for i in range(num_anchors):
            self.drone_list.append(Drone())
        wp = []
        anc_alt = 100
        # Agents waypoints first
        wp.append([[-5,0,100],[0,10, 100],[0, 10, 10],[0, 40, 10],[0,40,100], [-5,50,100]])
        # Anchors 1, 2, 3,..., 10
        wp.append([[0,0,100],[0,10, 100],[0,10, anc_alt],[0,10, anc_alt],[0, 10, 100],[0,50, 100]])
        wp.append([[10,0,100],[30,10, anc_alt],[50,10, anc_alt],[50,10, anc_alt],[30, 20, 100],[5, 50, 100]])
        wp.append([[20,0,100],[60,10, anc_alt],[100,10, anc_alt],[100,10, anc_alt],[60, 20, 100],[10, 50, 100]])
        wp.append([[0,10,100],[0,50, anc_alt],[0,50, anc_alt],[0,50, anc_alt],[0, 50, 100],[0,60, 100]])
        wp.append([[10,10,100],[30,50, anc_alt],[50,50, anc_alt],[50,50, anc_alt],[30, 50, 100],[5,60, 100]])
        wp.append([[20,10,100],[60,50, anc_alt],[100,50, anc_alt],[100,50, anc_alt],[60, 50, 100],[10,60, 100]])
        wp.append([[0,4,100],[0,25, anc_alt],[0,25, anc_alt],[0,25, anc_alt],[0, 25, 100],[0,54, 100]])
        wp.append([[0,7,100],[0,35, anc_alt],[0,35, anc_alt],[0,35, anc_alt],[0, 35, 100],[0,57, 100]])
        wp.append([[20,4,100],[60,25, anc_alt],[60,25, anc_alt],[60,25, anc_alt],[60, 25, 100],[10,54, 100]])
        wp.append([[20,7,100],[60,35, anc_alt],[60,35, anc_alt],[60,35, anc_alt],[60, 35, 100],[10,57, 100]])
        waypoints = wp
        self.waypoints=wp
        self.agent = Drone(waypoints[0][0], anchor = False)
        self.agent.set_accel(waypoints[0][1])
        self.waypoints = waypoints
        self.dt = 0.2
        self.w = 1
        self.device = torch.device('cuda')
        self.loss = torch.nn.L1Loss()
        self.avail = None
        self.reset()

    def step_AC_continuous(self, action):
        criterion = self.agent.reach_wp(self.waypoints[0][self.w])
        if criterion == 1:
            self.w = self.w + 1
        if self.w == len(self.waypoints[0]):
            self.w = 0
            self.end_sim = True
        else:
            self.end_sim = False

        previous_error, Prev_DOP = self.test_trilat()
        state = self.pgnr
        for i in range(len(self.drone_list)):
            self.drone_list[i].set_x_velocity(action[i*3])
            self.drone_list[i].set_y_velocity(action[i*3+1])
            self.drone_list[i].set_z_velocity(action[i*3+2])
        for i in range(10):
            self.agent.set_accel(self.waypoints[0][self.w])
            self.agent.move(self.dt)
            for drone in self.drone_list:
                drone.move(self.dt)
        new_error, New_DOP = self.test_trilat()
        new_state = self.pgnr
        # reward = self.reward_fcn(previous_error, new_error)
        reward = self.DOP_Reward(Prev_DOP, New_DOP)
        return state, reward, self.end_sim, new_state

    def step(self, commands):

        criterion = self.agent.reach_wp(self.waypoints[0][self.w])
        if criterion == 1:
            self.w = self.w + 1
        if self.w == len(self.waypoints[0]):
            self.w = 0
            self.end_sim = True
        else:
            self.end_sim = False

        self.agent.set_accel(self.waypoints[0][self.w])
        previous_error, Prev_DOP = self.test_trilat()
        state = self.pgnr
        for (drone, command) in zip(self.drone_list, commands):
            drone.set_direction(command)
            drone.move(self.dt)
        self.agent.move(self.dt)
        new_error, New_DOP = self.test_trilat()
        new_state = self.pgnr#data
        # reward = self.reward_fcn(previous_error, new_error)
        reward = self.DOP_Reward(Prev_DOP, New_DOP)
        return state, reward, self.end_sim, new_state

    def DOP_Reward(self, pd, nd):
        runner = 0
        for drone in self.drone_list:
            if drone.x > 100 or drone.x < 0 or drone.y > 100 or drone.y < 0 or drone.z > 100 or drone.z < 60:
                runner += -0.1
        if nd < pd:
            runner +=1
        elif nd < 5:
            runner +=1
        return runner


    def reward_fcn(self, er_prev, error):
        # my_reward = []
        runner = 0
        for drone in self.drone_list:
            if drone.x > 100 or drone.x < 0 or drone.y > 100 or drone.y < 0 or drone.z > 100 or drone.z < 60:
                runner += -5
            else:
                if error < 2:
                    runner+=1
                elif error < er_prev:
                    runner+=1
                elif error == er_prev:
                    runner+=0
                else:
                    runner+=-1
        return runner #sum(my_reward)/len(my_reward)

    def test_trilat(self):
        data, avail = [], []
        r, x, y, z, xpg, ypg, zpg = [], [], [], [], [], [], []
        i = 0
        r.append([])
        for drone in self.drone_list:
            if i < 1:
                i = i+1
                pass
            else:
                r.append([])
                x.append([])
                y.append([])
                z.append([])
        self.agent.range(self.drone_list)
        # self.agent.range(self.drone_list[0], self.drone_list[1], self.drone_list[2])
        for index in range(len(self.agent.r)):
            avail.append(DroneData(self.drone_list[index].x, self.drone_list[index].y,
                self.drone_list[index].z, self.agent.r[index]))
        self.avail = avail

        self.pgdata = []
        for item in avail:
            xpg.append(item.x)
            ypg.append(item.y)
            zpg.append(item.z)
            # Determine targets, and reference origin
        target = torch.tensor([[(self.agent.x_tru - self.drone_list[0].x_tru),
                (self.agent.y_tru - self.drone_list[0].y_tru),
                (self.agent.z_tru - self.drone_list[0].z_tru)]]).to(torch.device("cuda"))

        self.pgdata = list(itertools.chain(self.pgdata, xpg, ypg, zpg))
        u = np.array([[self.agent.x_tru],[self.agent.y_tru],[self.agent.z_tru]])
        pos = Powell_Trilat(avail)
        PDOP = PDOP_Solver(u, avail)
        self.agentx = float(pos[0][0])
        self.agenty = float(pos[1][0])
        self.agentz = float(pos[2][0])
        self.pgnr = list(itertools.chain(xpg, ypg, zpg, [self.agentx, self.agenty, self.agentz, self.agent.vx, self.agent.vy, self.agent.vz]))
        self.agentxtru = self.agent.x_tru
        self.agentytru = self.agent.y_tru
        self.agentztru = self.agent.z_tru
        pos = torch.tensor([[pos[0][0], pos[1][0], pos[2][0]]]).to(self.device)
        return self.loss(pos, target).item(), PDOP

    def reset(self):
        self.w = 1
        self.drone_list = []
        for i in range(self.n_anchors):
            self.drone_list.append(Drone())
        self.agent = Drone(self.waypoints[0][0], anchor = False)
        data, avail = [], []
        r, x, y, z, xpg, ypg, zpg = [], [], [], [], [], [], []
        i = 1
        r.append([])
        for drone in self.drone_list:
            if i < 2:
                i = i+1
                pass
            else:
                r.append([])
                x.append([])
                y.append([])
                z.append([])
        for drone in self.drone_list:
            drone.gps_noise()

        self.agent.range(self.drone_list)
        for index in range(len(self.agent.r)):
            avail.append(DroneData(self.drone_list[index].x, self.drone_list[index].y,
                self.drone_list[index].z, self.agent.r[index]))
        self.avail = avail
        self.pgdata = []
        for i in range(len(avail)):
            self.pgdata.append(avail[i].r)
            xpg.append(avail[i].x)
            ypg.append(avail[i].y)
            zpg.append(avail[i].z)
        target = torch.tensor([[(self.agent.x_tru - self.drone_list[0].x_tru),
                (self.agent.y_tru - self.drone_list[0].y_tru),
                (self.agent.z_tru - self.drone_list[0].z_tru)]]).to(torch.device("cuda"))
        for i in range(len(x)):
            data = list(itertools.chain(data, x[i], y[i], z[i]))
        self.pgdata = list(itertools.chain(self.pgdata, xpg, ypg, zpg))
        data = torch.tensor([data]).to(torch.device('cuda'))
        self.data = data
        pos = Powell_Trilat(avail)
        self.agentx = float(pos[0][0])
        self.agenty = float(pos[1][0])
        self.agentz = float(pos[2][0])
        self.pgnr = list(itertools.chain(xpg, ypg, zpg, [self.agentx, self.agenty, self.agentz, self.agent.vx, self.agent.vy, self.agent.vz]))
        self.agentxtru = self.agent.x_tru
        self.agentytru = self.agent.y_tru
        self.agentztru = self.agent.z_tru
        _ = self.test_trilat()
        return self.pgnr
