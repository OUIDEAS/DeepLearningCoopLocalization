import torch
import numpy as np
import random as rand
import itertools
import math
from SimFunctions import *
from OLSsolver import *
import matplotlib.pyplot as plt
from PPO import PPO
from tqdm import tqdm
import os
import pandas as pd

class DroneData():
    def __init__(self, x, y, z, r):
        self.x = x
        self.y = y
        self.z = z
        self.r = r

def sensor_noise(r):
    b = 3.1*10**9
    c = 3e8
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

        
def Standardize(m):
    stdev = np.std(m)
    mean = np.mean(m)
    n = [(i-mean)/stdev for i in m]
    return n

class Swarm():
    def __init__(self, num_anchors):
        self.drone_list = []
        self.n_anchors = num_anchors
        for i in range(num_anchors):
            self.drone_list.append(Drone())
        wp=[[-5,0,100],[0,10, 100],[0, 10, 10],[0, 40, 10],[0,40,100], [-5,50,100]]
        self.waypoints=wp
        self.agent = Drone(wp[0], anchor = False)
        self.agent.set_accel(wp[1])
        self.dt = 0.2
        self.w = 1
        self.device = torch.device('cuda')
        self.loss = torch.nn.L1Loss()
        self.avail = None
        self.reset()
        self.single_policy = PPO(state_dim = num_anchors*3+num_anchors, 
                                action_dim = 3, 
                                lr_actor = 1e-4, 
                                lr_critic = 1e-3,
                                gamma = 0.99,
                                K_epochs = 10,
                                eps_clip = 0.2,
                                has_continuous_action_space = True)

    def step(self, ppo):
        criterion = self.agent.reach_wp(self.waypoints[self.w])
        if criterion == 1:
            self.w = self.w + 1
        if self.w == len(self.waypoints):
            self.w = 0
            self.end_sim = True
        else:
            self.end_sim = False

        previous_error, Prev_DOP = self.test_trilat()
        states = []
        for drone in self.drone_list:
            dxyz = []
            for (d, r) in zip(self.drone_list, self.agent.r):
                if d.x - drone.x != 0 or d.y - drone.y !=0 or d.z - drone.z != 0:
                    dxyz.append(r)
                    dxyz.append(d.x)
                    dxyz.append(d.y)
                    dxyz.append(d.z)
                else:
                    state = [r, drone.x, drone.y, drone.z]
            state = list(itertools.chain(state, Standardize(dxyz)))
            states.append(states)
            action = ppo.select_action(state)
                
            drone.set_x_velocity(action[0])
            drone.set_y_velocity(action[1])
            drone.set_z_velocity(action[2])

        for i in range(10):
            self.agent.set_accel(self.waypoints[self.w])
            self.agent.move(self.dt)
            for drone in self.drone_list:
                drone.move(self.dt)

        new_states = []
        for drone in self.drone_list:
            dxyz = []
            for (d, r) in zip(self.drone_list, self.agent.r):
                if d.x - drone.x != 0 or d.y - drone.y !=0 or d.z - drone.z != 0:
                    dxyz.append(r)
                    dxyz.append(d.x)
                    dxyz.append(d.y)
                    dxyz.append(d.z)
                else:
                    state = [r, drone.x, drone.y, drone.z]
            state = list(itertools.chain(state, Standardize(dxyz)))
            new_states.append(states)
           
        new_error, New_DOP = self.test_trilat()
        reward = self.Reward(Prev_DOP, New_DOP)
        return states, reward, self.end_sim, new_states
   

    def Reward(self, pd, nd):
        runner = 0
        for drone in self.drone_list:
            if drone.x > 100 or drone.x < 0 or drone.y > 100 or drone.y < 0 or drone.z > 100 or drone.z < 60:
                runner += -0.1
        if nd < pd:
            runner +=1
        elif nd < 5:
            runner +=1
        return runner


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
        for index in range(len(self.agent.r)):
            avail.append(DroneData(self.drone_list[index].x, self.drone_list[index].y,
                self.drone_list[index].z, self.agent.r[index]))
        self.avail = avail

        self.pgdata = []
        for item in avail:
            xpg.append(item.x)
            ypg.append(item.y)
            zpg.append(item.z)
            
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
        self.agent = Drone(self.waypoints[0], anchor = False)
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


class Runner():
    def __init__(self, number_of_episodes, number_of_anchors = 3, device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
        self.env = Swarm(number_of_anchors)
        self.PPO = PPO(state_dim = number_of_anchors*3+number_of_anchors, 
                       action_dim = 3, 
                       lr_actor = 1e-4, 
                       lr_critic = 1e-3,
                       gamma = 0.99,
                       K_epochs = 10,
                       eps_clip = 0.2,
                       has_continuous_action_space = True)
        self.num_episodes = number_of_episodes
        self.reward_logger = []
        self.device = device
        self.timestep = 0
        self.update_timestep = 10
        self.action_std_decay_freq = 300

    def run_episode(self, episode_num):
        state = self.env.reset()
        self.state = []
        self.action = []
        self.reward = []
        self.new_state = []
        end = False
        running_reward = 0
        step = 0
        while not end:
            state, reward, end, new_state = self.env.step(self.PPO)
            for (st, nst) in zip (state, new_state):
                self.PPO.buffer.rewards.append(reward)
                self.PPO.buffer.is_terminals.append(end)
            running_reward += reward
            self.timestep +=1
            step+=1
            if self.timestep % self.update_timestep == 0:
                self.PPO.update()
                
            if self.timestep % self.action_std_decay_freq == 0:
                self.PPO.decay_action_std(0.01, 0.1)
            state = new_state
        return running_reward/step

    def train(self):
        for i in tqdm(range(self.num_episodes)):
            episode_reward = self.run_episode(i)
            self.reward_logger.append(episode_reward)

if __name__ == "__main__":
    os.system('clear')
    n_episodes = 4000
    n_anchors = 5
    print("Running training for ", n_episodes, " number of episodes.")
    print("Training SAC network to account for ", n_anchors, " anchors.")
    agent = Runner(number_of_episodes = n_episodes, number_of_anchors = n_anchors)
    agent.train()
    plt.figure()
    x = np.linspace(0, agent.num_episodes, len(agent.reward_logger))
    unfiltered = pd.DataFrame(
        {'unfiltered': agent.reward_logger})
    ema = unfiltered.ewm(com=9).mean()
    plt.plot(x, unfiltered, label="Rewards")
    x = np.linspace(0, agent.num_episodes, len(ema))
    plt.plot(x, ema, label="Rewards - smoothened")
    plt.xlabel('Episode [-]')
    plt.ylabel('Reward per State [-]')
    plt.title(str(n_anchors)+' Anchor Drones: Reward vs Episode')
    plt.legend()
    plt.show()