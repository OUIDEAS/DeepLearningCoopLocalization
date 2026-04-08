import csv
import torch
import numpy as np
import random as rand
import torch.nn as nn
import torch.nn.functional as F
import csv
import os
from PosToGPS import *
import itertools
from haversine import inverse_haversine, Direction, Unit
import math
from dadjokes import Dadjoke
from SimFunctions import *
from NN import *
from NNLib import *
from OLSsolver import *
from torch.distributions import Categorical
import sys
import matplotlib.pyplot as plt
import random


class DQN(nn.Module):
    def __init__(self):
        super().__init__()
        size = 1000
        layers = 3
        rb = [torch.nn.Linear(40, size), nn.PReLU(num_parameters=size)]
        for i in range(layers):
            rb.append(nn.Linear(size, size))
            rb.append(nn.PReLU(num_parameters=size))
        rb.append(nn.Linear(size, 1))
        self.StateVal = nn.Sequential(*rb)

    def forward(self, x):
        return self.StateVal(x)

class OutputLayers(nn.Module):
    def __init__(self, layers, size):
        super().__init__()
        out = []
        for layer in range(layers):
            out.append(nn.Linear(size, size))
            out.append(nn.PReLU(num_parameters=size))
        out.append(nn.Linear(size, 7))
        self.outputlayers = nn.Sequential(*out)
    def forward(self, x):
        x = self.outputlayers(x)
        x = torch.nn.functional.softmax(x, dim=1)
        return x

class AnchorHeadingPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        size = 1000
        layers = 3
        out = 7
        rb = [torch.nn.Linear(40, size), nn.ReLU()]
        for i in range(layers):
            rb.append(torch.nn.Linear(size, size))
            rb.append(nn.ReLU())

        self.Policy = nn.Sequential(*rb)
        self.out = nn.ModuleList([OutputLayers(layers, size) for i in range(10)])


    def forward(self,x):
        x = self.Policy(x)
        outputs = []
        for layer in self.out:
            outputs.append(layer(x))

        return outputs

def reward(er_prev, error, anchor):
    if anchor.x > 100 or anchor.x < 0 or anchor.y > 100 or anchor.y < 0 or anchor.z > 100 or anchor.z < 75:
        my_reward = -10 + math.exp(10/error)
    else:
        my_reward = math.exp(10/error) - 1 #(er_prev - error)/(0.001*er_prev*error) if (er_prev - error) > 0 else -10
    return my_reward

def main(waypoints, load):
    # Publish results for a another program to plot live
    num_episode = 1000
    batch_size = 1
    learning_rate = 0.001
    gamma = 0#.9
    epsilon = 0.5
    eps_decay = 0.0005
    eps_tracker = []

    os.system('clear')
    PATH="/home/rgeng98/BobGeng-Thesis/Anchor Control/NN_Architectures/L2-Reg-ResNet.pt"
    model = torch.load(PATH)
    print(model)
    window_size = 6
    w = 1
    num_anchors = 10
    loss = torch.nn.L1Loss()
    drone_list = []
    if load:
        AnchorPolicy = torch.load("Policy.pt").to(torch.device("cuda"))
        critic = torch.load("critic.pt").to(torch.device('cuda'))
    else:
        AnchorPolicy = AnchorHeadingPolicy().to(torch.device("cuda"))
        critic = DQN().to(torch.device('cuda'))
    print("Actor:")
    print(AnchorPolicy)
    print("Critic:")
    print(critic)
    optim = torch.optim.Adam(AnchorPolicy.parameters(), lr = learning_rate)
    optimc = torch.optim.Adam(critic.parameters(), lr = learning_rate)
    critic_loss = nn.L1Loss()
    print('Dad Joke:')
    dadjoke = Dadjoke()
    print(dadjoke.joke, '\n')
    nnx, nny, nnz = [], [], []
    nnL = []
    trux, truy, truz = [], [], []
    dt = 0.25
    comm_int = []

    # Batch History
    state_pool = []
    action_pool = []
    reward_pool = []
    steps = 0
    rewards = []
    running_reward = []
    pgloss_list = []
    pglossnm_list = []
    for e in range(num_episode):
        sys.stdout.write("\rEpisode: {0}/{1}".format(e+1, num_episode))
        sys.stdout.flush()
        agent = Drone(waypoints[0][0], anchor = False)
        drone_list = []
        for i in range(len(waypoints)-1):
            drone_list.append(Drone())
        for i in range(len(waypoints)-1):
            drone_list[i].gps_noise()
        agent.set_accel(waypoints[0][1])
        end_sim = False
        first_state = True
        count = 0
        while not end_sim:
            count=count+1
            criterion = agent.reach_wp(waypoints[0][w])

            if criterion == 1:
                w = w + 1

            if w == len(waypoints[0]):
                w = 5
                end_sim = True

            agent.set_accel(waypoints[0][w])
            data, avail = [], []
            r, x, y, z, xpg, ypg, zpg = [], [], [], [], [], [], []
            i = 1
            r.append([])
            for drone in drone_list:
                if i < 2:
                    i = i+1
                    pass
                else:
                    r.append([])
                    x.append([])
                    y.append([])
                    z.append([])


            agent.move(dt)
            for drone in drone_list:
                drone.gps_noise()

            for b in range(6):
                for drone in drone_list:
                    drone.gps_noise()
                avail.append([])
                agent.range(drone_list[0],drone_list[1],drone_list[2],drone_list[3],drone_list[4],drone_list[5],drone_list[6],drone_list[7],drone_list[8],drone_list[9])
                for index in range(len(agent.r)):
                    avail[b].append(DroneData(drone_list[index].x, drone_list[index].y, drone_list[index].z, agent.r[index]))
                # 1st anchor drone is where others positions are referenced to, so start recycling inputs with the second one

            # For each drone available in the list, save all timesteps of range readings
            for i in range(len(avail[1])):
                for item in avail:
                    data.append(item[i].r)#/math.sqrt(27))

            # For each drone in the list, save all time steps of ENU estimates
            for i in range(len(avail[1])-1):
                for item in avail:
                    x[i].append((item[i+1].x - item[0].x))
                    y[i].append((item[i+1].y - item[0].y))
                    z[i].append((item[i+1].z - item[0].z))
            pgdata = []
            for i in range(len(avail[1])):
                pgdata.append(avail[0][i].r)
                xpg.append(avail[0][i].x)
                ypg.append(avail[0][i].y)
                zpg.append(avail[0][i].z)

            # Determine targets, and reference origin
            target = torch.tensor([[(drone_list[0].x_tru - drone_list[1].x_tru),
                    (drone_list[0].y_tru - drone_list[1].y_tru),
                    (drone_list[0].z_tru - drone_list[1].z_tru)]]).to(torch.device("cuda"))
            for i in range(len(x)):
                data = list(itertools.chain(data, x[i], y[i], z[i]))

            pgdata = list(itertools.chain(pgdata, xpg, ypg, zpg))

            data = torch.tensor([data]).to(torch.device("cuda"))
            with torch.no_grad():
                pos = model(data)
            error = loss(pos, target)
            current_error=error.item()
            pgdata = torch.tensor([pgdata]).to(torch.device("cuda"))
            state_pool.append(pgdata)

            ouputs = AnchorPolicy(pgdata)
            stateval = critic(pgdata)
            # For actions in outputs loss = m.logprob(action)*stateval
            # train critic network based on the measured rewards
            commands = []
            pglosses=[]
            optim.zero_grad()
            for (i, drone) in zip(ouputs, drone_list):
                # Mask options that tell drone to exit workspace
                mask = drone.set_mask()

                for j in mask:
                    i[0][j] = 0

                # Create Probability Distribution
                m = Categorical(i)

                # Randomly select whether to use policy or to select random action, weighted with epsilon term
                if torch.distributions.Bernoulli(torch.tensor(epsilon)).sample().item() == 0:
                    action = m.sample().cpu()
                    action = action.data.numpy().astype(int)[0]

                # Create an evenly distributed action space that maintains the previous mask
                else:
                    ones = torch.ones(i.size()).to(torch.device('cuda'))
                    s = torch.where(i == 0, i, ones)
                    p = Categorical(s)
                    action = p.sample().cpu()
                    action = action.data.numpy().astype(int)[0]

                pglosses.append(m.log_prob(torch.tensor([action]).to(torch.device('cuda'))) * stateval)

                # Send command to the Anchor Drones
                drone.set_direction(action)

                # Add action for the specific drone to the list
                commands.append(action)
            pgloss = torch.stack(pglosses).sum()
            optim.step()
            # Save Actions
            action_pool.append(commands)

            # Decay Epsilon
            epsilon = epsilon-eps_decay*epsilon
            eps_tracker.append(epsilon)

            # Once all drones are given their command, move them in their directions
            for drone in drone_list:
                drone.move(dt)

            if not first_state:
                reward_list = np.array([])
                for drone in drone_list:
                    reward_list = np.append(reward_list, reward(previous_error, current_error, drone))
                reward_pool.append(reward_list)#np.mean(reward_list))

                calculated_reward = np.mean(reward_list)
                if count%5 == 0:
                    rewards.append(float(calculated_reward))
                    running_reward.append(sum(rewards[-20:])/len(rewards[-20:]))
            first_state=False
            previous_error = current_error

        # --------------------------------------------------------------------------
        #
        # Train the anchor Policy
        #
        if e > 0 and e % batch_size == 0:
            # Discount reward
            running_add = 0
            # Should not have to do this. Something is wrong with the way the lists are set up
            for i in reversed(range(len(state_pool)-2)):
                # if reward_pool[i].any() == 0:
                #     running_add = 0
                #else:
                running_add = running_add * gamma + reward_pool[i]
                reward_pool[i] = running_add

            reward_mean = np.mean(reward_pool)
            reward_std = np.std(reward_pool)
            # for i in range(steps):
            #     reward_pool[i] = (reward_pool[i] - reward_mean) / reward_std

            # Gradient Desent
            optim.zero_grad()
            for i in range(len(state_pool)-2):
                state = state_pool[i]
                action = action_pool[i]
                state_reward = reward_pool[i]
                outputs = AnchorPolicy(state)
                # print(outputs)
                optimc.zero_grad()
                stateval = critic(state)
                # print(stateval)
                critic_error = critic_loss(stateval, torch.tensor([[np.mean(state_reward)]]).to(torch.device('cuda')))
                critic_error.backward()
                optimc.step()
                pglosses = []
                for (probs, a, r) in zip(outputs, action, state_reward):
                    a = Variable(torch.FloatTensor([a])).to(torch.device('cuda'))
                    m = Categorical(probs)
                    r = Variable(torch.tensor(r)).to(torch.device('cuda'))
                    a = torch.tensor([a]).to(torch.device('cuda'))
                    pglosses.append(-m.log_prob(a) * r)
                    # pgloss = -m.log_prob(a)*r
                    # pgloss.backward(retain_graph=True)
                pgloss = torch.stack(pglosses).sum()
                optim.step()
            state_pool = []
            action_pool = []
            reward_pool = []
            steps = 0

    torch.save(AnchorPolicy, "Policy.pt")

    # --------------------------------------------------------------------------
    #
    # Plot the Rewards throughout training
    #
    plt.figure()
    plt.plot(eps_tracker)
    plt.title('Probability of Exploring Actions Space')
    plt.ylabel('Probability [-]')
    plt.xlabel('Iteration [-]')
    plt.show()

    plt.figure()
    plt.plot(rewards, label='Reward')
    plt.plot(running_reward, label='Average Reward')
    plt.title('Anchor Policy Reward')
    plt.ylabel('Reward')
    plt.xlabel('Iteration')
    plt.legend()
    plt.show()

    # --------------------------------------------------------------------------
    #
    # Run a Flight Simulation to determine the accuracy with AI determined
    # Anchor orientations
    #
    nnx, nny, nnz = [], [], []
    trux, truy, truz =[], [], []
    agent = Drone(waypoints[0][0], anchor = False)
    for i in range(len(waypoints)-1):
        drone_list.append(Drone(waypoints[i+1][2]))
    for i in range(len(waypoints)-1):
        drone_list[i].gps_noise()
    agent.set_accel(waypoints[0][1])
    end_sim = False
    first_state = True
    while not end_sim:
        criterion = agent.reach_wp(waypoints[0][w])

        if criterion == 1:
            w = w + 1

        if w == len(waypoints[0]):
            w = 5
            end_sim = True

        agent.set_accel(waypoints[0][w])
        data = []
        avail = []
        r, x, y, z, xpg, ypg, zpg = [], [], [], [], [], [], []
        i = 1
        r.append([])
        for drone in drone_list:
            if i < 2:
                i = i+1
                pass
            else:
                r.append([])
                x.append([])
                y.append([])
                z.append([])

        agent.move(dt)
        for drone in drone_list:
            drone.gps_noise()

        for b in range(6):
            for drone in drone_list:
                drone.gps_noise()
            avail.append([])
            agent.range(drone_list[0],drone_list[1],drone_list[2],drone_list[3],drone_list[4],drone_list[5],drone_list[6],drone_list[7],drone_list[8],drone_list[9])
            for index in range(len(agent.r)):
                avail[b].append(DroneData(drone_list[index].x, drone_list[index].y, drone_list[index].z, agent.r[index]))
            # 1st anchor drone is where others positions are referenced to, so start recycling inputs with the second one
            i = 0
            comm_int.append(num_anchors-len(avail[b]))
            while len(avail[b]) < num_anchors:
                avail[b].append(avail[b][i])
                i = i+1

        # For each drone available in the list, save all timesteps of range readings
        for i in range(len(avail[1])):
            for item in avail:
                data.append(item[i].r)#/math.sqrt(27))
        pgdata = []
        for i in range(len(avail[1])):
            pgdata.append(avail[0][i].r)
            xpg.append(avail[0][i].x)
            ypg.append(avail[0][i].y)
            zpg.append(avail[0][i].z)

        # For each drone in the list, save all time steps of ENU estimates
        for i in range(len(avail[1])-1):
            for item in avail:
                x[i].append((item[i+1].x - item[0].x))
                y[i].append((item[i+1].y - item[0].y))
                z[i].append((item[i+1].z - item[0].z))


        # Determine targets, and reference origin
        target = torch.tensor([[(drone_list[0].x_tru - drone_list[1].x_tru), (drone_list[0].y_tru - drone_list[1].y_tru), (drone_list[0].z_tru - drone_list[1].z_tru)]]).to(torch.device("cuda"))
        for i in range(len(x)):
            data = list(itertools.chain(data, x[i], y[i], z[i]))

        pgdata = list(itertools.chain(pgdata, xpg, ypg, zpg))

        data = torch.tensor([data]).to(torch.device("cuda"))
        pgdata = torch.tensor([pgdata]).to(torch.device("cuda"))
        with torch.no_grad():
            pos = model(data)
            nnx.append(drone_list[0].x + float(pos[0][0]))
            nny.append(drone_list[0].y + float(pos[0][1]))
            nnz.append(drone_list[0].z + float(pos[0][2]))
        trux.append(agent.x_tru)
        truy.append(agent.y_tru)
        truz.append(agent.z_tru)
        error = loss(pos, target)
        ouputs = AnchorPolicy(pgdata)
        commands = []
        for (i, drone) in zip(ouputs, drone_list):
            # Mask options that tell drone to exit workspace
            mask = drone.set_mask()
            for j in mask:
                i[0][j] = 0
            m = Categorical(i)
            action = m.sample().cpu()
            action = action.data.numpy().astype(int)[0]

            # Send command to the Anchor Drones
            drone.set_direction(action)

        for drone in drone_list:
            drone.move(dt)

    fig = plt.figure()
    ax = plt.axes(projection='3d')

    ax.set_xlabel('East [-]')
    ax.set_ylabel('North [-]')
    ax.set_zlabel('Up [-]')
    ax.set_xlim(-15,15)
    ax.set_ylim(-10,60)
    plt.title('ResNet Dynamic Anchor Test')
    ax.plot3D(trux, truy, truz, color='gray', label='True Position')
    ax.plot3D(nnx, nny, nnz, color='green', label='ANN')
    plt.legend()
    plt.show()

if __name__ == "__main__":
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

    main(wp, load = False)
