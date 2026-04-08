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
from torch.distributions import Categorical, Bernoulli
import sys
import matplotlib.pyplot as plt
import random
import environment
from collections import namedtuple, deque
from MapleLeaf import Goal
from torch.autograd import Variable
import time
import pickle
import numpy as np


class OutputLayers(nn.Module):

    def __init__(self, layers, size):

        super().__init__()
        out = []
        for layer in range(layers):
            out.append(nn.Linear(size, size))
            out.append(nn.ReLU())
            out.append(nn.LayerNorm(size))

        out.append(nn.Linear(size, 7))
        self.outputlayers = nn.Sequential(*out)

    def forward(self, x):

        x = self.outputlayers(x)
        x = torch.nn.functional.softmax(x, dim=0)
        return Categorical(x)

class AnchorHeadingPolicy(nn.Module):

    def __init__(self, layernorm = True):

        super().__init__()
        size = 1500
        layers = 4
        out = 7
        self.in_layers = nn.Sequential(*[torch.nn.Linear(36, size), nn.ReLU(), nn.LayerNorm(size)])
        rb = [torch.nn.Linear(size, size), nn.ReLU()]
        if layernorm:
            rb.append(nn.LayerNorm(size))
        for i in range(layers):
            rb.append(torch.nn.Linear(size, size))
            rb.append(nn.ReLU())
            if layernorm:
                rb.append(nn.LayerNorm(size))

        self.Policy = nn.Sequential(*rb)
        self.out = nn.ModuleList([OutputLayers(layers, size) for i in range(10)])

    def forward(self,x):
        x = self.in_layers(x)
        x = x + self.Policy(x)
        outputs = []
        for layer in self.out:
            outputs.append(layer(x))

        return outputs

def main(wp, load):
    os.system('clear')
    print("Start: ", time.ctime())
    APATH = "Actor.pt"
    CPATH = "Critic.pt"
    num_episode = 700#5000
    batch_size = 1
    actor_learning_rate = 0.0001
    gamma = 0.99
    cl = []
    replay_train = 10
    if load:
        open_file = open("Reward_History.pkl", "rb")
        rlav = pickle.load(open_file)
        open_file.close()
        Actor = torch.load(APATH)
    else:
        rlav = []
        Actor = AnchorHeadingPolicy().to(torch.device("cuda"))
    Actor_optim = torch.optim.RMSprop(Actor.parameters(), lr = actor_learning_rate, eps=1e-5)
    reward_list = []
    running_reward_loss = []
    env = environment.env()
    for i in range(num_episode):
        sys.stdout.write("\rEpisode: {0}/{1}".format(i+1, num_episode))
        sys.stdout.flush()
        end = False
        _ = env.reset()
        commands = []
        input_tensor = torch.tensor(env.pgnr).to(torch.device('cuda'))
        with torch.no_grad():
            output = Actor(input_tensor)
            for m in output:
                action = m.sample().cpu()
                action = action.data.numpy().astype(int)
                commands.append(action)

        fs = True
        rewards = []
        states = []
        actions = []
        while not end:
            state, reward, end, new_state = env.step(commands)
            rewards.append(reward)
            states.append(state)

            commands = []
            input_tensor = torch.tensor(env.pgnr).to(torch.device('cuda'))
            with torch.no_grad():
                output = Actor(input_tensor)
                for (m, drone) in zip(output, env.drone_list):
                    mask = drone.set_mask()
                    action = m.sample().cpu()
                    action = action.item()#data.numpy().astype(int)
                    commands.append(action)
            actions.append(commands)
        Actor_optim.zero_grad()
        rewards = np.array(rewards)
        running_add1 = 0
        running_add2 = 0
        running_add3 = 0
        running_add4 = 0
        running_add5 = 0
        running_add6 = 0
        running_add7 = 0
        running_add8 = 0
        running_add9 = 0
        running_add0 = 0

        train_rewards = [[],[],[],[],[],[],[],[],[],[]]
        for a in reversed(range(len(states))):
            running_add1 = running_add1*gamma + rewards[a][0]
            rewards[a][0] = running_add1
            running_add2 = running_add2*gamma + rewards[a][1]
            rewards[a][1] = running_add2
            running_add3 = running_add3*gamma + rewards[a][2]
            rewards[a][2] = running_add3
            running_add4 = running_add4*gamma + rewards[a][3]
            rewards[a][3] = running_add4
            running_add5 = running_add5*gamma + rewards[a][4]
            rewards[a][4] = running_add5
            running_add6 = running_add6*gamma + rewards[a][5]
            rewards[a][5] = running_add6
            running_add7 = running_add7*gamma + rewards[a][6]
            rewards[a][6] = running_add7
            running_add8 = running_add8*gamma + rewards[a][7]
            rewards[a][7] = running_add8
            running_add9 = running_add9*gamma + rewards[a][8]
            rewards[a][8] = running_add9
            running_add0 = running_add0*gamma + rewards[a][9]
            rewards[a][9] = running_add0

            reward_list.append(sum(rewards[a])/len(rewards[a]))
            rlav.append(sum(reward_list[-100:])/len(reward_list[-100:]))
        for a in range(len(states)):
            state = torch.tensor(states[a])
            for m in range(10):
                try:
                    rewards[a][m] = (rewards[a][m] - np.mean(rewards[-(len(rewards)-1):][m]))/np.std(rewards[-(len(rewards)-1):][m])
                except:
                    rewards[a][m] = rewards[a][m]

            probs = Actor(Variable(state).to(torch.device('cuda')))
            pgloss = (-probs[0].log_prob(torch.tensor(actions[a][0]).to(torch.device('cuda')))*rewards[a][0])
            pgloss.backward(retain_graph=True)
            pgloss = (-probs[1].log_prob(torch.tensor(actions[a][1]).to(torch.device('cuda')))*rewards[a][1])
            pgloss.backward(retain_graph=True)
            pgloss = (-probs[2].log_prob(torch.tensor(actions[a][2]).to(torch.device('cuda')))*rewards[a][2])
            pgloss.backward(retain_graph=True)
            pgloss = (-probs[3].log_prob(torch.tensor(actions[a][3]).to(torch.device('cuda')))*rewards[a][3])
            pgloss.backward(retain_graph=True)
            pgloss = (-probs[4].log_prob(torch.tensor(actions[a][4]).to(torch.device('cuda')))*rewards[a][4])
            pgloss.backward(retain_graph=True)
            pgloss = (-probs[5].log_prob(torch.tensor(actions[a][5]).to(torch.device('cuda')))*rewards[a][5])
            pgloss.backward(retain_graph=True)
            pgloss = (-probs[6].log_prob(torch.tensor(actions[a][6]).to(torch.device('cuda')))*rewards[a][6])
            pgloss.backward(retain_graph=True)
            pgloss = (-probs[7].log_prob(torch.tensor(actions[a][7]).to(torch.device('cuda')))*rewards[a][7])
            pgloss.backward(retain_graph=True)
            pgloss = (-probs[8].log_prob(torch.tensor(actions[a][8]).to(torch.device('cuda')))*rewards[a][8])
            pgloss.backward(retain_graph=True)
            pgloss = (-probs[9].log_prob(torch.tensor(actions[a][9]).to(torch.device('cuda')))*rewards[a][9])
            pgloss.backward(retain_graph = True)
        for param in Actor.parameters():
            param.grad.data.clamp_(-1, 1)
        Actor_optim.step()

    os.system('clear')
    # Goal.Horn()
    os.system('clear')
    x = np.linspace(0, num_episode, len(rlav))
    plt.figure()
    plt.title('Reward')
    plt.plot(x, rlav, label='Moving Average')
    plt.show()
    open_file = open("Reward_History.pkl", "wb")
    pickle.dump(rlav, open_file)
    open_file.close()

    # Generate Plot
    end = False
    _ = env.reset()
    commands = []
    input_tensor = torch.tensor([env.pgnr]).to(torch.device('cuda'))
    with torch.no_grad():
        output = Actor(input_tensor)
        for m in output:
            action = m.sample().cpu()
            action = action.data.numpy().astype(int)[0]
            commands.append(action)

    fs = True
    x_tru, y_tru, z_tru = [], [], []
    x, y, z = [], [], []
    env.dt = 0.1
    while not end:
        state, reward, end, new_state = env.step(commands)
        x_tru.append(env.agent.x_tru)
        y_tru.append(env.agent.y_tru)
        z_tru.append(env.agent.z_tru)
        x.append(env.agentx)
        y.append(env.agenty)
        z.append(env.agentz)
        commands = []
        input_tensor = torch.tensor([env.pgnr]).to(torch.device('cuda'))
        with torch.no_grad():
            output = Actor(input_tensor)
            for m in output:
                action = m.sample().cpu()
                action = action.data.numpy().astype(int)[0]
                commands.append(action)

    fig = plt.figure()
    ax = plt.axes(projection='3d')

    ax.set_xlabel('East [-]')
    ax.set_ylabel('North [-]')
    ax.set_zlabel('Up [-]')
    plt.title('ResNet Dynamic Anchor Test')
    ax.plot3D(x_tru, y_tru, z_tru, color='gray', label='True Position')
    ax.plot3D(x, y, z, color='green', label='ANN')
    plt.legend()
    plt.show()
    torch.save(Actor, APATH)


if __name__ == "__main__":
    wp = []
    anc_alt = 100
    # Agents waypoints first
    wp.append([[-5,0,100],[0,10, 100],[0, 10, 10],[0, 40, 10],[0,40,100], [-5,50,100]])

    # Anchors 1, 2, 3, ..., 10
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

    main(wp, load=False)
