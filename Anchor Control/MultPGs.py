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
import environment
from collections import namedtuple, deque

class DQN(nn.Module):

    def __init__(self):
        super().__init__()
        size = 100
        layers = 5
        rb = [torch.nn.Linear(50, size), nn.ReLU()]
        for i in range(layers):
            rb.append(nn.Linear(size, size))
            rb.append(nn.ReLU())
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
            out.append(nn.ReLU())
        out.append(nn.Linear(size, 7))
        out.append(nn.Softmax(dim=1))
        self.outputlayers = nn.Sequential(*out)

    def forward(self, x):
        x = self.outputlayers(x)
        return x

class AnchorHeadingPolicy(nn.Module):

    def __init__(self):
        super().__init__()
        size = 2000
        layers = 2
        out = 7
        rb = [torch.nn.Linear(40, size), nn.ReLU()]
        for i in range(layers):
            rb.append(torch.nn.Linear(size, size))
            rb.append(nn.ReLU())

        self.Policy = nn.Sequential(*rb)
        self.out = OutputLayers(layers, size)

    def forward(self,x):
        x = self.Policy(x)
        x = self.out(x)

        return x #outputs

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([],maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

    def fetch(self, index):
        return [self.memory[index]]

def main(wp, load):
    os.system('clear')
    num_episode = 100
    batch_size = 1
    actor_learning_rate = 0.0000001
    critic_learning_rate = 0.00000001
    gamma = 0.9
    epsilon = 0.99
    eps_decay = 0.0001
    eps_tracker = []
    reward_loss = []
    memory = ReplayMemory(1000)
    cl = []
    replay_train = 100
    Policy_List = []
    actor_optims = []
    for i in range(10):
        A = AnchorHeadingPolicy().to(torch.device("cuda"))
        AO = torch.optim.Adam(A.parameters(), lr = actor_learning_rate)
        Policy_List.append(A)
        actor_optims.append(AO)
    # Critic = DQN().to(torch.device("cuda"))
    # Critic_optim = torch.optim.Adam(Critic.parameters(), lr = critic_learning_rate)

    #
    # critic_loss = nn.L1Loss()

    reward_list = []
    running_reward_loss = []
    rlav = []
    clav = []
    rav = 0
    for i in range(num_episode):
        sys.stdout.write("\rEpisode: {0}/{1}".format(i+1, num_episode))
        sys.stdout.flush()
        env = environment.env()
        end = False
        _ = env.reset()
        commands = []
        input_tensor = torch.tensor([env.pgdata]).to(torch.device('cuda'))
        c_tens=env.pgdata
        a = []
        with torch.no_grad():
            for (drone, Policy, optima) in zip(drone_list, Policy_List, actor_optims):
                output = Policy(input_tensor)
                m = Categorical(output)
                action = m.sample().cpu()
                action = action.data.numpy().astype(int)[0]
                a.append(action)
                c_tens = list(itertools.chain(c_tens, [action]))

        stateval = Critic(torch.tensor([c_tens]).to(torch.device('cuda')))
        pgloss = None
        for (Policy, optima, action) in zip(Policy_List, actor_optims, a):
            optima.zero_grad()
            output = Policy(input_tensor)
            m = Categorical(output)
            pgloss = -m.log_prob(torch.tensor(action).to(torch.device('cuda'))) * (stateval.item()-rav)
            pgloss.backward()
            optima.step()
            # Send command to the Anchor Drones
            drone.set_direction(action)

            # Add action for the specific drone to the list
            commands.append(action)


        fs = True
        statevals = []
        while not end:
            state, reward, end, new_state = env.step(commands)
            reward_list.append(reward)
            rlav.append(sum(reward_list[-20:])/len(reward_list[-20:]))
            rav = sum(reward_list)/len(reward_list)
            memory.push(state, commands, new_state, reward)


            Critic_optim.zero_grad()
            out = Critic(torch.tensor([c_tens]).to(torch.device('cuda')))
            error = critic_loss(out, torch.tensor([[reward]]).to(torch.device('cuda')))
            error.backward()
            cl.append(error.item())
            Critic_optim.step()
            clav.append(sum(cl[-20:])/len(cl[-20:]))

            commands = []
            c_tens = env.pgdata
            a = []
            input_tensor = torch.tensor([env.pgdata]).to(torch.device('cuda'))
            with torch.no_grad():
                for (drone, Policy, optima) in zip(drone_list, Policy_List, actor_optims):
                    output = Policy(input_tensor)
                    m = Categorical(output)
                    action = m.sample().cpu()
                    action = action.data.numpy().astype(int)[0]
                    a.append(action)
                    c_tens = list(itertools.chain(c_tens, [action]))

            stateval = Critic(torch.tensor([c_tens]).to(torch.device('cuda')))
            pgloss = None
            for (Policy, optima, action) in zip(Policy_List, actor_optims, a):
                optima.zero_grad()
                output = Policy(input_tensor)
                m = Categorical(output)
                pgloss = -m.log_prob(torch.tensor(action).to(torch.device('cuda'))) * (stateval.item()-rav)
                pgloss.backward()
                optima.step()
                # Send command to the Anchor Drones
                drone.set_direction(action)

                # Add action for the specific drone to the list
                commands.append(action)

        for a in range(len(memory)):
            transitions = memory.sample(1)
            batch = Transition(*zip(*transitions))
            state = batch.state
            next_state = batch.next_state
            actions = batch.action[0]#.cpu().detach().numpy()
            reward = batch.reward
            for (a, pg, optima) in zip(actions, Policy_List, actor_optims):
                probs = pg(torch.tensor([state]).to(torch.device('cuda')))
                m = Categorical(probs)
                pgloss = -m.log_prob(torch.tensor([a]).to(torch.device('cuda')))*reward[0]
                optima.zero_grad()
                pgloss.backward()
                optima.step()



    plt.figure()
    plt.title('Reward')
    plt.plot(reward_list, label='Reward')
    plt.plot(rlav, label='Moving Average')
    plt.show()

    plt.figure()
    plt.title('Critic Loss')
    plt.plot(cl, label='Critic Loss')
    plt.plot(running_reward_loss, label='Moving Average')
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
    main(wp, load=False)
