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
from statistics import mean

class Critic_net(nn.Module):
    def __init__(self):
        super().__init__()
        size = 500
        layers = 2
        rb = [torch.nn.Linear(106, size), nn.ReLU(), nn.LayerNorm(size)]
        for i in range(layers):
            rb.append(torch.nn.Linear(size, size))
            rb.append(nn.ReLU())
        rb.append(nn.Linear(size, 1))
        self.network = nn.Sequential(*rb)
    def forward(self, x, action):
        for i in action:
            x = torch.cat((x, i.to(torch.device('cuda'))), 0) # torch.tensor(x).to(torch.device('cuda'))
        return self.network(x)


class OutputLayers(nn.Module):
    def __init__(self, layers, size):
        super().__init__()
        out = []
        for layer in range(layers):
            out.append(nn.Linear(size, size))
            out.append(nn.ReLU())
        out.append(nn.Linear(size, 7))
        self.outputlayers = nn.Sequential(*out)
    def forward(self, x):
        x = self.outputlayers(x)
        return x

class Actor_net(nn.Module):
    def __init__(self):
        super().__init__()
        size = 1500
        layers = 2
        out = 7
        rb = [torch.nn.Linear(36, size), nn.ReLU(), nn.LayerNorm(size)]
        for i in range(layers):
            rb.append(torch.nn.Linear(size, size))
            rb.append(nn.Tanh())
            rb.append(nn.LayerNorm(size))
        self.Policy = nn.Sequential(*rb)
        self.out = nn.ModuleList([OutputLayers(layers, size) for i in range(10)])
    def forward(self,x):
        x = self.Policy(x)
        outputs = []
        for layer in self.out:
            outputs.append(layer(x))
        return outputs

class DDPG_Agent():
    def __init__(self, load):
        self.device = torch.device('cuda')
        if load:
            self.load()
        else:
            self.Actor = Actor_net().to(self.device)
            self.Critic = Critic_net().to(self.device)
        self.Actor_Target = Actor_net().to(self.device)
        self.Critic_Target = Critic_net().to(self.device)
        for target_param, param in zip(self.Actor_Target.parameters(), self.Actor.parameters()):
            target_param.data.copy_(param.data)
        for target_param, param in zip(self.Critic_Target.parameters(), self.Critic.parameters()):
            target_param.data.copy_(param.data)
        self.env = environment.env()
        self.num_episode = 50#5000
        self.batch_size = 1
        self.actor_learning_rate = 0.0001
        self.critic_learning_rate = 0.001
        self.gamma = 0.95
        self.Actor_optim = torch.optim.RMSprop(self.Actor.parameters(), lr = self.actor_learning_rate, eps=1e-5)
        self.Critic_optim = torch.optim.Adam(self.Critic.parameters(), lr=self.critic_learning_rate)
        self.state = []
        self.next_state = []
        self.action = []
        self.reward = []
        self.rlav = []
        self.rewards = []
        self.selected_actions = []
        self.critic_loss = nn.L1Loss()
        self.tau = 0.0001
        self.critic_results = []

    def save(self):
        torch.save(self.Actor, "ddpgActor.pt")
        torch.save(self.Critic, "ddpgCritic.pt")

    def load(self):
        self.Actor = torch.load("ddpgActor.pt")
        self.Critic = torch.load("ddpgCritic.pt")
    def train(self):
        for i in range(self.num_episode):
            sys.stdout.write("\rEpisode: {0}/{1}".format(i+1, self.num_episode))
            sys.stdout.flush()
            self.run_episode()
            self.Update_Params()
        self.save()
        x = np.linspace(0, self.num_episode, len(self.rewards))
        x2 = np.linspace(0, self.num_episode, len(self.rlav))
        plt.figure()
        plt.plot(x, self.rewards)
        plt.plot(x2, self.rlav)
        plt.show()
        plt.figure()
        plt.plot(self.critic_results)
        plt.show()

    def run_episode(self):
        self.env.reset()
        commands = []
        input_tensor = torch.tensor(self.env.pgnr).to(self.device)
        policy_out = []
        with torch.no_grad():
            output = self.Actor(input_tensor)
            for m in output:
                policy_out.append(m)
                m = nn.functional.softmax(m, dim=0)
                m = Categorical(m)
                action = m.sample().cpu()
                action = action.data.numpy().astype(int)
                commands.append(action)
        fs = True
        rewards = []
        states = []
        actions = []
        a = []
        end = False
        while not end:
            state, reward, end, new_state = self.env.step(commands)
            self.state.append(state)
            self.reward.append(reward)
            self.rewards.append(mean(reward))
            self.rlav.append(sum(self.rewards[-100:])/len(self.rewards[-100:]))
            self.next_state.append(new_state)
            self.action.append(policy_out)
            commands = []
            input_tensor = torch.tensor(self.env.pgnr).to(self.device)
            policy_out = []
            with torch.no_grad():
                output = self.Actor(input_tensor)
                for (m, drone) in zip(output, self.env.drone_list):
                    mask = drone.set_mask()
                    policy_out.append(m)
                    m = nn.functional.softmax(m, dim=0)
                    m = Categorical(m)
                    action = m.sample().cpu()
                    commands.append(action)
            self.selected_actions.append(commands)

    def Update_Params(self):
        self.state = self.state[-100:]
        self.action = self.action[-100:]
        self.reward = self.reward[-100:]
        self.next_state = self.next_state[-100:]
        self.selected_sctions = self.selected_actions[-100:]
        for i in range(len(self.state)):
            index = random.randrange(0,len(self.state)-1)
            state = torch.tensor(self.state[index]).to(self.device)
            action = self.action[index]
            reward = mean(self.reward[index])
            command = self.selected_actions[index]
            next_state = torch.tensor(self.next_state[index]).to(self.device)
            Qval = self.Critic(state, action)
            next_action = self.Actor_Target(next_state)
            next_q = self.Critic_Target(next_state, next_action)
            Q_target = reward + self.gamma*next_q
            critic_loss = self.critic_loss(Qval, Q_target)
            self.Critic_optim.zero_grad()
            critic_loss.backward()
            self.Critic_optim.step()
            self.critic_results.append(critic_loss.item())
            probs = self.Actor(Variable(state).to(self.device))
            self.Actor_optim.zero_grad()
            c = 0
            for prob in probs:
                prob = F.softmax(prob, dim = 0)
                prob = Categorical(prob)
                pgloss = -prob.log_prob(command[c].to(torch.device('cuda')))*(reward + self.gamma*self.Critic(next_state, action) - self.Critic(state, action))
                pgloss.backward(retain_graph=True)
                c = c+1
            for param in self.Actor.parameters():
                param.grad.data.clamp_(-1, 1)
            self.Actor_optim.step()

            #policy_loss = -self.Critic(state, probs)
            #self.Actor_optim.zero_grad()
            # policy_loss.backward()
            # self.Actor_optim.step()
            for target_param, param in zip(self.Actor_Target.parameters(), self.Actor.parameters()):
                target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))
            for target_param, param in zip(self.Critic_Target.parameters(), self.Critic.parameters()):
                target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))

if __name__ == "__main__":
    os.system('clear')
    Agent = DDPG_Agent(load = False)
    Agent.train()
