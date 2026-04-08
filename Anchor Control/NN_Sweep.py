import wandb
import pprint
import torch
import torch.nn as nn
from NNLib import *
import matplotlib.pyplot as plt
import os
import time
from DLoaderRI import *
from torch.utils.data import DataLoader
import sys
import json
import csv
import numpy as np
import random as rand
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
from OLSsolver import *
from torch.distributions import Categorical
import environment

class OutputLayers(nn.Module):

    def __init__(self, layers, size):

        super().__init__()
        out = []
        for layer in range(layers):
            out.append(nn.Linear(size, size))
            out.append(nn.Tanh())

        out.append(nn.Linear(size, 7))
        self.outputlayers = nn.Sequential(*out)

    def forward(self, x):

        x = self.outputlayers(x)
        x = torch.nn.functional.softmax(x, dim=0)
        return Categorical(x)

class AnchorHeadingPolicy(nn.Module):

    def __init__(self, size, layers):

        super().__init__()
        out = 7
        rb = [torch.nn.Linear(36, size), nn.Tanh(), nn.LayerNorm(size)]
        for i in range(layers):
            rb.append(torch.nn.Linear(size, size))
            rb.append(nn.Tanh())

        self.Policy = nn.Sequential(*rb)
        self.out = nn.ModuleList([OutputLayers(layers, size) for i in range(10)])

    def forward(self,x):
        x = self.Policy(x)
        outputs = []
        for layer in self.out:
            outputs.append(layer(x))

        return outputs

def train(config=None):
    with wandb.init(config=config):

        config = wandb.config

        num_episode = config["episodes"]
        actor_learning_rate = config["lr"]
        gamma = config["gamma"]
        APATH = "Actor.pt"
        CPATH = "Critic.pt"
        batch_size = 1
        gamma = 0.95
        # memory = ReplayMemory(10000)
        cl = []
        replay_train = 10
        Actor = AnchorHeadingPolicy(config["fc_layer_size"], config["layers"]).to(torch.device("cuda"))

        print("************************** Created Network *************************************")
        if config["optimizer"] == "adam":
            print("ADAM")
            Actor_optim = torch.optim.Adam(Actor.parameters(), lr=config["lr"], eps = 1e-5)#config["learning_rate"])
        elif config["optimizer"] =="nadam":
            print("NADAM")
            Actor_optim = torch.optim.NAdam(Actor.parameters(), lr=config["lr"], eps = 1e-5)#config["learning_rate"])
        elif config["optimizer"]=="radam":
            print("RADAM")
            Actor_optim = torch.optim.RAdam(Actor.parameters(), lr=config["lr"], eps = 1e-5)#config["learning_rate"])
        elif config["optimizer"]=="RMSprop":
            print("RMSprop")
            Actor_optim = torch.optim.RMSprop(Actor.parameters(), lr=config["lr"], eps = 1e-5)#config["learning_rate"])
        print("************************** Configured optimizer *********************************")
        reward_list = []
        running_reward_loss = []
        rlav = []
        env = environment.env()
        print("************************** TRAINING ********************************************")
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
            Actor_optim.step()
            wandb.log({"Reward":rlav[len(rlav)-1]})
        Actor = Actor.to(torch.device('cpu'))
        Actor = None
        env = None

if __name__ == "__main__":
    os.system('clear')
    wandb.login()
    with open('sweep_data.json') as config_file:
        config = json.load(config_file)
        id = config["ID"]
    sweep_id = "rgeng98/Cooperative Localization NN/"+id
    wandb.agent(sweep_id, function=train, count=100)
