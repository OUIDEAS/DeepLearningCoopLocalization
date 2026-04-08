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
from torch.autograd import Variable
import time
import pickle
import numpy as np

def col_from_list(a):
    a1x, a1y, a1z = [], [], []
    for b in a:
        a1x.append(int(b[0]))
        a1y.append(int(b[1]))
        a1z.append(int(b[2]))

    return torch.tensor(a1x), torch.tensor(a1y), torch.tensor(a1z)

class Residual_Layers(nn.Module):
    def __init__(self, layers, size):
        super().__init__()
        self.layers = layers
        res = []
        for i in range(layers):
            res.append(nn.Linear(size, size))
            res.append(nn.PReLU())
            res.append(nn.LayerNorm(size))
        self.res = nn.Sequential(*res)
        self.apply(self.xav_init)

    def forward(self, x):
        return x + self.res(x)
    def rg_init(self, m):
        if isinstance(m, torch.nn.Linear):
            (fan_in, fan_out) = nn.init._calculate_fan_in_and_fan_out(m.weight)
            c = 1
            var = c/(self.layers*fan_in)
            nn.init.normal_(m.weight, 0, math.sqrt(var))
            m.bias.data.zero_()
    def xav_init(self, m):
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.xavier_normal_(m.weight)
            m.bias.data.fill_(0)

class OutputLayers(nn.Module):

    def __init__(self, layers, size):

        super().__init__()
        out = []
        # for layer in range(2):
        out.append(nn.Linear(size, size))
        out.append(nn.PReLU())
        out.append(nn.LayerNorm(size))

        self.outputlayers = nn.Sequential(*out)
        self.outx = nn.Linear(size, 9)
        self.outy = nn.Linear(size, 9)
        self.outz = nn.Linear(size, 9)


    def forward(self, x):
        x = self.outputlayers(x)
        outx = torch.nn.functional.softmax(self.outx(x), dim=0)
        outy = torch.nn.functional.softmax(self.outy(x), dim=0)
        outz = torch.nn.functional.softmax(self.outz(x), dim=0)
        return [Categorical(outx), Categorical(outy), Categorical(outz)]

class AnchorHeadingPolicy(nn.Module):

    def __init__(self, layernorm = True):

        super().__init__()
        size = 750
        layers = 1
        out = 7
        residuals = 7
        self.in_layers = nn.Sequential(*[torch.nn.Linear(36, size), nn.PReLU(),
                                         torch.nn.Linear(size, size), nn.PReLU()])#,
                            # torch.nn.Linear(size, size), nn.PReLU(), nn.LayerNorm(size)])

        # self.ResidualNetwork = nn.ModuleList([Residual_Layers(layers, size) for i in range(residuals)])
        self.out = nn.ModuleList([OutputLayers(layers, size) for i in range(10)])
        self.apply(self.kai_init)

    def forward(self,x):
        x = self.in_layers(x)
        # for layer in self.ResidualNetwork:
        #     x = layer(x)
        outputs = []
        for layer in self.out:
            outputs.append(layer(x))

        return outputs

    def xav_init(self, m):
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.xavier_normal_(m.weight)
            m.bias.data.fill_(0)

    def kai_init(self, m):
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.kaiming_normal_(m.weight)
            m.bias.data.fill_(0)

def imagetens(data):
    rows = []
    for i in range(6):
        my_row = []
        for a in range(10):
            my_row.append(data[0][6*i])
        for a in range(9):
            my_row.append(data[0][6*i+6+a*18])
            my_row.append(data[0][6*i+12+a*18])
            my_row.append(data[0][6*i+18+a*18])
        rows.append(my_row)

    return(torch.tensor([[rows]]))

def main(wp, load):
    os.system('clear')
    device = torch.device('cpu') # 'cuda' if torch.cuda.is_available() else 'cpu')
    print("Start: ", time.ctime())
    APATH = "Actor.pt"
    CPATH = "Critic.pt"
    num_episode = 100000
    batch_size = 5#00
    actor_learning_rate = 3e-5
    gamma = 0.99
    cl = []
    if load:
        open_file = open("Reward_History.pkl", "rb")
        rlav = pickle.load(open_file)
        open_file.close()
        Actor = torch.load(APATH)
    else:
        rlav = []
        Actor = AnchorHeadingPolicy().to(device)

    Actor_optim = torch.optim.RMSprop(Actor.parameters(), lr = actor_learning_rate, eps=1e-5)
    reward_list = []
    running_reward_loss = []
    env = environment.env()
    rewards = np.array([[]])
    states = []
    actions = []
    plot_rewards = []
    for i in range(num_episode):
        sys.stdout.write("\rTraining Episode: {0}/{1}".format(i+1, num_episode))
        sys.stdout.flush()
        end = False
        _ = env.reset()
        commands = []
        input_tensor = torch.tensor(env.pgnr).to(device)
        with torch.no_grad():
            output = Actor(input_tensor)
            for m in output:
                actionx = m[0].sample().cpu()
                actionx = actionx.data.numpy().astype(int)

                actiony = m[1].sample().cpu()
                actiony = actiony.data.numpy().astype(int)

                actionz = m[2].sample().cpu()
                actionz = actionz.data.numpy().astype(int)
                action = [actionx, actiony, actionz]
                commands.append(action)

        fs = True

        rewardholder = []
        while not end:
            state, reward, end, new_state = env.step_fc(commands)
            rewardholder.append(reward)
            states.append(torch.tensor(env.pgnr))

            commands = []
            input_tensor = torch.tensor(env.pgnr).to(device)
            with torch.no_grad():
                output = Actor(input_tensor)
                for (m, drone) in zip(output, env.drone_list):
                    for m in output:
                        actionx = m[0].sample().cpu()
                        actionx = actionx.data.numpy().astype(int)

                        actiony = m[1].sample().cpu()
                        actiony = actiony.data.numpy().astype(int)

                        actionz = m[2].sample().cpu()
                        actionz = actionz.data.numpy().astype(int)
                        action = [actionx, actiony, actionz]
                        commands.append(action)
            actions.append(commands)
        rewardholder = np.array(rewardholder)
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
        for a in reversed(range(len(rewardholder))):
            running_add1 = running_add1*gamma + rewardholder[a][0]
            rewardholder[a][0] = running_add1
            running_add2 = running_add2*gamma + rewardholder[a][1]
            rewardholder[a][1] = running_add2
            running_add3 = running_add3*gamma + rewardholder[a][2]
            rewardholder[a][2] = running_add3
            running_add4 = running_add4*gamma + rewardholder[a][3]
            rewardholder[a][3] = running_add4
            running_add5 = running_add5*gamma + rewardholder[a][4]
            rewardholder[a][4] = running_add5
            running_add6 = running_add6*gamma + rewardholder[a][5]
            rewardholder[a][5] = running_add6
            running_add7 = running_add7*gamma + rewardholder[a][6]
            rewardholder[a][6] = running_add7
            running_add8 = running_add8*gamma + rewardholder[a][7]
            rewardholder[a][7] = running_add8
            running_add9 = running_add9*gamma + rewardholder[a][8]
            rewardholder[a][8] = running_add9
            running_add0 = running_add0*gamma + rewardholder[a][9]
            rewardholder[a][9] = running_add0

            reward_list.append(sum(rewardholder[a])/len(rewardholder[a]))
            rlav.append(sum(reward_list[-100:])/len(reward_list[-100:]))
        # for a in range(len(states)):
        # for a in range(len(rewardholder)):
            # Most rewards are negative, which leads to a lower standard deviation which results in negative reward values exploding
            # This can be causing the policy to become overly deterministic too early
            # DreamerV3 Paper
            # for m in range(10):
            #     try:
            #         rewardholder[a][m] = float((rewardholder[a][m] - np.mean(rewardholder[-(len(rewardholder)-1):][m]))/np.std(rewardholder[-(len(rewardholder)-1):][m]))
            #     except:
            #         rewardholder[a][m] = rewardholder[a][m]
        if rewards.shape == (1,0):
            rewards = np.array(rewardholder)
        else:
            rewards = np.append(rewards, np.array(rewardholder), axis=0)
        rewardholder = []
        if i % batch_size == 0 and i != 0:
            state = torch.stack(states)
            # if i % 5 == 0:
            r1, r2, r3, r4, r5, r6, r7, r8, r9, r10 = [], [], [], [], [], [], [], [], [], []
            a1, a2, a3, a4, a5, a6, a7, a8, a9, a10 = [], [], [], [], [], [], [], [], [], []
            for a in range(len(states)):
                # for m in range(10):
                #     try:
                #         rewards[a][m] = float((rewardholder[a][m] - np.mean(rewardholder[-(len(rewardholder)-1):][m]))/np.std(rewardholder[-(len(rewardholder)-1):][m]))
                #     except:
                #         rewards[a][m] = rewardholder[a][m]
                r1.append(rewards[a][0])
                r2.append(rewards[a][1])
                r3.append(rewards[a][2])
                r4.append(rewards[a][3])
                r5.append(rewards[a][4])
                r6.append(rewards[a][5])
                r7.append(rewards[a][6])
                r8.append(rewards[a][7])
                r9.append(rewards[a][8])
                r10.append(rewards[a][9])

                a1.append(actions[a][0])
                a2.append(actions[a][1])
                a3.append(actions[a][2])
                a4.append(actions[a][3])
                a5.append(actions[a][4])
                a6.append(actions[a][5])
                a7.append(actions[a][6])
                a8.append(actions[a][7])
                a9.append(actions[a][8])
                a10.append(actions[a][9])

            Actor_optim.zero_grad()
            probs = Actor(Variable(state).to(device))

            a1x, a1y, a1z = col_from_list(a1)
            a2x, a2y, a2z = col_from_list(a2)
            a3x, a3y, a3z = col_from_list(a3)
            a4x, a4y, a4z = col_from_list(a4)
            a5x, a5y, a5z = col_from_list(a5)
            a6x, a6y, a6z = col_from_list(a6)
            a7x, a7y, a7z = col_from_list(a7)
            a8x, a8y, a8z = col_from_list(a8)
            a9x, a9y, a9z = col_from_list(a9)
            a10x, a10y, a10z = col_from_list(a10)

            pgloss1x = (-probs[0][0].log_prob(a1x.to(device)))
            pgloss1y = (-probs[0][1].log_prob(a1y.to(device)))
            pgloss1z = (-probs[0][2].log_prob(a1z.to(device)))

            pgloss2x = (-probs[1][0].log_prob(a2x.to(device)))
            pgloss2y = (-probs[1][1].log_prob(a2y.to(device)))
            pgloss2z = (-probs[1][2].log_prob(a2z.to(device)))

            pgloss3x = (-probs[2][0].log_prob(a3x.to(device)))
            pgloss3y = (-probs[2][1].log_prob(a3y.to(device)))
            pgloss3z = (-probs[2][2].log_prob(a3z.to(device)))

            pgloss4x = (-probs[3][0].log_prob(a4x.to(device)))
            pgloss4y = (-probs[3][1].log_prob(a4y.to(device)))
            pgloss4z = (-probs[3][2].log_prob(a4z.to(device)))

            pgloss5x = (-probs[4][0].log_prob(a5x.to(device)))
            pgloss5y = (-probs[4][1].log_prob(a5y.to(device)))
            pgloss5z = (-probs[4][2].log_prob(a5z.to(device)))

            pgloss6x = (-probs[5][0].log_prob(a6x.to(device)))
            pgloss6y = (-probs[5][1].log_prob(a6y.to(device)))
            pgloss6z = (-probs[5][2].log_prob(a6z.to(device)))

            pgloss7x = (-probs[6][0].log_prob(a7x.to(device)))
            pgloss7y = (-probs[6][1].log_prob(a7y.to(device)))
            pgloss7z = (-probs[6][2].log_prob(a7z.to(device)))

            pgloss8x = (-probs[7][0].log_prob(a8x.to(device)))
            pgloss8y = (-probs[7][1].log_prob(a8y.to(device)))
            pgloss8z = (-probs[7][2].log_prob(a8z.to(device)))

            pgloss9x = (-probs[8][0].log_prob(a9x.to(device)))
            pgloss9y = (-probs[8][1].log_prob(a9y.to(device)))
            pgloss9z = (-probs[8][2].log_prob(a9z.to(device)))

            pgloss10x = (-probs[9][0].log_prob(a10x.to(device)))
            pgloss10y = (-probs[9][1].log_prob(a10y.to(device)))
            pgloss10z = (-probs[9][2].log_prob(a10z.to(device)))

            for i in range(len(r1)):
                pgloss1x[i] = pgloss1x[i]*r1[i]
                pgloss1y[i] = pgloss1y[i]*r1[i]
                pgloss1z[i] = pgloss1z[i]*r1[i]

                # pgloss1x[i].backward(retain_graph=True)
                # pgloss1y[i].backward(retain_graph=True)
                # pgloss1z[i].backward(retain_graph=True)

                pgloss2x[i] = pgloss2x[i]*r2[i]
                pgloss2y[i] = pgloss2y[i]*r2[i]
                pgloss2z[i] = pgloss2z[i]*r2[i]

                # pgloss2x[i].backward(retain_graph=True)
                # pgloss2y[i].backward(retain_graph=True)
                # pgloss2z[i].backward(retain_graph=True)

                pgloss3x[i] = pgloss3x[i]*r3[i]
                pgloss3y[i] = pgloss3y[i]*r3[i]
                pgloss3z[i] = pgloss3z[i]*r3[i]

                # pgloss3x[i].backward(retain_graph=True)
                # pgloss3y[i].backward(retain_graph=True)
                # pgloss3z[i].backward(retain_graph=True)

                pgloss4x[i] = pgloss4x[i]*r4[i]
                pgloss4y[i] = pgloss4y[i]*r4[i]
                pgloss4z[i] = pgloss4z[i]*r4[i]

                # pgloss4x[i].backward(retain_graph=True)
                # pgloss4y[i].backward(retain_graph=True)
                # pgloss4z[i].backward(retain_graph=True)

                pgloss5x[i] = pgloss5x[i]*r5[i]
                pgloss5y[i] = pgloss5y[i]*r5[i]
                pgloss5z[i] = pgloss5z[i]*r5[i]

                # pgloss5x[i].backward(retain_graph=True)
                # pgloss5y[i].backward(retain_graph=True)
                # pgloss5z[i].backward(retain_graph=True)

                pgloss6x[i] = pgloss6x[i]*r6[i]
                pgloss6y[i] = pgloss6y[i]*r6[i]
                pgloss6z[i] = pgloss6z[i]*r6[i]

                # pgloss6x[i].backward(retain_graph=True)
                # pgloss6y[i].backward(retain_graph=True)
                # pgloss6z[i].backward(retain_graph=True)

                pgloss7x[i] = pgloss7x[i]*r7[i]
                pgloss7y[i] = pgloss7y[i]*r7[i]
                pgloss7z[i] = pgloss7z[i]*r7[i]

                # pgloss7x[i].backward(retain_graph=True)
                # pgloss7y[i].backward(retain_graph=True)
                # pgloss7z[i].backward(retain_graph=True)

                pgloss8x[i] = pgloss8x[i]*r8[i]
                pgloss8y[i] = pgloss8y[i]*r8[i]
                pgloss8z[i] = pgloss8z[i]*r8[i]

                # pgloss8x[i].backward(retain_graph=True)
                # pgloss8y[i].backward(retain_graph=True)
                # pgloss8z[i].backward(retain_graph=True)

                pgloss9x[i] = pgloss9x[i]*r9[i]
                pgloss9y[i] = pgloss9y[i]*r9[i]
                pgloss9z[i] = pgloss9z[i]*r9[i]
                #
                # pgloss9x[i].backward(retain_graph=True)
                # pgloss9y[i].backward(retain_graph=True)
                # pgloss9z[i].backward(retain_graph=True)

                pgloss10x[i] = pgloss10x[i]*r10[i]
                pgloss10y[i] = pgloss10y[i]*r10[i]
                pgloss10z[i] = pgloss10z[i]*r10[i]

                # pgloss10x[i].backward(retain_graph=True)
                # pgloss10y[i].backward(retain_graph=True)
                # pgloss10z[i].backward(retain_graph=True)


            pgloss1x = pgloss1x.sum()
            pgloss1y = pgloss1y.sum()
            pgloss1z = pgloss1z.sum()

            pgloss2x = pgloss2x.sum()
            pgloss2y = pgloss2y.sum()
            pgloss2z = pgloss2z.sum()

            pgloss3x = pgloss3x.sum()
            pgloss3y = pgloss3y.sum()
            pgloss3z = pgloss3z.sum()

            pgloss4x = pgloss4x.sum()
            pgloss4y = pgloss4y.sum()
            pgloss4z = pgloss4z.sum()

            pgloss5x = pgloss5x.sum()
            pgloss5y = pgloss5y.sum()
            pgloss5z = pgloss5z.sum()

            pgloss6x = pgloss6x.sum()
            pgloss6y = pgloss6y.sum()
            pgloss6z = pgloss6z.sum()

            pgloss7x = pgloss7x.sum()
            pgloss7y = pgloss7y.sum()
            pgloss7z = pgloss7z.sum()

            pgloss8x = pgloss8x.sum()
            pgloss8y = pgloss8y.sum()
            pgloss8z = pgloss8z.sum()

            pgloss9x = pgloss9x.sum()
            pgloss9y = pgloss9y.sum()
            pgloss9z = pgloss9z.sum()

            pgloss10x = pgloss10x.sum()
            pgloss10y = pgloss10y.sum()
            pgloss10z = pgloss10z.sum()

            pgloss1x.backward(retain_graph=True)
            pgloss1y.backward(retain_graph=True)
            pgloss1z.backward(retain_graph=True)

            pgloss2x.backward(retain_graph=True)
            pgloss2y.backward(retain_graph=True)
            pgloss2z.backward(retain_graph=True)

            pgloss3x.backward(retain_graph=True)
            pgloss3y.backward(retain_graph=True)
            pgloss3z.backward(retain_graph=True)

            pgloss4x.backward(retain_graph=True)
            pgloss4y.backward(retain_graph=True)
            pgloss4z.backward(retain_graph=True)

            pgloss5x.backward(retain_graph=True)
            pgloss5y.backward(retain_graph=True)
            pgloss5z.backward(retain_graph=True)

            pgloss6x.backward(retain_graph=True)
            pgloss6y.backward(retain_graph=True)
            pgloss6z.backward(retain_graph=True)

            pgloss7x.backward(retain_graph=True)
            pgloss7y.backward(retain_graph=True)
            pgloss7z.backward(retain_graph=True)

            pgloss8x.backward(retain_graph=True)
            pgloss8y.backward(retain_graph=True)
            pgloss8z.backward(retain_graph=True)

            pgloss9x.backward(retain_graph=True)
            pgloss9y.backward(retain_graph=True)
            pgloss9z.backward(retain_graph=True)

            pgloss10x.backward(retain_graph=True)
            pgloss10y.backward(retain_graph=True)
            pgloss10z.backward(retain_graph=True)

            for param in Actor.parameters():
                param.grad.data.clamp_(-1, 1)
            Actor_optim.step()

            # loss = torch.mean(torch.stack([pgloss1x, pgloss1y, pgloss1z,
            #                                pgloss2x, pgloss2y, pgloss2z,
            #                                pgloss3x, pgloss3y, pgloss3z,
            #                                pgloss4x, pgloss4y, pgloss4z,
            #                                pgloss5x, pgloss5y, pgloss5z,
            #                                pgloss6x, pgloss6y, pgloss6z,
            #                                pgloss7x, pgloss7y, pgloss7z,
            #                                pgloss8x, pgloss8y, pgloss8z,
            #                                pgloss9x, pgloss9y, pgloss9z,
            #                                pgloss10x, pgloss10y, pgloss10z]))
            # loss.backward()
            plot_rewards.append(np.mean(rewards))

            torch.save(Actor, APATH)
            reward_list = []
            running_reward_loss = []
            rewards = np.array([[]])
            states = []
            actions = []
            print('\nStepped Optimizer')

        # rewards = list(rewards)
        # r = []
        # for j in rewards:
        #     r.append(list(j))
        # rewards = r
    torch.save(Actor, APATH)
    os.system('clear')
    # Goal.Horn()
    os.system('clear')
    x = np.linspace(0, num_episode, len(plot_rewards))
    plt.figure()
    plt.title('Reward')
    plt.plot(x, plot_rewards, label='Moving Average')
    plt.show()
    open_file = open("Reward_History.pkl", "wb")
    pickle.dump(rlav, open_file)
    open_file.close()

    # Generate Plot
    end = False
    _ = env.reset()
    commands = []
    input_tensor = torch.tensor([env.pgnr]).to(device)
    with torch.no_grad():
        output = Actor(input_tensor)
        for m in output:
            actionx = m[0].sample().cpu()
            actionx = actionx.data.numpy().astype(int)

            actiony = m[1].sample().cpu()
            actiony = actiony.data.numpy().astype(int)

            actionz = m[2].sample().cpu()
            actionz = actionz.data.numpy().astype(int)
            action = [actionx, actiony, actionz]
            commands.append(action)

    fs = True
    x_tru, y_tru, z_tru = [], [], []
    x, y, z = [], [], []
    while not end:
        state, reward, end, new_state = env.step_fc(commands)
        x_tru.append(env.agent.x_tru)
        y_tru.append(env.agent.y_tru)
        z_tru.append(env.agent.z_tru)
        x.append(env.agentx)
        y.append(env.agenty)
        z.append(env.agentz)
        commands = []
        input_tensor = torch.tensor([env.pgnr]).to(device)
        with torch.no_grad():
            output = Actor(input_tensor)
            for m in output:
                actionx = m[0].sample().cpu()
                actionx = actionx.data.numpy().astype(int)

                actiony = m[1].sample().cpu()
                actiony = actiony.data.numpy().astype(int)

                actionz = m[2].sample().cpu()
                actionz = actionz.data.numpy().astype(int)
                action = [actionx, actiony, actionz]
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
