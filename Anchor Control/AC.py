import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
# Test this on openAI gym to ensure the algorithm is correct before moving into the custom environment
import gym
# import necessary packages
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


class Actor(nn.Module):
    def __init__(self, inputs: int, outputs:int):
        super().__init__()
        actor = [nn.Linear(inputs, 512), nn.PReLU(),
                 nn.Linear(512, 512), nn.PReLU(),
                 nn.Linear(512, outputs)]
        self.Actor = nn.Sequential(*actor)
        self.apply(self.kai_init)

    def forward(self, x):
        return Categorical(torch.nn.functional.softmax(self.Actor(x), dim=0))

    def kai_init(self, m):
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.kaiming_normal_(m.weight)
            m.bias.data.fill_(0)

class Critic(nn.Module):
    def __init__(self, inputs:int):
        super().__init__()
        critic = [nn.Linear(inputs+1, 512), nn.PReLU(),
                 nn.Linear(512, 512), nn.PReLU(),
                 nn.Linear(512, 1)]
        self.Critic = nn.Sequential(*critic)
        self.apply(self.kai_init)

    def forward(self, x):
        return self.Critic(x)

    def kai_init(self, m):
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.kaiming_normal_(m.weight)
            m.bias.data.fill_(0)

class ActorCritic():
    def __init__(self, environment, episodes):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # self.env = environment
        self.episodes = episodes
        self.gamma = 0.99
        self.env = gym.make(environment)
        # print(self.env.observation_space)
        self.Actor = Actor(4, 2).to(self.device)#self.env.observation_space.n, self.env.action_space.n).to(self.device)
        self.Critic = Critic(4).to(self.device)#self.env.observation_space.n).to(self.device)
        self.actor_optim = optim.Adam(self.Actor.parameters(), lr = 1e-5, eps=1e-8)
        self.critic_optim = optim.Adam(self.Critic.parameters(), lr = 1e-2, eps=1e-8)
        self.loss = nn.L1Loss()
        self.state = []
        self.action = []
        self.reward = []
        self.score = []
        self.critic_loss = []

    def train(self):
         for i in range(self.episodes):
             self.run_episode(i)

    def run_episode(self, ep_num):
        state = self.env.reset()

        input = torch.tensor(state).to(self.device)
        output = self.Actor(input)
        action = output.sample().cpu()
        action = action.data.numpy().astype(int)
        end = False
        iter = 0
        while not end:
            new_state, reward, end, _ = self.env.step(action)
            next_output = self.Actor(torch.tensor(new_state).to(self.device))
            na = next_output.sample().cpu()
            next_action = na.data.numpy().astype(int)
            a1 = torch.tensor(action).to(self.device)
            a2 = torch.tensor(next_action).to(self.device)
            self.actor_optim.zero_grad()
            ns = torch.tensor(new_state).to(self.device)
            ctens = torch.cat((torch.reshape(ns,(1,-1)), torch.reshape(a2,(1,1))), -1).to(self.device)
            actor_loss = -(output.log_prob(a1)*reward + self.gamma*next_output.log_prob(a2)*self.Critic(ctens))
            actor_loss.backward()
            self.actor_optim.step()
            self.state.append(state)
            self.action.append(a1)
            self.reward.append(reward)

            state = new_state
            input = torch.tensor(state).to(self.device)
            output = self.Actor(input)
            action = output.sample().cpu()
            action = action.data.numpy().astype(int)
        self.score.append(len(self.reward))
        self.train_critic()

    def train_critic(self):
        running_add = 0
        for i in reversed(range(len(self.reward))):
            running_add = running_add*self.gamma + self.reward[i]
            self.reward[i] = torch.tensor([running_add]).to(self.device)
        for i in range(len(self.state)):
            state = torch.tensor(self.state[i]).to(self.device)
            ctens = torch.cat((torch.reshape(state,(1,-1)), torch.reshape(self.action[i],(1,1))), -1).to(self.device)
            output = self.Critic(ctens)
            self.critic_optim.zero_grad()
            self.reward[i] = torch.reshape(self.reward[i], output.size())
            loss = self.loss(output, self.reward[i])
            self.critic_loss.append(loss.item())
            loss.backward()
            self.critic_optim.step()
        self.reward = []
        self.state = []
        self.action = []



if __name__ == "__main__":
    agent = ActorCritic("CartPole-v1", 2000)
    agent.train()
    x = np.linspace(0, agent.episodes, len(agent.critic_loss))
    lossValues = pd.DataFrame(
    	{'lossValues': agent.critic_loss})
    ema = lossValues.ewm(com=9).mean()
    plt.figure(1)
    plt.plot(x, lossValues, label="Critic Loss")
    x = np.linspace(0, agent.episodes, len(ema))
    plt.plot(x, ema, label="Critic Loss - smoothened")
    plt.legend()

    plt.figure(2)
    x = np.linspace(0, agent.episodes, len(agent.score))
    episodeLength = pd.DataFrame(
    	{'episodeLength': agent.score})
    ema = lossValues.ewm(com=0.4).mean()
    plt.plot(x, episodeLength, label="Episode Lengths")
    x = np.linspace(0, agent.episodes, len(ema))
    # plt.plot(x, ema, label="Episode Lengths - smoothened")
    plt.legend()
    plt.show()
