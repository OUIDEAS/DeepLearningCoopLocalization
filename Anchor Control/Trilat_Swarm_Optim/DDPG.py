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
import sys
import os
import environment_AC

class ReplayMemory():
    def __init__(self, memory_size, batch_size):
        self.memory_size = memory_size
        self.minibatch_size = batch_size
        self.state=[None]*self.memory_size
        self.action=[None]*self.memory_size
        self.reward=[None]*self.memory_size
        self.new_state=[None]*self.memory_size
        self.current_index = 0
        self.size = 0

    def __len__(self):
        return(len(self.state))

    def store(self, observation, action, reward, newobservation):
        self.state[self.current_index]=observation
        self.action[self.current_index]=action
        self.reward[self.current_index]=reward
        self.new_state[self.current_index]=newobservation
        self.current_index += 1
        self.size = min(self.size+1, self.memory_size)
        if self.current_index >= self.memory_size:
            self.current_index -= self.memory_size

    def sample(self):
        if self.size < self.minibatch_size:
            return []
        samples_index = np.floor(np.random.random((self.minibatch_size,))*self.size)
        states = [self.state[int(i)] for i in samples_index]
        action = [self.action[int(i)] for i in samples_index]
        reward = [self.reward[int(i)] for i in samples_index]
        newstate = [self.new_state[int(i)] for i in samples_index]
        return states, action, reward, newstate

class Actor(nn.Module):
    def __init__(self, inputs: int, outputs:int):
        super().__init__()
        size = 500
        actor = [nn.Linear(inputs, size), nn.PReLU(),
                 nn.Linear(size, size), nn.PReLU()]
        self.Actor = nn.Sequential(*actor)
        self.out1 = nn.Linear(size, outputs)
        self.out2 = nn.Linear(size, outputs)
        self.out3 = nn.Linear(size, outputs)
        self.apply(self.kai_init)

    def forward(self, x):
        x = self.Actor(x)
        drone1 = Categorical(torch.nn.functional.softmax(self.out1(x), dim=0))
        drone2 = Categorical(torch.nn.functional.softmax(self.out2(x), dim=0))
        drone3 = Categorical(torch.nn.functional.softmax(self.out3(x), dim=0))
        return [drone1, drone2, drone3]

    def kai_init(self, m):
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.kaiming_normal_(m.weight)
            m.bias.data.fill_(0)

class Critic(nn.Module):

    def __init__(self, inputs:int):
        super().__init__()
        size = 500
        critic = [nn.Linear(inputs, size), nn.PReLU(),
                 nn.Linear(size, size), nn.PReLU(),
                 nn.Linear(size, 1)]
        self.Critic = nn.Sequential(*critic)
        self.apply(self.kai_init)

    def forward(self, x):
        return self.Critic(x)

    def kai_init(self, m):
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.kaiming_normal_(m.weight)
            m.bias.data.fill_(0)

class ActorCritic():
    def __init__(self, episodes):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # self.env = environment
        self.episodes = episodes
        self.gamma = 0.99
        self.tau = 0.0001
        self.env = environment_AC.env()
        # print(self.env.observation_space)
        self.Actor = Actor(15, 7).to(self.device)#self.env.observation_space.n, self.env.action_space.n).to(self.device)
        self.actor_target = Actor(15,7).to(self.device)

        self.Critic = Critic(18).to(self.device)#self.env.observation_space.n).to(self.device)
        self.critic_target = Critic(18).to(self.device)

        for target_param, param in zip(self.actor_target.parameters(), self.Actor.parameters()):
            target_param.data.copy_(param.data)
            target_param.requires_grad = False

        for target_param, param in zip(self.critic_target.parameters(), self.Critic.parameters()):
            target_param.data.copy_(param.data)
            target_param.requires_grad = False
        self.actor_optim = optim.Adam(self.Actor.parameters(), lr = 1e-4, eps=1e-8)
        self.critic_optim = optim.Adam(self.Critic.parameters(), lr = 1e-2, eps=1e-8)
        self.loss = nn.L1Loss()
        self.state = []
        self.action = []
        self.reward, self.rewards = [], []
        self.new_state = []
        self.score = []
        self.critic_loss = []
        self.memory = ReplayMemory(memory_size=1000, batch_size=1)

    def train(self):
         for i in range(self.episodes):
             sys.stdout.write("\rEpisode: {0}/{1}".format(i+1, self.episodes))
             sys.stdout.flush()
             self.run_episode(i)


    def run_episode(self, ep_num):
        state = self.env.reset()

        input = torch.tensor(state).to(self.device)
        output = self.Actor(input)
        action = []
        for m in output:
            m = m.sample().cpu()
            action.append(m.data.numpy().astype(int))
        action = np.array(action)
        action = torch.from_numpy(action).to(self.device)

        end = False
        iter = 0
        rewards = []
        while not end:
            new_state, reward, end, _ = self.env.step(action)#step_AC_continuous(action)

            self.reward.append(reward)
            self.state.append(input)
            self.action.append(action)
            self.new_state.append(torch.tensor(new_state).to(self.device))
            rewards.append(reward)
            state = new_state
            input = torch.tensor(state).to(self.device)
            output = self.Actor(input)
            action = []
            for m in output:
                m = m.sample().cpu()
                action.append(m.data.numpy().astype(int))
            action = np.array(action)
            action = torch.from_numpy(action).to(self.device)
        self.train_networks()

    def train_networks(self):
        running_r = 0
        for i in reversed(range(len(self.reward))):
            running_r = self.reward[i] + running_r*self.gamma
            self.reward[i] = running_r

        for i in range(len(self.reward)):
            self.memory.store(self.state[i], self.action[i], self.reward[i], self.new_state[i])

        i = 500
        if len(self.memory) < i:
            i = len(self.memory)

        for i in range(500):
            state, action, reward, new_state = self.memory.sample()

            ctens_curr = torch.cat((state[0], action[0]), -1)
            next_probs = self.actor_target(new_state[0])
            next_action = []
            for m in next_probs:
                na = m.sample().cpu()
                next_action.append(na.data.numpy().astype(int))
            next_action = np.array(next_action)
            a2 = torch.from_numpy(next_action).to(self.device)

            ctens_next = torch.cat((new_state[0], a2), -1)

            critic_target = reward[0] + self.gamma*self.critic_target(ctens_next)
            critic_value = self.Critic(ctens_curr)
            closs = self.loss(critic_value, critic_target)
            self.critic_optim.zero_grad()
            closs.backward()
            self.critic_optim.step()
            self.critic_loss.append(closs.item())


            actor_loss = -1 * self.Critic(ctens_curr)
            self.actor_optim.zero_grad()
            actor_loss.backward()
            self.actor_optim.step()

            for target_param, param in zip(self.actor_target.parameters(), self.Actor.parameters()):
                target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))
            for target_param, param in zip(self.critic_target.parameters(), self.Critic.parameters()):
                target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))
        self.rewards.append(sum(self.reward)/len(self.reward))
        self.reward = []
        self.state = []
        self.action = []
        self.new_state = []


if __name__ == "__main__":
    os.system('clear')
    agent = ActorCritic(20)
    agent.train()
    print("\n")
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
    x = np.linspace(0, agent.episodes, len(agent.rewards))
    episodeLength = pd.DataFrame(
    	{'episodeLength': agent.rewards})
    ema = lossValues.ewm(com=9).mean()
    plt.plot(x, episodeLength, label="Episode Lengths")
    x = np.linspace(0, agent.reward, len(ema))
    # plt.plot(x, ema, label="Episode Lengths - smoothened")
    plt.legend()
    plt.show()
