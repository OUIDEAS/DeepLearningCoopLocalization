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
        self.experience=[None]*self.memory_size
        self.current_index = 0
        self.size = 0

    def store(self, observation, action, reward, newobservation):
        self.experience[self.current_index] = (observation, action, reward, newobservation)
        self.current_index += 1
        self.size = min(self.size+1, self.memory_size)
        if self.current_index >= self.memory_size:
            self.current_index -= self.memory_size

    def sample(self):
        if self.size < self.minibatch_size:
            return []
        samples_index = np.floor(np.random.random((self.minibatch_size,))*self.size)
        samples = [self.experience[int(i)] for i in samples_index]
        return samples

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
        self.env = environment_AC.env()
        # print(self.env.observation_space)
        self.Actor = Actor(15, 7).to(self.device)#self.env.observation_space.n, self.env.action_space.n).to(self.device)
        self.Critic = Critic(18).to(self.device)#self.env.observation_space.n).to(self.device)
        self.actor_optim = optim.Adam(self.Actor.parameters(), lr = 1e-4, eps=1e-8)
        self.critic_optim = optim.Adam(self.Critic.parameters(), lr = 1e-2, eps=1e-8)
        self.loss = nn.L1Loss()
        self.state = []
        self.action = []
        self.reward = []
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
        while not end:
            new_state, reward, end, _ = self.env.step(action)#step_AC_continuous(action)
            next_output = self.Actor(torch.tensor(new_state).to(self.device))
            next_action = []
            for m in next_output:
                na = m.sample().cpu()
                next_action.append(na.data.numpy().astype(int))
            next_action = np.array(next_action)
            a2 = torch.from_numpy(next_action).to(self.device)
            self.actor_optim.zero_grad()
            ns = torch.tensor(new_state).to(self.device)
            ctens_current = torch.cat((torch.reshape(input,(1,-1)), torch.reshape(action,(1,-1))), -1).to(self.device)
            ctens_next = torch.cat((torch.reshape(torch.tensor(new_state).to(self.device),(1,-1)), torch.reshape(a2,(1,-1))), -1).to(self.device)
            # actor_loss = -1*torch.sum(self.Critic.forward(ctens))
            # actor_loss = -(output.log_prob(a1)*reward + self.gamma*next_output.log_prob(a2)*self.Critic(ctens))
            # actor_loss = -1*output.log_prob(a1)*(reward + self.gamma*self.Critic(torch.reshape(torch.tensor(new_state).to(self.device),(1,-1))))
            for i in range(3):

                # vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv<-- fine | not fine -->vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
                actor_loss = -1*output[i].log_prob(action[i]) * (reward + self.gamma*self.Critic(ctens_next))
                actor_loss.backward(retain_graph=True)
            self.actor_optim.step()
            self.state.append(state)
            self.action.append(action)
            self.reward.append(reward)

            state = new_state
            input = torch.tensor(state).to(self.device)
            output = self.Actor(input)
            action = []
            for m in output:
                m = m.sample().cpu()
                action.append(m.data.numpy().astype(int))
            action = np.array(action)
            action = torch.from_numpy(action).to(self.device)
        self.score.append(len(self.reward))
        self.train_critic()

    def train_critic(self):
        running_add = 0
        for i in reversed(range(len(self.reward))):
            running_add = running_add*self.gamma + self.reward[i]
            self.reward[i] = torch.tensor([running_add]).to(self.device)
        for i in range(len(self.state)-1):
            # Train the critic over multiple episodes to remove potential bias
            self.memory.store(torch.tensor(self.state[i]).to(torch.device('cpu')), self.action[i], self.reward[i], torch.tensor(self.state[i+1]).to(torch.device('cpu')))

        for i in range(500):
            values = self.memory.sample()
            state = torch.reshape(values[0][0], (1,-1)).to(self.device)
            action = torch.reshape(values[0][1], (1,-1)).to(self.device)
            ctens = torch.cat((state, action), -1)
            rewards = torch.reshape(values[0][2], (1,1)).to(self.device)
            # state = torch.tensor(self.state[i]).to(self.device)
            # ctens = torch.cat((torch.reshape(state,(1,-1)), torch.reshape(self.action[i],(1,1))), -1).to(self.device)
            output = self.Critic(ctens)
            self.critic_optim.zero_grad()
            # self.reward[i] = torch.reshape(self.reward[i], output.size())
            loss = self.loss(output, rewards)
            self.critic_loss.append(loss.item())
            loss.backward()
            self.critic_optim.step()
        self.reward = []
        self.state = []
        self.action = []



if __name__ == "__main__":
    os.system('clear')
    agent = ActorCritic(100)
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
    ema = lossValues.ewm(com=9).mean()
    plt.plot(x, episodeLength, label="Episode Lengths")
    x = np.linspace(0, agent.episodes, len(ema))
    # plt.plot(x, ema, label="Episode Lengths - smoothened")
    plt.legend()
    plt.show()
