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
from environment import env
import utils
# Try this on the cooperative localization environment, and if it doesn't work, try only training network when reward is positive
# Seems to be working so far on Cart-Pole
class ReplayMemory():
    def __init__(self, memory_size):
        self.memory_size = memory_size
        self.state = [None]*self.memory_size
        self.action = [None]*self.memory_size
        self.reward = [None]*self.memory_size
        self.new_state = [None]*self.memory_size
        self.current_index = 0
        self.size = 0

    def store(self, observation, action, reward, newobservation):
        self.state[self.current_index] = observation
        self.action[self.current_index] = action
        self.reward[self.current_index] = reward
        self.new_state[self.current_index] = newobservation
        self.current_index += 1
        self.size = min(self.size+1, self.memory_size)
        if self.current_index >= self.memory_size:
            self.current_index -= self.memory_size

    def sample(self, batch_size):
        if self.size < batch_size:
            batch_size = self.size
        samples_index = np.floor(np.random.random((batch_size,))*self.size)
        state = np.array([[self.state[int(i)].numpy() for i in samples_index]])
        action = np.array([[self.action[int(i)].numpy() for i in samples_index]])
        reward = np.array([[self.reward[int(i)].numpy() for i in samples_index]])
        new_state = np.array([[self.new_state[int(i)].numpy() for i in samples_index]])
        return torch.from_numpy(state), torch.from_numpy(action), torch.from_numpy(reward), torch.from_numpy(new_state)

class Actor(nn.Module):
    def __init__(self, inputs: int, outputs:int = 30):
        super().__init__()
        size = 256
        actor = [nn.Linear(inputs, size), nn.PReLU(),
                 nn.Linear(size, size), nn.PReLU(),
                 nn.Linear(size, outputs)]
        self.Actor = nn.Sequential(*actor)
        self.apply(self.kai_init)

    def set_action_limit(self, x):
        self.action_limit = x

    def forward(self, x):
        return self.action_limit * torch.nn.functional.tanh(self.Actor(x), dim=0)

    def kai_init(self, m):
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.kaiming_normal_(m.weight)
            m.bias.data.fill_(0)

class Critic(nn.Module):

    def __init__(self, inputs:int):
        super().__init__()
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
    def __init__(self, environment, episodes, max_memory):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # self.env = environment
        self.episodes = episodes
        self.gamma, self.tau = 0.99, 0.001
        self.env = environment
        _ = self.env.reset()
        obs_space = len(self.env.pgnr)
        # print(self.env.observation_space)
        action_space = 30
        self.Actor = Actor(obs_space, action_space).to(self.device)#self.env.observation_space.n, self.env.action_space.n).to(self.device)
        self.Critic = Critic(obs_space).to(self.device)#self.env.observation_space.n).to(self.device)
        self.Actor_target = Actor(obs_space, action_space).to(self.device)
        self.Critic_target = Critic(obs_space).to(self.device)
        utils.hard_update(self.Actor_target, self.Actor)
        utils.hard_update(self.Critic_target, self.Critic)
        self.noise = utils.OrnsteinUhlenbeckActionNoise(30)
        self.ram = ReplayMemory(max_memory)
        self.actor_optim = optim.Adam(self.Actor.parameters(), lr = 1e-6, eps=1e-8)
        self.critic_optim = optim.Adam(self.Critic.parameters(), lr = 1e-3, eps=1e-8)
        self.loss = nn.L1Loss()
        self.reward = []
        self.critic_loss = []
        self.average_episode_score = []
        self.action_limit = 5
        self.batch_size = 128

    def explore_action(self, state):
        state = Variable(torch.tensor(state)).to(torch.device('cuda'))
        action = self.Actor.forward(state).detach()
        new_action = action.data.numpy() + (self.noise.sample()*self.action_limit)
        return torch.from_numpy(new_action)

    def exploit_action(self, state):
        state = Variable(torch.tensor(state))
        action = self.Actor_target.forward(state).detach()
        return action

    def optimize(self):
        s1, a1, r1, s2 = self.ram.sample(self.batch_size)

        a2 = self.target_actor.forward(s2).detach()
		next_val = torch.squeeze(self.target_critic.forward(s2).detach())
		y_expected = r1 + self.gamma*next_val
		y_predicted = torch.squeeze(self.critic.forward(s1, a1))
		loss_critic = torch.nn.functional.smooth_l1_loss(y_predicted, y_expected)
		self.critic_optim.zero_grad()
		loss_critic.backward()
		self.critic_optim.step()

        pred_a1 = self.Actor.forward(s1)
		loss_actor = -1*torch.sum(self.Critic.forward(s1))
		self.actor_optim.zero_grad()
		loss_actor.backward()
		self.actor_optim.step()

		utils.soft_update(self.Actor_target, self.Actor, self.tau)
		utils.soft_update(self.Critic_target, self.Critic, self.tau)

    def train(self):
         for i in range(self.episodes):
             self.run_episode(i)
             sys.stdout.write("\rEpisode: {0}/{1}".format(i+1, self.episodes))
             sys.stdout.flush()
             if i%100 == 0:
                 self.save_models(i)

    def average(self, x):
        return sum(x)/len(x)

    def run_episode(self):
        self.state = []
        self.action = []
        self.reward = []
        self.new_state = []
        end = False
        _ = self.env.reset()
        commands = []
        action = self.explore_action(env.pgnr)
        input_tensor = torch.tensor(env.pgnr).to(self.device)
        while not end:
            state, reward, end, new_state = self.env.step_AC_continuous(action)
            self.reward.append(reward)
            self.actor_optim.zero_grad()
            actor_loss = -1*torch.sum(self.Critic.forward(Variable(torch.tensor(state))))
            actor_loss.backward()
            self.actor_optim.step()
            new_input_tensor = torch.tensor(env.pgnr)
            self.state.append(state)
            self.action.append(actoin)
            self.reward.append(reward)
            self.new_state.append(new_state)
            self.ram.store(input_tensor, action, reward, new_input_tensor)
            input_tensor = new_input_tensor
            self.optimize()

        self.score.append(self.average(self.reward))
        self.train_critic()

    def train_critic(self):
        running_add = 0
        for i in reversed(range(len(self.reward))):
            running_add = running_add*self.gamma + self.reward[i]
            self.reward[i] = torch.tensor([running_add]).to(self.device)
        for i in range(len(self.state)-1):
            # Train the critic over multiple episodes to remove potential bias
            self.ram.store(torch.tensor(self.state[i]).to(torch.device('cpu')), self.action[i], self.reward[i], torch.tensor(self.new_state[i]).to(torch.device('cpu')))

        for i in range(500):
            state, action, reward, new_state = self.ram.sample()
            state = state.to(self.device)
            action = action.to(self.device)
            reward = reward.to(self.device)
            output = self.Critic(state)
            self.critic_optim.zero_grad()
            # self.reward[i] = torch.reshape(self.reward[i], output.size())
            loss = self.loss(output, reward)
            self.critic_loss.append(loss.item())
            loss.backward()
            self.critic_optim.step()
        self.reward = []
        self.state = []
        self.action = []


    def save_models(self, episode_count):
		torch.save(self.Actor_target.state_dict(), './Models/' + str(episode_count) + '_actor.pt')
		torch.save(self.Critic_target.state_dict(), './Models/' + str(episode_count) + '_critic.pt')

    def load_models(self, episode):
		self.Actor.load_state_dict(torch.load('./Models/' + str(episode) + '_actor.pt'))
		self.Critic.load_state_dict(torch.load('./Models/' + str(episode) + '_critic.pt'))
		utils.hard_update(self.Actor_target, self.actor)
		utils.hard_update(self.Critic_target, self.critic)


if __name__ == "__main__":
    env = env()
    max_episodes = 5000
    max_buffer = 1000000
    DDPG_agent = ActorCritic(env, max_episodes, max_buffer)
