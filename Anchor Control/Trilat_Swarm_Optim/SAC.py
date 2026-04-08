import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from torch.optim import Adam
import environment_AC
from utils import hard_update, soft_update
import matplotlib.pyplot as plt
import sys
import numpy as np
import os
import pandas as pd
from AC_lib import *
import itertools

LOG_SIG_MIN = -20
LOG_SIG_MAX = 2


class ReplayMemory():
    def __init__(self, memory_size, observation_space, n_actions):
        self.observation_space = observation_space
        self.num_actions = n_actions
        self.memory_size = memory_size
        self.state = [None]*self.memory_size
        self.action = [None]*self.memory_size
        self.reward = [None]*self.memory_size
        self.new_state = [None]*self.memory_size
        self.current_index = 0
        self.size = 0
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

        state = np.array([[self.state[int(i)].cpu().numpy() for i in samples_index]])
        action = np.array([[self.action[int(i)] for i in samples_index]])
        reward = np.array([[self.reward[int(i)].cpu().numpy() for i in samples_index]])
        new_state = np.array([[self.new_state[int(i)].numpy() for i in samples_index]])

        state = torch.reshape(torch.from_numpy(state), (-1, self.observation_space)).to(self.device)
        action = torch.reshape(torch.from_numpy(action), (-1, self.num_actions)).to(self.device)
        reward = torch.reshape(torch.from_numpy(reward), (-1,1)).to(self.device)
        new_state = torch.reshape(torch.from_numpy(new_state), (-1, self.observation_space)).to(self.device)

        return state, action, reward, new_state


class SAC():
    def __init__(self, n_in, hidden_dim, n_layers, n_actions):
        device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
        self.device = device
        self.ac = ActorCritic(n_in, n_actions, hidden_dim, num_layers = n_layers).to(device)
        print(self.ac.pi)
        self.ac_targ = ActorCritic(n_in, n_actions, hidden_dim, num_layers = n_layers).to(device)
        for target_param, param in zip(self.ac_targ.parameters(), self.ac.parameters()):
            target_param.data.copy_(param.data)
            target_param.requires_grad = False
        self.actor_learning_rate = 1e-6
        q_params = itertools.chain(self.ac.q1.parameters(), self.ac.q2.parameters())
        self.q_optim = Adam(q_params, lr = 1e-4)
        self.pi_optim = Adam(self.ac.pi.parameters(), lr = self.actor_learning_rate)
        self.alpha = 0.2
        self.RAM = ReplayMemory(1000000, n_in, n_actions)
        self.batch_size = 1000
        self.gamma = 0.9
        self.tau = 0.995

    def calc_q_loss(self, o, a, r, o2):
        q1 = self.ac.q1(o,a)
        q2 = self.ac.q2(o,a)

        # Bellman backup for Q functions
        with torch.no_grad():
            # Target actions come from *current* policy
            a2, logp_a2= self.ac.pi(o2, deterministic = True)

            # Target Q-values
            q1_pi_targ = self.ac_targ.q1(o2, a2)
            q2_pi_targ = self.ac_targ.q2(o2, a2)
            q_pi_targ = torch.min(q1_pi_targ, q2_pi_targ)
            backup = r + self.gamma * (q_pi_targ - self.alpha * logp_a2)

        # MSE loss against Bellman backup
        loss_q1 = ((q1 - backup)**2).mean()
        loss_q2 = ((q2 - backup)**2).mean()
        loss_q = loss_q1 + loss_q2

        return loss_q

    def compute_loss_pi(self, o):
        pi, logp_pi = self.ac.pi(o)
        q1_pi = self.ac.q1(o, pi)
        q2_pi = self.ac.q2(o, pi)
        q_pi = torch.min(q1_pi, q2_pi)

        loss_pi = ((self.alpha * logp_pi) - q_pi).mean()

        return loss_pi


    def update_Networks(self, steps):
        for i in range(steps):
            state, action, reward, nextstate = self.RAM.sample(self.batch_size)

            self.q_optim.zero_grad()
            loss_q = self.calc_q_loss(state, action, reward, nextstate)
            loss_q.backward()
            self.q_optim.step()

            for (q1, q2) in zip(self.ac.q1.parameters(), self.ac.q2.parameters()):
                q1.requires_grad = False
                q2.requires_grad = False

            # Next run one gradient descent step for pi.
            self.pi_optim.zero_grad()
            loss_pi = self.compute_loss_pi(state)
            if loss_pi.item() < 0:
                loss_pi.backward()
                self.pi_optim.step()
            else:
                self.pi_optim.zero_grad()

            # Unfreeze Q-networks so you can optimize it at next DDPG step.
            for (q1, q2) in zip(self.ac.q1.parameters(), self.ac.q2.parameters()):
                q1.requires_grad = True
                q2.requires_grad = True

            with torch.no_grad():
                for p, p_targ in zip(self.ac.parameters(), self.ac_targ.parameters()):
                    p_targ.data.mul_(self.tau)
                    p_targ.data.add_((1 - self.tau) * p.data)

    def select_actions(self, state, deterministic=False):
        return self.ac.choose_actions(torch.as_tensor(state, dtype=torch.float32), deterministic=deterministic)

class Runner():
    def __init__(self, number_of_episodes, number_of_anchors = 3, device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
        self.env = environment_AC.env(number_of_anchors)
        self.SAC = SAC(n_in = number_of_anchors*3+6, hidden_dim = 1024, n_layers = 3, n_actions = number_of_anchors*3)
        self.num_episodes = number_of_episodes
        self.reward_logger = []
        self.device = device

    def run_episode(self, episode_num):
        state = self.env.reset()
        self.state = []
        self.action = []
        self.reward = []
        self.new_state = []
        end = False
        running_reward = 0
        steps = 0
        if episode_num > self.num_episodes*1/4:
            deterministic = True
        else:
            deterministic = False

        while not end:
            action = self.SAC.select_actions(torch.tensor(state, dtype=torch.float32).to(self.device), deterministic=deterministic)
            state, reward, end, new_state = self.env.step_AC_continuous(action)
            running_reward += reward
            self.SAC.RAM.store(torch.tensor(state), action, torch.tensor(reward), torch.tensor(new_state))
            state = new_state
            steps = steps+1
        self.SAC.update_Networks(100)
        return running_reward/steps

    def train(self):
        for i in range(self.num_episodes):
            sys.stdout.write("\rEpisode: {0}/{1}".format(i+1, self.num_episodes))
            sys.stdout.flush()
            episode_reward = self.run_episode(i)
            self.reward_logger.append(episode_reward)

if __name__ == "__main__":
    os.system('clear')
    n_episodes = 5000
    n_anchors = 5
    print("Running training for ", n_episodes, " number of episodes.")
    print("Training SAC network to account for ", n_anchors, " anchors.")
    agent = Runner(number_of_episodes = n_episodes, number_of_anchors = n_anchors)
    agent.train()
    plt.figure()
    x = np.linspace(0, agent.num_episodes, len(agent.reward_logger))
    unfiltered = pd.DataFrame(
        {'unfiltered': agent.reward_logger})
    ema = unfiltered.ewm(com=9).mean()
    plt.plot(x, unfiltered, label="Rewards")
    x = np.linspace(0, agent.num_episodes, len(ema))
    plt.plot(x, ema, label="Rewards - smoothened")
    plt.xlabel('Episode [-]')
    plt.ylabel('Reward per State [-]')
    plt.title(str(n_anchors)+' Anchor Drones: Reward vs Episode')
    plt.legend()
    # Hopefully this will clear out the memory usage on the GPU before
    # starting the next loop breaking the Residual NN training program
    agent.SAC.ac = None
    agent.SAC.ac_targ = None
    plt.show()
