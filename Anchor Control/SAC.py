import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from torch.optim import Adam
import environment
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
epsilon = 0.1

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

def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)

# V-Network (U with a vertical line through it)
class ValueNetwork(nn.Module):
    def __init__(self, num_inputs, hidden_dim):
        super(ValueNetwork, self).__init__()

        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)

        self.apply(weights_init_)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x

# Theta-Network
class QNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim):
        super(QNetwork, self).__init__()

        # Q1 architecture
        self.linear1 = nn.Linear(num_inputs + num_actions, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)

        # Q2 architecture
        self.linear4 = nn.Linear(num_inputs + num_actions, hidden_dim)
        self.linear5 = nn.Linear(hidden_dim, hidden_dim)
        self.linear6 = nn.Linear(hidden_dim, 1)

        self.apply(weights_init_)

    def forward(self, state, action):
        xu = torch.cat([state, action], -1)

        x1 = F.relu(self.linear1(xu))
        x1 = F.relu(self.linear2(x1))
        x1 = self.linear3(x1)

        x2 = F.relu(self.linear4(xu))
        x2 = F.relu(self.linear5(x2))
        x2 = self.linear6(x2)

        return x1, x2

# Phi-Network
class Policy(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim, action_scalar=3):
        super(Policy, self).__init__()

        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)

        self.mean_linear = nn.Linear(hidden_dim, num_actions)
        self.log_std_linear = nn.Linear(hidden_dim, num_actions)

        self.apply(weights_init_)

        # action rescaling
        self.action_scale = torch.tensor(3.)
        self.action_bias = torch.tensor(0.)


    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        return mean, log_std

    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + epsilon)
        log_prob = log_prob.sum(-1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        return super(Policy, self).to(device)

class SAC():
    def __init__(self, n_in, hidden_dim, n_actions):
        device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
        self.device = device
        self.ac = MLPActorCritic(n_in, n_actions).to(device)
        self.ac_targ = MLPActorCritic(n_in, n_actions).to(device)
        # self.QNet = QNetwork(n_in, n_actions, hidden_dim).to(device)
        # self.Q_targ = QNetwork(n_in, n_actions, hidden_dim).to(device)
        # self.PiNet = Policy(n_in, n_actions, hidden_dim).to(device)
        for target_param, param in zip(self.ac_targ.parameters(), self.ac.parameters()):
            target_param.data.copy_(param.data)
            target_param.requires_grad = False
        q_params = itertools.chain(self.ac.q1.parameters(), self.ac.q2.parameters())
        self.q_optim = Adam(q_params, lr = 1e-3)
        self.pi_optim = Adam(self.ac.pi.parameters(), lr = 1e-3)
        # self.Pioptim = Adam(self.PiNet.parameters(), lr = 1e-5)
        self.alpha = 0.2
        self.RAM = ReplayMemory(100000, n_in, n_actions)
        self.batch_size = 100
        self.gamma = 0.9
        self.tau = 0.995

    def calc_q_loss(self, o, a, r, o2):
        q1 = self.ac.q1(o,a)
        q2 = self.ac.q2(o,a)

        # Bellman backup for Q functions
        with torch.no_grad():
            # Target actions come from *current* policy
            a2, logp_a2 = self.ac.pi(o2)

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

        # Entropy-regularized policy loss
        loss_pi = (self.alpha * logp_pi - q_pi).mean()

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
            loss_pi.backward()
            self.pi_optim.step()

            # Unfreeze Q-networks so you can optimize it at next DDPG step.
            for (q1, q2) in zip(self.ac.q1.parameters(), self.ac.q2.parameters()):
                q1.requires_grad = True
                q2.requires_grad = True

            # Finally, update target networks by polyak averaging.
            with torch.no_grad():
                for p, p_targ in zip(self.ac.parameters(), self.ac_targ.parameters()):
                    # NB: We use an in-place operations "mul_", "add_" to update target
                    # params, as opposed to "mul" and "add", which would make new tensors.
                    p_targ.data.mul_(self.tau)
                    p_targ.data.add_((1 - self.tau) * p.data)

    def select_actions(self, state, deterministic=False):
        return self.ac.act(torch.as_tensor(state, dtype=torch.float32),
                      deterministic)

class Runner():
    def __init__(self, number_of_episodes):
        self.env = environment.env()
        self.SAC = SAC(n_in = 36, hidden_dim = 512, n_actions = 30)
        self.num_episodes = number_of_episodes
        self.reward_logger = []
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def run_episode(self):
        state = self.env.reset()
        self.state = []
        self.action = []
        self.reward = []
        self.new_state = []
        end = False
        running_reward = 0
        while not end:
            action = self.SAC.select_actions(torch.tensor(state, dtype=torch.float32).to(self.device))
            state, reward, end, new_state = self.env.step_AC_continuous(action)
            running_reward += sum(reward)/len(reward)
            self.SAC.RAM.store(torch.tensor(state), action, torch.tensor(sum(reward)/len(reward)), torch.tensor(new_state))
            state = new_state
        self.SAC.update_Networks(100)
        return running_reward

    def train(self):
        for i in range(self.num_episodes):
            sys.stdout.write("\rEpisode: {0}/{1}".format(i+1, self.num_episodes))
            sys.stdout.flush()
            episode_reward = self.run_episode()
            self.reward_logger.append(episode_reward)

if __name__ == "__main__":
    os.system('clear')
    agent = Runner(number_of_episodes = 200)
    agent.train()
    plt.figure()
    x = np.linspace(0, agent.num_episodes, len(agent.reward_logger))
    unfiltered = pd.DataFrame(
    	{'unfiltered': agent.reward_logger})
    ema = unfiltered.ewm(com=9).mean()
    plt.plot(x, unfiltered, label="Episode Rewards")
    x = np.linspace(0, agent.num_episodes, len(ema))
    plt.plot(x, ema, label="Episode Rewards - smoothened")
    plt.legend()
    plt.show()
