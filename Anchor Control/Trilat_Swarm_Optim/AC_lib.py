import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.distributions.normal import Normal
import itertools
import torch.optim as optim

def weights_init_(m):
    if isinstance(m, torch.nn.Linear):
        nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('leaky_relu',1e-2))

LOG_STD_MAX = 2
LOG_STD_MIN = -20
epsilon = 1e-6

class GaussianFeedForwardActor(nn.Module):
    def __init__(self, obs_space, action_space, hidden_size, num_layers, activation, action_limit):
        super().__init__()
        layers = [nn.Linear(obs_space, hidden_size), activation()]
        for j in range(num_layers-1):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(activation())
        self.net = nn.Sequential(*layers)
        self.mu = nn.Linear(hidden_size, action_space)
        self.log = nn.Linear(hidden_size, action_space)
        self.action_limit = action_limit
        self.apply(weights_init_)

    def forward(self, obs, deterministic=False, with_logprob=True):
        output = self.net(obs.to(torch.float32)/torch.max(torch.abs(obs.to(torch.float32))))
        m = self.mu(output)
        l = self.log(output)
        l = torch.clamp(l, LOG_STD_MIN, LOG_STD_MAX)
        std = torch.exp(l)

        pi_distribution = torch.distributions.normal.Normal(m, std)

        if deterministic:
            pi_action = m
        else:
            pi_action = pi_distribution.rsample()

        if with_logprob:
            logp_pi = pi_distribution.log_prob(pi_action).sum(axis=-1)
            logp_pi -= (2*(np.log(2) - pi_action - F.softplus(-2*pi_action))).sum(axis=-1)
        else:
            logp_pi = None

        pi_action = torch.tanh(pi_action)
        pi_action = self.action_limit * pi_action

        return pi_action, logp_pi

class QNet(nn.Module):
    def __init__(self, obs_space: int, act_dim: int, hidden_size: int, num_layers: int, activation):
        super().__init__()
        layers = [nn.Linear(obs_space+act_dim, hidden_size)]
        for j in range(num_layers-1):
            layers.append(activation())
            layers.append(nn.Linear(hidden_size, hidden_size))
        layers.append(nn.Linear(hidden_size, 1))
        self.q = nn.Sequential(*layers)
        self.apply(weights_init_)

    def forward(self, obs, act):
        input = torch.cat([obs.to(torch.float32), act.to(torch.float32)], dim=-1)
        q = self.q(input)
        return torch.squeeze(q, -1) # Critical to ensure q has right shape.

class ActorCritic(nn.Module):
    def __init__(self, observation_space, action_space, hidden_size=512, num_layers=2,
                 activation=nn.LeakyReLU):
        super().__init__()
        act_limit = 15
        # build policy and value functions
        self.pi = GaussianFeedForwardActor(observation_space, action_space,
                                hidden_size, num_layers, activation, act_limit)
        self.q1 = QNet(observation_space, action_space, hidden_size, num_layers,
                                                                    activation)
        self.q2 = QNet(observation_space, action_space, hidden_size, num_layers,
                                                                    activation)

    def choose_actions(self, obs, deterministic=False):
        with torch.no_grad():
            a, log_pi = self.pi(obs, deterministic=deterministic)
            return a.cpu().numpy()

class SoftActorCritic():
    def __init__(self, n_in, hidden_dim, n_actions):
        device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
        self.device = device
        self.ac = ActorCritic(n_in, n_actions, hidden_dim).to(device)
        self.ac_targ = ActorCritic(n_in, n_actions, hidden_dim).to(device)
        for target_param, param in zip(self.ac_targ.parameters(), self.ac.parameters()):
            target_param.data.copy_(param.data)
            target_param.requires_grad = False
        self.actor_learning_rate = 1e-7
        q_params = itertools.chain(self.ac.q1.parameters(), self.ac.q2.parameters())
        self.q_optim = optim.Adam(q_params, lr = 1e-3)
        self.pi_optim = optim.Adam(self.ac.pi.parameters(), lr = self.actor_learning_rate)
        self.alpha = 0.2
        self.RAM = ReplayMem(100000, n_in, n_actions)
        self.batch_size = 1000
        self.gamma = 0.9
        self.tau = 0.995

    def set_alpha(self, alpha):
        self.alpha = alpha

    def set_batch_size(self, batch_size):
        self.batch_size = batch_size

    def set_gamma(self, gamma):
        self.gamma = gamma

    def set_tau(self, tau):
        self.tau = tau

    def calc_q_loss(self, o, a, r, o2):
        q1 = self.ac.q1(o,a)
        q2 = self.ac.q2(o,a)
        with torch.no_grad():
            a2, logp2 = self.ac.pi(o2)
            q1pi_targ = self.ac_targ.q1(o2, a2)
            q2pi_targ = self.ac_targ.q2(o2, a2)
            qpi_targ = torch.min(q1pi_targ, q2pi_targ)
            backup = r + self.gamma * (qpi_targ - self.alpha * logp2)

        loss_q1 = ((q1 - backup)**2).mean()
        loss_q2 = ((q2 - backup)**2).mean()
        loss_q = loss_q1 + loss_q2
        return loss_q

    def compute_loss_pi(self, o):
        pi, logp = self.ac.pi(o)
        q1pi = self.ac.q1(o, pi)
        q2pi = self.ac.q2(o, pi)
        qpi = torch.min(q1pi, q2pi)
        loss_pi = (self.alpha * logp - qpi).mean()
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

            self.pi_optim.zero_grad()
            loss_pi = self.compute_loss_pi(state)
            loss_pi.backward()
            self.pi_optim.step()
            for (q1, q2) in zip(self.ac.q1.parameters(), self.ac.q2.parameters()):
                q1.requires_grad = True
                q2.requires_grad = True

            with torch.no_grad():
                for p, p_targ in zip(self.ac.parameters(), self.ac_targ.parameters()):
                    p_targ.data.mul_(self.tau)
                    p_targ.data.add_((1 - self.tau) * p.data)

    def select_actions(self, state, deterministic=False):
        return self.ac.choose_actions(torch.as_tensor(state, dtype=torch.float32),
                      deterministic)

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
        log_std = torch.clamp(log_std, min=LOG_STD_MIN, max=LOG_STD_MAX)
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



class ReplayMem():
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
