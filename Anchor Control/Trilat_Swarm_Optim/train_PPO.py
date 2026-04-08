import os
os.system('clear')
from PPO import PPO
import torch
import environment_AC
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from AC_lib import *
from tqdm import tqdm
from datetime import datetime

class Runner():
    def __init__(self, number_of_episodes, number_of_anchors = 3, device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
        self.env = environment_AC.env(number_of_anchors)
        self.PPO = PPO(state_dim = number_of_anchors*3+6,
                       action_dim = number_of_anchors*3,
                       lr_actor = 1e-6,
                       lr_critic = 1e-4,
                       gamma = 0.9,
                       K_epochs = 10,
                       eps_clip = 0.3,
                       has_continuous_action_space = True)
        self.num_episodes = number_of_episodes
        self.reward_logger = []
        self.device = device
        self.timestep = 0
        self.update_timestep = 5000
        self.action_std_decay_freq = 10000

    def run_episode(self, episode_num):
        state = self.env.reset()
        self.state = []
        self.action = []
        self.reward = []
        self.new_state = []
        end = False
        running_reward = 0
        step = 0
        while not end:
            action = self.PPO.select_action(state)
            state, reward, end, new_state = self.env.step_AC_continuous(action)
            self.PPO.buffer.rewards.append(reward)
            self.PPO.buffer.is_terminals.append(end)
            running_reward += reward
            self.timestep +=1
            step+=1
            if self.timestep % self.update_timestep == 0:
                self.PPO.update()

            if self.timestep % self.action_std_decay_freq == 0:
                self.PPO.decay_action_std(0.01, 0.1)
            state = new_state
        return running_reward/step

    def train(self):
        start_time = datetime.now().replace(microsecond=0)
        print("================================================================================")
        print("Started training at (EST) : ", start_time)
        print("================================================================================")
        for i in tqdm(range(self.num_episodes)):
            episode_reward = self.run_episode(i)
            self.reward_logger.append(episode_reward)

if __name__ == "__main__":
    n_episodes = 300
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
    plt.show()
