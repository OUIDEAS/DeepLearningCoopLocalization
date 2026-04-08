import torch
import torch.nn as nn
import torch.optim as optim
import environment
from torch.distributions import Categorical
import matplotlib.pyplot as plt

class EncoderDecoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMActor, self).__init__()
        self.encoder = nn.LSTM(input_size, hidden_size, num_layers)
        self.decoder = nn.LSTM(hidden_size, hidden_size, num_layers)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        hidden = None
        cell = None
        seq = x
        for i in seq:
            i = torch.reshape(i, (1,1,-1))
            if hidden is not None and cell is not None:
                x, (hidden, cell) = self.encoder(i, (hidden, cell))
            else:
                x, (hidden, cell) = self.encoder(i)
        outputs = []
        for i in range(len(seq)):
            x, (hidden, cell) = self.decoder(x, (hidden, cell))
            outputs.append(Categorical(torch.nn.functional.softmax(self.fc(x), dim=0)))
        return outputs

class Critic(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(Critic, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers)
        self.fc = nn.Linear(hidden_size, output_size)
    def forward(self, x):
        hidden, cell = torch.zeros(num_layers, batch_size, hidden_size), torch.zeros(num_layers, batch_size, hidden_size)
        seq = x
        for i in seq:
            x, (hidden, cell) = self.lstm(i, (hidden, cell))
        x = self.fc(x)
        return x

# Initialize the actor and critic networks
actor = EncoderDecoder(input_size = 4, hidden_size = 256, num_layers = 3, output_size = 7)
critic = Critic(input_size=5, hidden_size=256, num_layers=2, 1)

# Define optimizers for actor and critic
actor_optimizer = optim.Adam(actor.parameters(), lr=1e-5)
critic_optimizer = optim.Adam(critic.parameters(), lr=1e-3)

critic_criterion = nn.MSELoss()

env = environment.env()
critic_loss_log = []
# Training loop
accumulated_reward_log = []
for i in range(num_episodes):

    # Pass input through actor to get action sequence
    # Input is a list of each anchors position and its range to the agent
    env.reset()
    end = False
    accumulated_reward = 0
    while not end:
        input = env.states
        action_sequence = actor(input)
        critic_inputs = []
        actions = []
        for action, state in zip(action_sequence, input):
            a = action.sample().cpu()
            actions.append(a.data.numpy().astype(int))
            critic_inputs.append(torch.tensor([state[0].item(), state[1].item(),
                    state[2].item(), state[3].item(), a.data.numpy().astype(int)]))
        v_current = critic(critic_inputs)
        state, reward, end, new_state = env.stepsequence(actions)
        accumulated_reward+=reward
        next_action = actor(new_state)
        for action, state in zip(next_action, new_state):
            a = action.sample().cpu()
            critic_inputs.append(torch.tensor([state[0].item(), state[1].item(),
                    state[2].item(), state[3].item(), a.data.numpy().astype(int)]))
        # Pass input through critic to get value function
        # Input is a list of each anchors position, its range to the agent, and its action
        v_future = critic(critic_inputs)
        # Calculate the expected return for each state-action pair
        expected_return = reward + gamma * v_future

        # Update the critic network
        critic_optimizer.zero_grad()
        critic_loss = critic_criterion(v_current, expected_return)
        critic_loss_log.append(critic_loss.item())
        critic_loss.backward()
        critic_optimizer.step()
        # Update the actor network
        actor_optimizer.zero_grad()

        for action, probs in zip(actions, action_sequence):
            actor_loss = -probs.log_prob(action)*expected_return
            actor_loss.backward(retain_graph=True)

        actor_optimizer.step()
    accumulated_reward_log.append(accumulated_reward)

plt.figure()
plt.plot(critic_loss_log)
plt.title("Critic Loss")

plt.figure()
plt.plot(accumulated_reward_log)
plt.title('Reward')
plt.show()
