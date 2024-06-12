import numpy as np
import sys

if '../' not in sys.path:
    sys.path.append('../')
import torch.optim as optim
from DQNNet import DQNNet
import torch
import torch.nn as nn
import random
import torch.nn.functional as F

# if GPU is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

number_timesteps = 30


class DQNAgent(object):

    def __init__(self, state_dim, hidden_dim, action_dim, lr,
                 gamma, epsilon, target_replace, seed=5):
        self.action_size = action_dim
        self.state_size = state_dim
        # build net
        self.policy_net = DQNNet(state_dim, hidden_dim, action_dim).to(device)
        self.target_net = DQNNet(state_dim, hidden_dim, action_dim).to(device)
        # copy param
        self.target_net.load_state_dict(self.policy_net.state_dict())
        # adam optim
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.gamma = gamma
        self.epsilon = epsilon
        self.target_replace = target_replace
        self.replace_count = 0
        # fixed seed
        self._seed(seed)

    def _seed(self, SEED):
        np.random.seed(SEED)
        random.seed(SEED)

    def predict_action(self, state):
        state = torch.tensor(state, dtype=torch.float).to(device)
        return self.policy_net(state).argmax().item()

    def get_action(self, state, i_episode, num_episode):
        explore_frac = 0.1
        epsilon = lambda i: 1 - 0.99 * min(1, i / (num_episode * explore_frac))

        eps = epsilon(i_episode)
        # eps = 0.01
        if np.random.rand() <= eps:
            return np.random.randint(self.action_size)
        else:
            # state = np.array(state[0])
            # print(state, type(state))
            state = torch.tensor(state, dtype=torch.float).to(device)
            with torch.no_grad():
                return self.policy_net(state).argmax().item()

    def update(self, transition_dict):
        states = torch.tensor(transition_dict['states'], dtype=torch.float).view(-1, self.state_size).to(device)
        actions = torch.tensor(transition_dict['actions'], dtype=torch.long).view(-1, 1).to(device)
        rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float).view(-1, 1).to(device)
        next_states = torch.tensor(transition_dict['next_states'], dtype=torch.float).view(-1, self.state_size).to(
            device)
        dones = torch.tensor(transition_dict['dones'], dtype=torch.float).view(-1, 1).to(device)

        #
        q_values = self.policy_net(states).gather(1, actions)
        max_next_q_value = self.target_net(next_states).max(1)[0].view(-1, 1)

        q_targets = rewards + self.gamma * max_next_q_value * (1 - dones)
        # q_targets = rewards + self.gamma * max_next_q_value
        q_loss = torch.mean(F.mse_loss(q_values, q_targets))

        # process grad and back forward
        self.optimizer.zero_grad()
        q_loss.backward()
        self.optimizer.step()

        if self.replace_count % self.target_replace == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        self.replace_count += 1
