import numpy as np
import sys

if '../' not in sys.path:
    sys.path.append('../')

from PolicyNet import PolicyNet
import torch.optim as optim

import torch
import torch.nn as nn
import torch.nn.functional as F

# if GPU is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class REINFORCEAgent(object):

    def __init__(self, state_dim, hidden_dim, action_dim, lr,
                 gamma, seed=5):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.policy_net = PolicyNet(state_dim, hidden_dim, action_dim).to(device)
        self.gamma = gamma
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)

        # seed
        self._seed(seed)

    def _seed(self, seed):
        np.random.seed(seed)
        torch.manual_seed(seed)

    def get_action(self, state):
        state = torch.tensor(np.array([state]), dtype=torch.float).to(device)
        probs = self.policy_net(state)
        action_dist = torch.distributions.Categorical(probs)
        # print(probs, action_dist)
        return action_dist.sample().item()

    def predict_action(self, state):
        state = torch.tensor(np.array([state]), dtype=torch.float).to(device)
        probs = self.policy_net(state)
        return probs.argmax().item()

    def update(self, transition_dict):
        reward_list = transition_dict['rewards']
        state_list = transition_dict['states']
        action_list = transition_dict['actions']

        G = 0

        self.optimizer.zero_grad()
        for i in reversed(range(len(reward_list))):
            reward = reward_list[i]
            state = torch.tensor(np.array([state_list[i]]), dtype=torch.float).view(-1, self.state_dim).to(device)
            action = torch.tensor(np.array([action_list[i]]), dtype=torch.long).view(-1, 1).to(device)

            log_prob = torch.log(self.policy_net(state).gather(1, action))
            G = self.gamma * G + reward
            # loss 只能最小化，不能最大化
            # 所以我们 最小化 - log * G 相当于最大化 log * G
            loss = - log_prob * G
            loss.backward()
        self.optimizer.step()
