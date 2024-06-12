import sys

import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

if '../' not in sys.path:
    sys.path.append('../')

from PolicyNet import PolicyNet
from ValueNet import ValueNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ACAgent(object):
    def __init__(self, state_dim, hidden_dim, action_dim, actor_lr,
                 critic_lr, gamma, seed):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.actor = PolicyNet(state_dim, hidden_dim, action_dim).to(device)
        self.critic = ValueNet(state_dim, hidden_dim).to(device)

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)

        self.gamma = gamma
        # seed
        self._seed(seed)

    def _seed(self, seed):
        np.random.seed(seed)
        torch.manual_seed(seed)

    def get_action(self, state):
        state = torch.tensor(np.array([state]), dtype=torch.float).to(device)
        probs = self.actor(state)
        action_dist = torch.distributions.Categorical(probs)
        # print(probs, action_dist)
        return action_dist.sample().item()

    def predict_action(self, state):
        state = torch.tensor(np.array([state]), dtype=torch.float).to(device)
        probs = self.actor(state)
        return probs.argmax().item()

    def update(self, state, action, next_state, reward, done):
        state = torch.tensor(np.array([state]), dtype=torch.float).view(-1, self.state_dim).to(device)
        action = torch.tensor(np.array([action]), dtype=torch.long).view(-1, 1).to(device)
        next_state = torch.tensor(np.array([next_state]), dtype=torch.float).view(-1, self.state_dim).to(device)
        reward = torch.tensor(np.array([reward]), dtype=torch.float).view(-1, 1).to(device)
        done = torch.tensor(np.array([done]), dtype=torch.float).view(-1, 1).to(device)

        # cal critic
        td_target = reward + self.gamma * self.critic(next_state) * (1 - done)
        td_error = td_target - self.critic(state)
        # tensor.detach() 生成一个新的 tensor
        # 和原来唯一的区别是不会计算梯度
        critic_loss = torch.mean(F.mse_loss(self.critic(state), td_target.detach()))

        # cal actor
        log_prob = torch.log(self.actor(state).gather(1, action))
        actor_loss = torch.mean(- log_prob * td_error.detach())

        # grad
        self.actor_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()
        actor_loss.backward()
        critic_loss.backward()
        self.actor_optimizer.step()
        self.critic_optimizer.step()
