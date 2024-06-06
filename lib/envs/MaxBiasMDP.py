import numpy as np


class MDP(object):
    """
        例 6.7 Maximization Bias Example
    """

    def __init__(self, seed):
        self.nS = 4
        self.nA = 2
        # loc:
        # 0 -> N(-0.1,1) , 1 -> B , 2 -> A , 3 ->0
        self.loc = 0
        self._seed(seed)

    def reset(self):
        return self._reset()

    def step(self, action):
        return self._step(action)

    def _reset(self):
        self.loc = 2
        return self.loc

    def _seed(self, seed=None):
        np.random.seed(seed)

    # action : 0 -> left , 1 -> right
    def _step(self, action):
        self.loc = self.loc + 1 if action == 1 else self.loc - 1
        reward = np.random.normal(-0.1, 1) if self.loc == 0 else 0
        done = True if self.loc == 0 or self.loc == 3 else False
        return self.loc, reward, done
