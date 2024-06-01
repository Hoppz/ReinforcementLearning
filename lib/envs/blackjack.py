import numpy as np
from gym import spaces

# 每一张牌都有无限多
# 1 = Ace, 2-10 Number cards, Jack/Queen/King = 10
deck = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10]


# a > b -> 1
# a < b -> -1
# a = b -> 0
def cmp(a, b):
    return int((a > b)) - int((a < b))


# 等概率的选一张牌
def draw_card():
    return int(np.random.choice(deck))


# 开局选两张牌做手牌
def draw_hand():
    return [draw_card(), draw_card()]


# 1. 手里是否有 A
# 2. A 是否可以作为 11
# 同时满足返回 True
def usable_ace(hand):
    return 1 in hand and sum(hand) + 10 <= 21


# 求手牌的和
def sum_hand(hand):
    if usable_ace(hand):
        return sum(hand) + 10
    return sum(hand)


# 是否爆牌
def is_bust(hand):
    return sum_hand(hand) > 21


def score(hand):
    return 0 if is_bust(hand) else sum_hand(hand)


def is_natural(hand):
    return sorted(hand) == [1, 10]


def _seed(seed):
    np.random.seed(seed)


class BlackjackEnv(object):
    """
        args:
            action_space: 可以做的操作，1. 要牌， 2.停牌
            observation_space:
                            1. 自己手牌点数和
                            2. 对手显示出来的一张牌的和
                            3. 自己是否有 11 点 A
    """

    def __init__(self, seed=4, natural=False):
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Tuple((
            spaces.Discrete(32),
            spaces.Discrete(11),
            spaces.Discrete(2)))
        _seed(seed)

        self.natural = natural
        self._reset()
        self.nA = 2

    def reset(self):
        return self._reset()

    def step(self, action):
        return self._step(action)

    def _reset(self):
        self.dealer = draw_hand()
        self.player = draw_hand()

        # 如果手牌点数小于 12 自动抽牌（满足书上的条件）
        while sum_hand(self.player) < 12:
            self.player.append(draw_card())

        return self._get_obs()

    def _step(self, action):
        assert self.action_space.contains(action)
        # 自己抽牌
        if action:
            self.player.append(draw_card())
            if is_bust(self.player):
                done = True
                reward = -1
            else:
                done = False
                reward = 0
        # 对手抽牌
        else:
            done = True
            while sum_hand(self.dealer) < 17:
                self.dealer.append(draw_card())
            reward = cmp(score(self.player), score(self.dealer))
            if self.natural and is_natural(self.dealer) and reward == 1:
                reward = 1.5
        return self._get_obs(), reward, done, {}

    # 自己的手牌和，对手的第一张牌，是否有可用的 A
    def _get_obs(self):
        return (sum_hand(self.player), self.dealer[0], usable_ace(self.player))
