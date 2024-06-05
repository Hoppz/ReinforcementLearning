from typing import List

import numpy as np

LEFT = 0
DOWN = 1
RIGHT = 2
UP = 3
directions = [(0, -1), (1, 0), (0, 1), (-1, 0)]


# DFS to check that it's a valid path.
def is_valid(grid: np.ndarray) -> bool:
    max_size = grid.shape[0]
    frontier, discovered = [], set()
    frontier.append((0, 0))
    while frontier:
        r, c = frontier.pop()
        if not (r, c) in discovered:
            discovered.add((r, c))

            for x, y in directions:
                r_new = r + x
                c_new = c + y
                if r_new < 0 or r_new >= max_size or c_new < 0 or c_new >= max_size:
                    continue
                if grid[r_new][c_new] > 0:
                    return True
                if grid[r_new][c_new] == -1:
                    frontier.append((r_new, c_new))
    return False


# Generates a random valid map (one that has a path from start to goal)
def generate_random_map(
        size: int = 5, p: float = 0.5, punish: int = -1, seed: int = None
) -> np.array:
    valid = False
    grid = np.full((size, size), -1)
    p = [1 - p, p]

    if seed is not None:
        np.random.seed(seed)

    while not valid:

        # generrate
        for i in range(size):
            for j in range(size):
                grid[i, j] = np.random.choice([-1, punish], p=p)

        grid[0][0] = -1
        grid[0][-1] = 1
        valid = is_valid(grid)
    return grid


class Env(object):

    def __init__(self, shape: List, punish: int = -10, seed: int = None, p: float = 0.5):
        super(Env, self).__init__()
        self.s = None
        self.nrow, self.ncol = nrow, ncol = shape
        self.punish = punish
        self.seed = seed
        self.observation_size = self.nrow * self.ncol
        self.action_size = 4
        # generate gird
        self.grid = generate_random_map(size=self.nrow, punish=punish, p=p, seed=seed)

        # 嵌套字典，存储所有的 s,a 的 experience
        self.P = {s: {a: [] for a in range(self.action_size)} for s in range(self.observation_size)}

        def two2one(row, col):
            return row * self.ncol + col

        def inc(row, col, a):
            newrow = row + directions[a][0]
            newcol = col + directions[a][1]
            if check_in_world(newrow, newcol):
                return newrow, newcol, self.grid[newrow][newcol]
            else:
                return row, col, -1

        def check_in_world(row, col):
            if row < 0 or row >= self.nrow or col < 0 or col >= self.ncol:
                return False
            else:
                return True

        def update_probability_matrix(row, col, action):
            newrow, newcol, reward = inc(row, col, action)
            newstate = two2one(newrow, newcol)
            terminated = reward > 0
            return newstate, reward, terminated

        for row in range(nrow):
            for col in range(ncol):
                s = two2one(row, col)
                for a in range(4):
                    li = self.P[s][a]
                    # next_state, reward, terminated
                    li.append((update_probability_matrix(row, col, a)))

    def reset(self):
        self.s = 0
        return int(self.s)

    def step(self, action):
        next_state, reward, terminated = self.P[self.s][action][0]
        self.s = next_state
        return next_state, reward, terminated
