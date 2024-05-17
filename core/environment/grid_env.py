import numpy as np
import gym
from gym import spaces
import random

class GridWorldEnv(gym.Env):
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3

    def __init__(self, grid_matrix, seed=np.random.randint(int(1e5)), max_steps=100):
        super(GridWorldEnv, self).__init__()
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Tuple((spaces.Discrete(len(grid_matrix)), spaces.Discrete(len(grid_matrix[0]))))
        self.grid_matrix = grid_matrix
        self.max_steps = max_steps
        self.num_steps = 0
        self.seed = seed
        np.random.seed(self.seed)
        self.empty_cells = self.get_empty_cells()
        self.fixed_goal_coords = (11, 1)
        self.goal_coords = self.fixed_goal_coords 
        self.state = None
        self.P, self.r = self.initialize_probabilities_and_rewards()
        self.reset()  

    def reset(self):
        self.num_steps = 0
        self.state = (1, 11) if (1, 11) in self.empty_cells and (1, 11) != self.goal_coords else self.random_empty_cell()
        return self.state

    def step(self, action):
        action = int(action)
        self.num_steps += 1
        next_state = self.move(self.state, action)

        done = self.num_steps >= self.max_steps

        reward = 1 if next_state == self.goal_coords else 0
        self.state = next_state
        return next_state, reward, done, {}

    def is_wall(self, x, y):
        return self.grid_matrix[y][x] == 1
    
    def initialize_probabilities_and_rewards(self):
        num_states = len(self.grid_matrix) * len(self.grid_matrix[0])
        P = np.zeros((num_states, 4, num_states))
        r = np.zeros((num_states, 4))
        for y in range(len(self.grid_matrix)):
            for x in range(len(self.grid_matrix[0])):
                state_index = y * len(self.grid_matrix[0]) + x
                for action in [self.UP, self.DOWN, self.LEFT, self.RIGHT]:
                    next_state = self.move((x, y), action)
                    next_state_index = next_state[1] * len(self.grid_matrix[0]) + next_state[0]
                    P[state_index, action, next_state_index] = 1
                    r[state_index, action] = 1 if next_state == self.goal_coords else 0
        return P, r

    def move(self, state, action):
        x, y = state
        new_state = state 
        if action == self.UP and y > 0 and not self.is_wall(x, y - 1):
            new_state = (x, y - 1)
        elif action == self.DOWN and y < len(self.grid_matrix) - 1 and not self.is_wall(x, y + 1):
            new_state = (x, y + 1)
        elif action == self.LEFT and x > 0 and not self.is_wall(x - 1, y):
            new_state = (x - 1, y)
        elif action == self.RIGHT and x < len(self.grid_matrix[0]) - 1 and not self.is_wall(x + 1, y):
            new_state = (x + 1, y)
        return new_state

    def get_empty_cells(self):
        return [(x, y) for y in range(len(self.grid_matrix)) for x in range(len(self.grid_matrix[0])) if not self.is_wall(x, y)]

    def random_empty_cell(self):
        return random.choice(self.empty_cells)
    
