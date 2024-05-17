import os
from core.environment.grid_env import GridWorldEnv

class EnvFactory:
    @classmethod
    def create_env_fn(cls, cfg):
        if cfg.env_name == 'grid':
            return lambda: GridWorldEnv(grid_matrix, cfg.seed)
        else:
            print(cfg.env_name)


grid_matrix = [
    [1,1,1,1,1,1,1,1,1,1,1,1,1],
    [1,0,0,0,0,0,1,0,0,0,0,0,1],
    [1,0,0,0,0,0,1,0,0,0,0,0,1],
    [1,0,0,0,0,0,0,0,0,0,0,0,1],
    [1,0,0,0,0,0,1,0,0,0,0,0,1],
    [1,0,0,0,0,0,1,0,0,0,0,0,1],
    [1,1,0,1,1,1,1,0,0,0,0,0,1],
    [1,0,0,0,0,0,1,1,1,0,1,1,1],
    [1,0,0,0,0,0,1,0,0,0,0,0,1],
    [1,0,0,0,0,0,1,0,0,0,0,0,1],
    [1,0,0,0,0,0,0,0,0,0,0,0,1],
    [1,0,0,0,0,0,1,0,0,0,0,0,1],
    [1,1,1,1,1,1,1,1,1,1,1,1,1]
]
