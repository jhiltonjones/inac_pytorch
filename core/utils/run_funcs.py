import pickle
import time
import numpy as np
import os
import matplotlib.pyplot as plt

def load_data(dataset_path):
    with open(dataset_path, 'rb') as file:
        data_dict = pickle.load(file) 

    if 'pkl' in data_dict:
        data_dict = data_dict['pkl']
    else:
        print("No 'pkl' key in data dictionary")

    for key, value in data_dict.items():
        if hasattr(value, 'shape'):
            print(f"Loaded data - {key} shape: {value.shape}")
        else:
            print(f"Loaded data - {key} is not an array")

    return data_dict

import matplotlib.pyplot as plt

def evaluate_and_visualize(agent, env):
    grid_shape = env.grid_matrix.shape
    best_actions = np.empty(grid_shape, dtype=np.object)
    state_values = np.empty(grid_shape, dtype=np.float32)

    for y in range(grid_shape[0]):
        for x in range(grid_shape[1]):
            if not env.is_wall(x, y):
                state = (x, y)
                best_actions[y, x] = agent.best_action(state)
                state_values[y, x] = agent.value_estimation(state)
            else:
                best_actions[y, x] = 'Wall'
                state_values[y, x] = np.nan  

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(env.grid_matrix, cmap='Greys', interpolation='none')

    for y in range(grid_shape[0]):
        for x in range(grid_shape[1]):
            if best_actions[y, x] != 'Wall':
                action = best_actions[y, x]
                char = {0: '↑', 1: '↓', 2: '←', 3: '→'}.get(action, '')
                ax.text(x, y, char, ha='center', va='center', color='blue')

    ax.grid(True)
    plt.title('Best Actions in Each State')
    plt.show()

    plt.figure(figsize=(10, 10))
    plt.imshow(state_values, cmap='viridis', interpolation='none')
    plt.colorbar(label='State Value')
    plt.title('State Values in Each State')
    plt.grid(True)
    plt.show()


def run_steps(agent, max_steps, log_interval, eval_pth):
    t0 = time.time()
    evaluations = []
    agent.populate_returns(initialize=True)
    while True:
        if log_interval and not agent.total_steps % log_interval:
            mean, median, min_, max_ = agent.log_file(elapsed_time=log_interval / (time.time() - t0), test=True)
            evaluations.append(mean)
            t0 = time.time()
        if max_steps and agent.total_steps >= max_steps:
            break
        agent.step()
    agent.save()
    np.save(eval_pth+"/evaluations.npy", np.array(evaluations))