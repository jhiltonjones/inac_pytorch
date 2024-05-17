import pickle
import numpy as np
from grid_env import GridWorldEnv

def load_data(file_path):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data

def save_data(file_path, data_dict):
    with open(file_path, 'wb') as f:
        pickle.dump(data_dict, f)

def remove_down_actions_in_upper_left_room(data, grid_bounds, down_action):
    x_min, x_max, y_min, y_max = grid_bounds
    filtered_indices = []

    for idx, (state, action) in enumerate(zip(data['states'], data['actions'])):
        if x_min <= state[0] <= x_max and y_min <= state[1] <= y_max and action == down_action:
            continue 
        filtered_indices.append(idx)

    modified_data = {
        key: np.array([data[key][i] for i in filtered_indices])
        for key in data.keys()
    }

    return modified_data

if __name__ == '__main__':
    data = load_data('core/complete_data_mixed.pkl')

    upper_left_bounds = (1, 5, 1, 5)
    down_action = GridWorldEnv.DOWN  

    missing_data = remove_down_actions_in_upper_left_room(data, upper_left_bounds, down_action)

    save_data('core/complete_data_missing2.pkl', missing_data)
