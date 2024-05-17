import numpy as np
import pickle
from grid_env import GridWorldEnv

def random_action(env):
    return np.random.choice([GridWorldEnv.UP, GridWorldEnv.DOWN, GridWorldEnv.LEFT, GridWorldEnv.RIGHT])

def generate_dataset_formatted(env, transitions=10000):
    data = {
        'states': [],
        'actions': [],
        'rewards': [],
        'next_states': [],
        'terminations': []
    }
    empty_cells = env.get_empty_cells()

    while len(data['states']) < transitions:
        start_index = np.random.choice(len(empty_cells))
        env.state = empty_cells[start_index]

        while len(data['states']) < transitions:
            action = random_action(env)
            if action is None:
                print("No valid action found, skipping this state:", env.state)
                break
            next_state, reward, done, _ = env.step(action)

            data['states'].append(np.array(env.state))
            data['actions'].append(action)
            data['rewards'].append(reward)
            data['next_states'].append(np.array(next_state))
            data['terminations'].append(False)

            if done:
                break 
            env.state = next_state

    for key in data:
        data[key] = np.array(data[key])[:transitions]

    return data


if __name__ == '__main__':
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
    env = GridWorldEnv(grid_matrix)
    data = generate_dataset_formatted(env, transitions=10000)

    with open('core/complete_data_random.pkl', 'wb') as f:
        pickle.dump(data, f)
