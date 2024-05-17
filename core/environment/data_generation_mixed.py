import numpy as np
import pickle
from grid_env import GridWorldEnv

def value_iteration(env, goal, discount_factor=0.90, theta=0.01):
    states = env.get_empty_cells()
    actions = [GridWorldEnv.UP, GridWorldEnv.DOWN, GridWorldEnv.LEFT, GridWorldEnv.RIGHT]
    value_map = np.zeros_like(env.grid_matrix, dtype=np.float32)
    policy_map = np.full((len(env.grid_matrix), len(env.grid_matrix[0])), None)

    while True:
        delta = 0
        for state in states:
            v = value_map[state[1], state[0]]
            max_value = float('-inf')
            for action in actions:
                next_state = env.move(state, action)
                reward = 1 if next_state == goal else 0
                value = reward + discount_factor * value_map[next_state[1], next_state[0]]
                if value > max_value:
                    max_value = value
                    policy_map[state[1], state[0]] = action
            value_map[state[1], state[0]] = max_value
            delta = max(delta, abs(v - max_value))

        if delta < theta:
            break

    return policy_map

def generate_dataset_formatted(env, num_expert=100, num_random=9900):
    total_transitions = num_expert + num_random
    data = {
        'states': [],
        'actions': [],
        'rewards': [],
        'next_states': [],
        'terminations': []
    }

    empty_cells = env.get_empty_cells()
    total_data_collected = 0 

    for idx in range(total_transitions):
        if idx < num_expert:
            env.state = (1, 11)  
            policy_map = value_iteration(env, (11, 1))
        else:
            env.state = empty_cells[np.random.choice(len(empty_cells))]

        while True:
            if idx < num_expert:
                action = policy_map[env.state[1], env.state[0]]
            else:
                action = np.random.choice([GridWorldEnv.UP, GridWorldEnv.DOWN, GridWorldEnv.LEFT, GridWorldEnv.RIGHT])

            next_state, reward, done, _ = env.step(action)
            data['states'].append(env.state)
            data['actions'].append(action)
            data['rewards'].append(reward)
            data['next_states'].append(next_state)
            data['terminations'].append(False)

            total_data_collected += 1

            if done or total_data_collected >= total_transitions:
                break
            env.state = next_state

    for key in data:
        data[key] = np.array(data[key][:total_transitions])

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
    data = generate_dataset_formatted(env, num_expert=10000, num_random=0)

    with open('core/complete_data_expert.pkl', 'wb') as f:
        pickle.dump(data, f)
