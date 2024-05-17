import numpy as np
import pickle
from grid_env import GridWorldEnv

def value_iteration(transition_probs, rewards, gamma, iterations):
    states, actions, _ = transition_probs.shape
    Q_values = np.zeros((states, actions))
    for _ in range(iterations):
        optimal_values = np.max(Q_values, axis=1, keepdims=True)
        Q_values = rewards + gamma * (transition_probs @ optimal_values).squeeze()
    return Q_values

def expert_policy(current_state, Q_table):
    grid_width = env.grid_matrix.shape[1]
    index = current_state[1] * grid_width + current_state[0]
    action_values = Q_table[index]
    best_action_indices = np.where(action_values == np.max(action_values))[0]
    return np.random.choice(best_action_indices)

def generate_dataset_formatted(env, q_table, transitions=10000):
    data = {
        'states': [],
        'actions': [],
        'rewards': [],
        'next_states': [],
        'terminations': []
    }
    empty_cells = env.get_empty_cells()

    env.goal_coords = env.fixed_goal_coords 
    for _ in range(transitions):
        start_index = np.random.choice(len(empty_cells))
        start = empty_cells[start_index]
        env.state = start
        done = False
        while not done:
            action = expert_policy(env.state, q_table)
            next_state, reward, done, _ = env.step(action)
            data['states'].append(np.array(env.state))
            data['actions'].append(action)
            data['rewards'].append(reward)
            data['next_states'].append(np.array(next_state))
            data['terminations'].append(False)
            if not done:
                env.state = next_state
            else:
                env.reset()  

    for key in data:
        data[key] = np.array(data[key])

    return data

if __name__ == '__main__':
    grid_matrix = np.array([
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
    ], dtype=int)
    env = GridWorldEnv(grid_matrix)
    optimal_q = value_iteration(env.P, env.r, 0.90, 10000) 
    data = generate_dataset_formatted(env, optimal_q, transitions=100)

    with open('core/complete_data_expert.pkl', 'wb') as f:
        pickle.dump(data, f)
