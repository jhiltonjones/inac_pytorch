import numpy as np
import pickle
import os
from collections import Counter

data_path = '/home/sam/jack_and_sam/reproducibility_challenge/core/complete_data_expert.pkl'

def analyze_data(data_path):
    with open(data_path, 'rb') as f:
        data = pickle.load(f) 

        if isinstance(data, dict) and all(isinstance(data[key], np.ndarray) for key in ['states', 'actions', 'rewards', 'next_states', 'terminations']):
            print("Data is well-structured.")
        else:
            print("Data structure issues detected.")


    if 'pkl' in data:
        data = data['pkl']['pkl']
    
    states = data['states']
    actions = data['actions']
    rewards = data['rewards']
    next_states = data['next_states']
    terminations = data['terminations']
    
    num_experiences = len(actions)
    unique_states = len(np.unique(states, axis=0))
    average_reward = np.mean(rewards)
    
    action_counts = Counter(actions)
    for action, count in action_counts.items():
        print(f"Action {action}: {count} occurrences ({count / num_experiences * 100:.2f}%)")

    print(f"\nTotal number of experiences: {num_experiences}")
    print(f"Unique states encountered: {unique_states}")
    print(f"Average reward per experience: {average_reward:.4f}")
    
    terminations_count = Counter(terminations)
    for term_state, count in terminations_count.items():
        print(f"Termination state {term_state}: {count} occurrences ({count / num_experiences * 100:.2f}%)")
    

if __name__ == "__main__":

    analyze_data(data_path)
