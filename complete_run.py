import argparse
import json
import os
import numpy as np
import matplotlib.pyplot as plt
from run_ac_offline import run_experiment

def ensure_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)

def run_multiple_experiments(learning_rates, num_runs, dataset_name):
    results = {}
    base_dir = 'results'

    for lr in learning_rates:
        lr_dir = os.path.join(base_dir, f'{dataset_name}_lr_{lr}')
        ensure_directory(lr_dir)

        all_rewards = []
        for run in range(num_runs):
            seed = np.random.randint(0, 10000)
            episode_rewards = run_experiment(learning_rate=lr, seed=seed, dataset_name=dataset_name)
            all_rewards.append(episode_rewards)

            rewards_filename = os.path.join(lr_dir, f'episode_rewards_run_{run}.json')
            with open(rewards_filename, 'w') as f:
                json.dump(episode_rewards, f)

        avg_rewards = np.mean(all_rewards, axis=0)
        results[lr] = avg_rewards.tolist() 

        avg_rewards_filename = os.path.join(lr_dir, 'average_rewards.json')
        with open(avg_rewards_filename, 'w') as f:
            json.dump(avg_rewards.tolist(), f)

        plt.figure()
        plt.plot(avg_rewards, label=f'LR: {lr}')
        plt.xlabel('Iteration')
        plt.ylabel('Average Return per Episode')
        plt.title(f'Average Return per Episode Over Iterations (LR: {lr})')
        plt.grid(True)
        plt.legend()
        plt.savefig(os.path.join(lr_dir, 'avg_rewards.png'))
        plt.close()

    combined_results_filename = os.path.join(base_dir, f'combined_results_{dataset_name}.json')
    with open(combined_results_filename, 'w') as f:
        json.dump(results, f)

    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run multiple experiments with a specific dataset')
    parser.add_argument('--dataset', choices=['expert', 'random', 'missing', 'mixed'], default='expert', help='Specify which dataset to use.')

    args = parser.parse_args()

    num_runs = 10
    learning_rates = [0.001]

    results = run_multiple_experiments(learning_rates, num_runs, args.dataset)

    plt.figure(figsize=(10, 6))
    for lr, avg_rewards in results.items():
        plt.plot(avg_rewards, label=f'LR: {lr}')
    plt.xlabel('Iteration')
    plt.ylabel('Average Return per Episode')
    plt.title('Average Return per Episode Over Iterations for Different Learning Rates')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join('results', f'combined_avg_rewards_{args.dataset}.png'))
    plt.show()
