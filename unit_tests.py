import unittest
import torch
import pickle
import gym
import os
import numpy as np
import copy
from core.agent.in_sample import InSampleAC
from core.environment.data_generation_expert import GridWorldEnv
from core.utils.run_funcs import run_steps, load_testset
from core.utils import logger

def load_expert_data(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)
    
class Config:
    def __init__(self):
        self.seed = 0
        self.env_name = 'grid_matrix'
        self.dataset = 'expert'
        self.discrete_control = 1
        self.state_dim = 2
        self.action_dim = 4
        self.tau = 0.5
        self.max_steps = 100000
        self.log_interval = 1000
        self.learning_rate = 0.01
        self.hidden_units = 256
        self.batch_size = 256
        self.timeout = 100
        self.gamma = 0.99
        self.use_target_network = 1
        self.target_network_update_freq = 100
        self.polyak = 0.9
        self.evaluation_criteria = 'return'
        self.device = 'cpu'
        self.info = '0'

class TestInSampleAC(unittest.TestCase):
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

    def setUp(self):
        cfg = Config()
        self.env_fn = lambda: GridWorldEnv(TestInSampleAC.grid_matrix)
        self.device = 'cpu'
        self.state_dim = 2
        self.action_dim = 4
        self.learning_rate = 0.01
        self.hidden_units = 256
        self.batch_size = 256
        self.timeout = 100
        self.gamma = 0.99
        self.tau = 0.5
        self.seed = 0
        self.offline_data_file = 'core/expert_data.pkl'
        self.offline_data = load_expert_data(self.offline_data_file)  

        project_root = os.path.abspath(os.path.dirname(__file__))      
        exp_path = "test_output"
        self.exp_path = os.path.join(project_root, exp_path)
        os.makedirs(self.exp_path, exist_ok=True)
        self.logger = logger.Logger(cfg, self.exp_path)        
        self.discrete_control = True

        self.agent = InSampleAC(device=self.device, discrete_control=self.discrete_control, state_dim=self.state_dim,
                                action_dim=self.action_dim, hidden_units=self.hidden_units, learning_rate=self.learning_rate,
                                tau=self.tau, polyak=0.9, exp_path=self.exp_path, seed=self.seed, env_fn=self.env_fn,
                                timeout=self.timeout, gamma=self.gamma, offline_data=self.offline_data, batch_size=self.batch_size,
                                use_target_network=True, target_network_update_freq=1, evaluation_criteria='return', logger=self.logger)

    def test_initialization(self):
        self.assertIsNotNone(self.agent, "Agent should be successfully initialized")

    def test_environment_interaction(self):
        state = self.env_fn().reset()
        action = self.agent.eval_step(state)
        self.assertIn(action, [0, 1, 2, 3], "Action should be valid")

    def test_decision_making(self):
        state = torch.rand(self.state_dim)
        action = self.agent.policy(state)
        self.assertIsInstance(action, np.ndarray, "Action should be a NumPy array")


    def test_training_step(self):
        data = self.agent.get_data()
        pre_update_params = copy.deepcopy(self.agent.ac.pi.state_dict())
        self.agent.update(data)
        post_update_params = self.agent.ac.pi.state_dict()
        
        params_changed = any(not torch.equal(pre_update_params[name], post_update_params[name])
                            for name in pre_update_params)
        self.assertTrue(params_changed, "Parameters should be updated after training step")

    def test_action_range(self):
        if isinstance(self.env_fn().action_space, gym.spaces.Discrete):
            state = self.env_fn().reset()
            action = self.agent.eval_step(state)
            self.assertIn(action, range(self.env_fn().action_space.n),
                        "Action should be within the valid range of discrete actions")

    def test_replay_buffer_integration(self):
        data = {'obs': np.array([0.0, 0.0]), 'act': 1, 'reward': 0.0, 'obs2': np.array([1.0, 1.0]), 'done': False}
        self.agent.replay.feed(data)
        sampled_data = self.agent.replay.sample()
        self.assertEqual(len(sampled_data), 5, "Sampled data should contain 5 elements (states, actions, rewards, next_states, dones)")

    def test_environment_reset(self):
        initial_state = self.env_fn().reset()
        self.assertIn(initial_state, [(1, 11)], "Initial state should be in the set of defined start states")

    def test_target_network_sync(self):
        original_q1q2 = copy.deepcopy(self.agent.ac.q1q2.state_dict())
        self.agent.sync_target()
        for param_name in self.agent.ac.q1q2.state_dict():
            self.assertTrue(torch.equal(original_q1q2[param_name], 
                                        self.agent.ac_targ.q1q2.state_dict()[param_name]),
                            "Target network parameters should match those of the main network after synchronization")

    def test_logging_functionality(self):
        log_initial_length = len(self.agent.logger.log_buffer)
        self.agent.logger.info("Test logging")
        self.assertTrue(len(self.agent.logger.log_buffer) > log_initial_length, "Logging should add entries to the log buffer")



    def test_loss_computation_beh_pi(self):
        data = self.agent.get_data()
        loss_beh_pi, _ = self.agent.compute_loss_beh_pi(data)
        self.assertIsInstance(loss_beh_pi, torch.Tensor, "Loss behavior policy should be a torch tensor")
        self.assertGreaterEqual(loss_beh_pi.item(), 0, "Loss behavior policy should be non-negative")

    def test_loss_computation_value(self):
        data = self.agent.get_data()
        loss_value, _, _ = self.agent.compute_loss_value(data)
        self.assertIsInstance(loss_value, torch.Tensor, "Loss value should be a torch tensor")
        self.assertGreaterEqual(loss_value.item(), 0, "Loss value should be non-negative")

    def test_loss_computation_q(self):
        data = self.agent.get_data()
        loss_q, _ = self.agent.compute_loss_q(data)
        self.assertIsInstance(loss_q, torch.Tensor, "Loss Q should be a torch tensor")
        self.assertGreaterEqual(loss_q.item(), 0, "Loss Q should be non-negative")

    def test_loss_computation_pi(self):
        data = self.agent.get_data()
        loss_pi, _ = self.agent.compute_loss_pi(data)
        self.assertIsInstance(loss_pi, torch.Tensor, "Loss policy should be a torch tensor")
        self.assertGreaterEqual(loss_pi.item(), 0, "Loss policy should be non-negative")

if __name__ == '__main__':
    unittest.main()
