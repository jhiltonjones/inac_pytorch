import copy
import numpy as np
from sklearn.mixture import GaussianMixture
from core.agent import base
import core.network.net_factory as network
import core.network.activations as activations
import core.utils.torch_utils as torch_utils
from collections import namedtuple
import os
import torch

from core.network.policy_factory import MLPCont, MLPDiscrete
from core.network.network_architectures import DoubleCriticNetwork, DoubleCriticDiscrete, FCNetwork

class InSampleACOnline(base.Agent):
    def __init__(self, cfg):
        super(InSampleACOnline, self).__init__(cfg)
        self.cfg = cfg
        
        def get_policy_func():
            if cfg.policy_fn_config['policy_type'] == "policy-cont":
                pi = MLPCont(cfg.device, np.prod(cfg.policy_fn_config['in_dim']),
                                           cfg.action_dim, cfg.policy_fn_config['hidden_units'],
                                           action_range=cfg.action_range,
                                           rep=None,
                                           init_type='xavier',
                                           info=cfg.policy_fn_config.get('info', None),
                                           )
            elif cfg.policy_fn_config['policy_type'] == 'policy-discrete':
                pi = MLPDiscrete(cfg.device, np.prod(cfg.policy_fn_config['in_dim']),
                                               cfg.action_dim, cfg.policy_fn_config['hidden_units'],
                                               rep=None,
                                               init_type='xavier',
                                               info=cfg.policy_fn_config.get('info', None),
                                               )
            return pi

        def get_critic_func():
            if cfg.critic_fn_config['network_type'] == 'fc':
                q1q2 = DoubleCriticDiscrete(cfg.device, np.prod(cfg.critic_fn_config['in_dim']),
                                                                          cfg.critic_fn_config['hidden_units'],
                                                                          cfg.critic_fn_config.get('out_dim', cfg.action_dim),
                                                                          rep=None,
                                                                          init_type=cfg.critic_fn_config.get('init_type', 'xavier'),
                                                                          info=cfg.critic_fn_config.get('info', None),
                                                                          )
            elif cfg.critic_fn_config['network_type'] == 'fc-insert-input':
                q1q2 = DoubleCriticNetwork(cfg.device, np.prod(cfg.critic_fn_config['in_dim']), cfg.action_dim,
                                                                         cfg.critic_fn_config['hidden_units'],
                                                                         rep=None)
            return q1q2
        
        
        pi = get_policy_func()
        q1q2 = get_critic_func()
        AC = namedtuple('AC', ['q1q2', 'pi'])
        self.ac = AC(q1q2=q1q2, pi=pi)

        pi_target = get_policy_func()
        q1q2_target = get_critic_func()
        q1q2_target.load_state_dict(q1q2.state_dict())
        pi_target.load_state_dict(pi.state_dict())
        ACTarg = namedtuple('ACTarg', ['q1q2', 'pi'])
        self.ac_targ = ACTarg(q1q2=q1q2_target, pi=pi_target)
        self.ac_targ.q1q2.load_state_dict(self.ac.q1q2.state_dict())
        self.ac_targ.pi.load_state_dict(self.ac.pi.state_dict())
        self.value_net = None
        self.pi_optimizer = torch.optim.Adam(list(self.ac.pi.parameters()), cfg.learning_rate)
        self.q_optimizer = torch.optim.Adam(list(self.ac.q1q2.parameters()), cfg.learning_rate)
        self.polyak = cfg.polyak #0 is hard sync

        if 'load_params' in self.cfg.policy_fn_config and self.cfg.policy_fn_config['load_params']:
            self.load_actor_fn(cfg.policy_fn_config['path'])
        if 'load_params' in self.cfg.critic_fn_config and self.cfg.critic_fn_config['load_params']:
            self.load_critic_fn(cfg.critic_fn_config['path'])

        if self.cfg.discrete_control:
            self.get_q_value = self.get_q_value_discrete
            self.get_q_value_target = self.get_q_value_target_discrete
        else:
            self.get_q_value = self.get_q_value_cont
            self.get_q_value_target = self.get_q_value_target_cont
        
        self.tau = cfg.tau
        self.value_net = FCNetwork(cfg.device, np.prod(cfg.val_fn_config['in_dim']),
                                        cfg.val_fn_config['hidden_units'], 1,
                                        rep=None,
                                        init_type='xavier',
                                        info=cfg.val_fn_config.get('info', None)
                                        )
        if 'load_params' in self.cfg.val_fn_config and self.cfg.val_fn_config['load_params']:
            self.load_state_value_fn(cfg.val_fn_config['path'])

        self.value_optimizer = torch.optim.Adam(list(self.value_net.parameters()), cfg.learning_rate)
        self.beh_pi = get_policy_func()
        self.beh_pi_optimizer = torch.optim.Adam(list(self.beh_pi.parameters()), cfg.learning_rate)

        # self.clip_grad_param = cfg.clip_grad_param
        self.exp_threshold = cfg.exp_threshold
        # self.beta_threshold = 1e-3

        if cfg.agent_name == 'InSampleACOnline' and cfg.load_offline_data:
            self.fill_offline_data_to_buffer()

    def get_q_value_discrete(self, o, a, with_grad=False):
        if with_grad:
            q1_pi, q2_pi = self.ac.q1q2(o)
            q1_pi, q2_pi = q1_pi[np.arange(len(a)), a], q2_pi[np.arange(len(a)), a]
            q_pi = torch.min(q1_pi, q2_pi)
        else:
            with torch.no_grad():
                q1_pi, q2_pi = self.ac.q1q2(o)
                q1_pi, q2_pi = q1_pi[np.arange(len(a)), a], q2_pi[np.arange(len(a)), a]
                q_pi = torch.min(q1_pi, q2_pi)
        return q_pi.squeeze(-1), q1_pi.squeeze(-1), q2_pi.squeeze(-1)

    def get_q_value_target_discrete(self, o, a):
        with torch.no_grad():
            q1_pi, q2_pi = self.ac_targ.q1q2(o)
            q1_pi, q2_pi = q1_pi[np.arange(len(a)), a], q2_pi[np.arange(len(a)), a]
            q_pi = torch.min(q1_pi, q2_pi)
        return q_pi.squeeze(-1), q1_pi.squeeze(-1), q2_pi.squeeze(-1)

    def get_q_value_cont(self, o, a, with_grad=False):
        if with_grad:
            q1_pi, q2_pi = self.ac.q1q2(o, a)
            q_pi = torch.min(q1_pi, q2_pi)
        else:
            with torch.no_grad():
                q1_pi, q2_pi = self.ac.q1q2(o, a)
                q_pi = torch.min(q1_pi, q2_pi)
        return q_pi.squeeze(-1), q1_pi.squeeze(-1), q2_pi.squeeze(-1)

    def get_q_value_target_cont(self, o, a):
        with torch.no_grad():
            q1_pi, q2_pi = self.ac_targ.q1q2(o, a)
            q_pi = torch.min(q1_pi, q2_pi)
        return q_pi.squeeze(-1), q1_pi.squeeze(-1), q2_pi.squeeze(-1)

    def sync_target(self):
        with torch.no_grad():
            for p, p_targ in zip(self.ac.q1q2.parameters(), self.ac_targ.q1q2.parameters()):
                p_targ.data.mul_(self.polyak)
                p_targ.data.add_((1 - self.polyak) * p.data)
            for p, p_targ in zip(self.ac.pi.parameters(), self.ac_targ.pi.parameters()):
                p_targ.data.mul_(self.polyak)
                p_targ.data.add_((1 - self.polyak) * p.data)

    def load_actor_fn(self, parameters_dir):
        path = os.path.join(self.data_root, parameters_dir)
        self.ac.pi.load_state_dict(torch.load(path, map_location=self.device))
        self.ac_targ.pi.load_state_dict(self.ac.pi.state_dict())
        self.logger.info("Load actor function from {}".format(path))

    def load_critic_fn(self, parameters_dir):
        path = os.path.join(self.data_root, parameters_dir)
        self.ac.q1q2.load_state_dict(torch.load(path, map_location=self.device))
        self.ac_targ.q1q2.load_state_dict(self.ac.q1q2.state_dict())
        self.logger.info("Load critic function from {}".format(path))

    def load_state_value_fn(self, parameters_dir):
        path = os.path.join(self.data_root, parameters_dir)
        self.value_net.load_state_dict(torch.load(path, map_location=self.device))
        self.logger.info("Load state value function from {}".format(path))

    #-----------------------------------------------------------------------------------------------
    def compute_loss_beh_pi(self, data):
        """L_{\omega}, learn behavior policy"""
        states, actions = data['obs'], data['act']
        beh_log_probs = self.beh_pi.get_logprob(states, actions)
        beh_loss = -beh_log_probs.mean()
        return beh_loss, beh_log_probs
    
    def compute_loss_value(self, data):
        """L_{\phi}, learn z for state value, v = tau log z"""
        states = data['obs']
        v_phi = self.value_net(states).squeeze(-1)
        with torch.no_grad():
            actions, log_probs = self.ac.pi(states)
            min_Q, _, _ = self.get_q_value_target(states, actions)
            # beh_log_prob = self.beh_pi.get_logprob(states, actions)
            # beh_log_prob = self.ac.pi.get_logprob(states, actions)
        target = min_Q - self.tau * log_probs#beh_log_prob
        value_loss = (0.5 * (v_phi - target) ** 2).mean()
        return value_loss, v_phi.detach().numpy(), log_probs.detach().numpy()
    
    def get_state_value(self, state):
        with torch.no_grad():
            value = self.value_net(state).squeeze(-1)
        return value

    def compute_loss_q(self, data):
        states, actions, rewards, next_states, dones = data['obs'], data['act'], data['reward'], data['obs2'], data['done']
        with torch.no_grad():
            next_actions, log_probs = self.ac.pi(next_states)
        min_Q, _, _ = self.get_q_value_target(next_states, next_actions)
        q_target = rewards + self.gamma * (1 - dones) * (min_Q - self.tau * log_probs)
    
        minq, q1, q2 = self.get_q_value(states, actions, with_grad=True)
    
        critic1_loss = (0.5 * (q_target - q1) ** 2).mean()
        critic2_loss = (0.5 * (q_target - q2) ** 2).mean()
        loss_q = (critic1_loss + critic2_loss) * 0.5
        # q_info = dict(Q1Vals=q1.detach().numpy(),
        #               Q2Vals=q2.detach().numpy())
        q_info = minq.detach().numpy()
        return loss_q, q_info

    def compute_loss_pi(self, data):
        """L_{\psi}, extract learned policy"""
        states, actions = data['obs'], data['act']

        log_probs = self.ac.pi.get_logprob(states, actions)
        min_Q, _, _ = self.get_q_value(states, actions, with_grad=False)
        # min_Q, _, _ = self.get_q_value_target(states, actions)
        with torch.no_grad():
            value = self.get_state_value(states)
            beh_log_prob = self.beh_pi.get_logprob(states, actions)

        clipped = torch.clip(torch.exp((min_Q - value) / self.tau - beh_log_prob), self.eps, self.exp_threshold)
        pi_loss = -(clipped * log_probs).mean()
        return pi_loss, ""
    
    def update_beta(self, data):
        loss_beh_pi, _ = self.compute_loss_beh_pi(data)
        self.beh_pi_optimizer.zero_grad()
        loss_beh_pi.backward()
        self.beh_pi_optimizer.step()
        # print(loss_beh_pi)
        return loss_beh_pi

    def update(self, data):
        loss_beta = self.update_beta(data).item()
        
        self.value_optimizer.zero_grad()
        loss_vs, v_info, logp_info = self.compute_loss_value(data)
        loss_vs.backward()
        self.value_optimizer.step()

        loss_q, qinfo = self.compute_loss_q(data)
        self.q_optimizer.zero_grad()
        loss_q.backward()
        self.q_optimizer.step()

        loss_pi, _ = self.compute_loss_pi(data)
        self.pi_optimizer.zero_grad()
        loss_pi.backward()
        self.pi_optimizer.step()
        
        if self.use_target_network and self.total_steps % self.target_network_update_freq == 0:
            self.sync_target()

        return {"beta": loss_beta,
                "actor": loss_pi.item(),
                "critic": loss_q.item(),
                "value": loss_vs.item(),
                "q_info": qinfo.mean(),
                "v_info": v_info.mean(),
                "logp_info": logp_info.mean(),
                }

    def save(self):
        parameters_dir = self.parameters_dir
        path = os.path.join(parameters_dir, "actor_net")
        torch.save(self.ac.pi.state_dict(), path)
    
        path = os.path.join(parameters_dir, "critic_net")
        torch.save(self.ac.q1q2.state_dict(), path)
    
        path = os.path.join(parameters_dir, "vs_net")
        torch.save(self.value_net.state_dict(), path)


class InSampleAC(InSampleACOnline):
    def __init__(self, cfg):
        super(InSampleAC, self).__init__(cfg)
        self.offline_param_init()

    def get_data(self):
        return self.get_offline_data()

    def feed_data(self):
        self.update_stats(0, None)
        return

