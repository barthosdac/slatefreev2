from typing import Callable, Union, Tuple
from enum import Enum
import random

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import joblib

from recsim.agents.replay_buffer import ReplayBuffer
#from slates.utils.logging_utils import RunningAverage
#from slates.utils.random import RNG


class TrainMode(Enum):
    TRAIN = 1
    EVAL = 2

def select_slate_topk(K,q_values) :
    """
    Args
       K: int, number of item to select among the candidates
       q_values: [batch_size,N] or [N] tensor, q-values of the candidates

    Returns:
       [batch_size,K] or [K] tensor, indices of selected items    
    """
    return torch.topk(q_values, K,axis=-1).indices

def compute_target_mse(reward, gamma, next_q_values,
                          terminals):
  """Computes the optimal target Q value with the greedy algorithm.

  This algorithm corresponds to the method "TT" in
  Ie et al. https://arxiv.org/abs/1905.12767.

  Args:
    reward: [batch_size] tensor, the immediate reward.
    gamma: float, discount factor with the usual RL meaning.
    next_q_values: [batch_size, K] tensor, the q values of the
      documents in the next step.
    next_candidates: [batch_size, 1 + N] tensor, the features for the
      user and the docuemnts in the next step.
    terminals: [batch_size] tensor, indicating if this is a terminal step.

  Returns:
    [batch_size] tensor, the target q values.
  """
  next_q_values = next_q_values.max(axis=-1).values
  return reward + gamma * next_q_values * (1. - terminals)

def compute_target_sum(reward, gamma, next_q_values,
                          terminals):
  """
  Args:
    reward: [batch_size] tensor, the immediate reward.
    gamma: float, discount factor with the usual RL meaning.
    next_q_values: [batch_size, K] tensor, the q values of the
      documents in the next step.
    next_candidates: [batch_size, N] tensor, the features for the
      user and the docuemnts in the next step.
    terminals: [batch_size] tensor, indicating if this is a terminal step.

  Returns:
    [batch_size] tensor, the target q values.
  """
  next_q_values = next_q_values.mean(axis=-1)
  return reward + gamma * next_q_values * (1. - terminals)

def compute_loss_mse(target,slate_q_values) :
  """
  Args:
    target: [batch_size] tensor,
    slate_q_values: [batch_size,K]

  Returns:
    float tensor,
  """
  error=((slate_q_values-target[:,None])**2).mean(axis=-1)
  return error.mean()
def compute_loss_sum(target,slate_q_values) :
  """
  Args:
    target: [batch_size] tensor,
    slate_q_values: [batch_size,K]

  Returns:
    float tensor,
  """
  error=(slate_q_values.mean(axis=-1)-target)**2
  return error.mean()

class DQN(nn.Module):
    """DQN https://www.nature.com/articles/nature14236.pdf"""

    def __init__(
        self,
        input_dim,
        hidden_layers
    ):
        super().__init__()
        self.ffn = nn.Sequential()
        self.ffn.append(nn.Linear(input_dim, hidden_layers[0]))
        self.ffn.append(nn.ReLU())
        for k in range(len(hidden_layers)-1) :
            self.ffn.append(nn.Linear(hidden_layers[k], hidden_layers[k+1]))
            self.ffn.append(nn.ReLU())
        self.ffn.append(nn.Linear(hidden_layers[-1], 1))

    def forward(self, x: torch.tensor):
        x = self.ffn(x)
        return x.reshape(x.shape[:-1])

class SlateFreeAgent:
    def __init__(
        self,
        env_params,
        compute_target_fnct,
        compute_loss_fnct,
        select_slate_fnct=select_slate_topk,
        sarsa=False,
        buffer_size: int=1000000,
        hidden_layers=[256,16],
        learning_rate: float = 0.00025,
        batch_size: int = 32,
        gamma: float = 1.,
        grad_norm_clip: float = 1.0,

        target_update_frequency: int = 8000,
        epsilon_decay_period=250000,
        min_replay_history=20000,
        epsilon=0.01,

        **kwargs,
    ):

        self.env_params=env_params
        self.sarsa=sarsa

        self.epsilon_decay_period=epsilon_decay_period
        self.min_replay_history=min_replay_history
        self.epsilon=epsilon

        # Initialize environment & networks
        self.policy_network = DQN(2*env_params['d'],hidden_layers)
        self.target_network = DQN(2*env_params['d'],hidden_layers)
        # Ensure network's parameters are the same
        self.target_update()
        self.target_network.eval()

        self.compute_target_fnct=compute_target_fnct
        self.compute_loss_fnct=compute_loss_fnct
        self.select_slate_fnct=select_slate_fnct

        self.optimizer = optim.Adam(self.policy_network.parameters(), lr=learning_rate)

        self.replay_buffer = ReplayBuffer(
            buffer_size,
            env_params['N'],
            env_params['K'],
            env_params['d']
        )

        # Hyperparameters
        self.batch_size = batch_size
        self.gamma = gamma
        self.grad_norm_clip = grad_norm_clip
        self.target_update_frequency = target_update_frequency

        # Logging
        self.num_train_steps = 0

        self.train_mode = TrainMode.TRAIN


    def eval_on(self) -> None:
        self.train_mode = TrainMode.EVAL
        self.policy_network.eval()

    def eval_off(self) -> None:
        self.train_mode = TrainMode.TRAIN
        self.policy_network.train()

    def espilon_fnct(self) :
        steps_left = self.epsilon_decay_period + self.min_replay_history - self.num_train_steps
        bonus = (1.0 - self.epsilon) * steps_left / self.epsilon_decay_period
        bonus = np.clip(bonus, 0., 1. - self.epsilon)
        return self.epsilon + bonus
    
    @torch.no_grad()
    def get_action(self, candidates) -> int:
        """Use policy_network to get an e-greedy action given the current obs."""
        epsilon = self.espilon_fnct()
        if self.train_mode == TrainMode.TRAIN and np.random.random() < epsilon:
            args=np.arange(self.env_params['N'])
            np.random.shuffle(args)
            return args[:self.env_params['K']]
        candidates=torch.as_tensor(candidates,dtype=torch.float32)
        user=torch.as_tensor(self.user, dtype=torch.float32)
        user_expended = user[None,:].expand(self.env_params['N'],-1)
        user_candidates = torch.concatenate([user_expended,candidates],axis=-1)
        q_values = self.policy_network(user_candidates)

        return self.select_slate_fnct(self.env_params['K'],q_values)

    def observe(self, candidates, action, choice, reward, done) -> None:
        if self.train_mode == TrainMode.TRAIN:
            self.replay_buffer.store(self.user, candidates, action, choice, reward, done)
            self.train()

    def observe_candidates(self, user,candidates) -> None:
        self.user=user
        if self.train_mode == TrainMode.TRAIN:
            self.replay_buffer.store_obs(user, candidates)

    def train(self) -> None:
        """Perform one gradient step of the network"""
        if not self.replay_buffer.can_sample(self.batch_size):
            return

        self.eval_off()
        users, candidates, actions, choices, rewards, candidates_next, actions_next, dones = self.replay_buffer.sample(
            self.batch_size
        )

        # We pull obss/next_obss as [batch-size x 1 x obs-dim]
        users = torch.as_tensor(users, dtype=torch.float32)
        candidates = torch.as_tensor(candidates, dtype=torch.float32)
        choices = torch.as_tensor(choices, dtype=torch.int64)
        actions = torch.as_tensor(actions, dtype=torch.int64)
        actions_next = torch.as_tensor(actions_next, dtype=torch.int64)
        rewards = torch.as_tensor(rewards,dtype=torch.float32)
        candidates_next = torch.as_tensor(candidates_next, dtype=torch.float32)
        dones = torch.as_tensor(dones, dtype=torch.long)

        #Concatenate
        users_expended = users[:,None,:].expand(-1,self.env_params['N'],-1)
        users_candidates = torch.concatenate([users_expended,candidates],axis=-1)
        users_candidates_next = torch.concatenate([users_expended,candidates_next],axis=-1)

        #compute target
        with torch.no_grad():
            next_q_values_target=self.target_network(users_candidates_next)
            if not self.sarsa :
                next_q_values_policy=self.policy_network(users_candidates_next)
                actions_next = self.select_slate_fnct(self.env_params['K'],next_q_values_policy)
            target=self.compute_target_fnct(rewards, self.gamma, next_q_values_target.gather(axis=1,index=actions_next), dones)
            
        q_values = self.policy_network(users_candidates)
        loss=self.compute_loss_fnct(target,q_values)

        # Optimization step
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        #norm = torch.nn.utils.clip_grad_norm_(
        #    self.policy_network.parameters(),
        #    self.grad_norm_clip,
        #    error_if_nonfinite=True,
        #)
        #self.grad_norms.add(norm.item())
        self.optimizer.step()
        self.num_train_steps += 1

        if self.num_train_steps % self.target_update_frequency == 0:
            self.target_update()

    def target_update(self) -> None:
        """Hard update where we copy the network parameters from the policy network to the target network"""
        self.target_network.load_state_dict(self.policy_network.state_dict())

def create_slatefree_agent(agent_name,env_params,**kwargs) :
    if agent_name == 'slatefree_q_sum' :
        return SlateFreeAgent(env_params,
                        compute_target_sum,
                        compute_loss_sum,
                        sarsa=False,
                        **kwargs)
    elif agent_name == 'slatefree_q_mse' :
        return SlateFreeAgent(env_params,
                        compute_target_mse,
                        compute_loss_mse,
                        sarsa=False,
                        **kwargs)
    elif agent_name == 'slatefree_sarsa_sum' :
        return SlateFreeAgent(env_params,
                        compute_target_sum,
                        compute_loss_sum,
                        sarsa=True,
                        **kwargs)
    elif agent_name == 'slatefree_sarsa_mse' :
        return SlateFreeAgent(env_params,
                        compute_target_mse,
                        compute_loss_mse,
                        sarsa=True,
                        **kwargs)
    else :
        raise Exception("agent_name not correct")