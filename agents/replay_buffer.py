import numpy as np
import random
from typing import Optional, Tuple, Union



class ReplayBuffer:
    """
    FIFO Replay Buffer which stores contexts of length ``context_len`` rather than single
        transitions

    Args:
        buffer_size: The number of transitions to store in the replay buffer
        env_obs_length: The size (length) of the environment's observation
        passed: context_len: The number of transitions that will be stored as an agent's context. Default: 1
    """

    def __init__(
        self,
        buffer_size: int,
        N,K,d
    ):
        self.buffer_size = buffer_size
        self.pos = 0


        self.candidatess=np.zeros((buffer_size,N,d),dtype=np.float32)
        self.users=np.zeros((buffer_size,d),dtype=np.float32)

        # Need the +1 so we have space to roll for the first observation
        self.choices = np.zeros(buffer_size,dtype=np.int64)
        self.actions = np.zeros((buffer_size,K),dtype=np.uint8)
        self.rewards = np.zeros(buffer_size, dtype=np.float32)
        self.dones = np.ones(buffer_size, dtype=np.bool_)

    def store(
        self,
        user: np.ndarray,
        candidates: np.ndarray,
        action: np.ndarray,
        choice: int,
        reward: int,
        done: bool
    ) -> None:
        idx=self.pos%self.buffer_size
        idx1=(self.pos+1)%self.buffer_size
        self.candidatess[idx1] = candidates
        self.users[idx1] = user
        self.actions[idx] = action
        self.choices[idx] = choice
        self.rewards[idx] = reward
        self.dones[idx] = done
        self.pos+=1

        
    def store_obs(self, user, candidates: np.ndarray) -> None:
        """Use this at the beginning of the episode to store the first obs"""
        idx=self.pos%self.buffer_size
        self.candidatess[idx] = candidates
        self.users[idx] = user


    def can_sample(self, batch_size: int) -> bool:
        return batch_size < self.pos

    def sample(
        self, batch_size: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        # Exclude the current episode we're in
        if self.pos>=self.buffer_size :
            valid_idxs=np.arange(self.buffer_size)
        else :
            valid_idxs=np.arange(self.pos)

        idxs = np.array([random.choice(valid_idxs) for _ in range(batch_size)])
        idxs1=(idxs+1)%self.buffer_size

        return (
            self.users[idxs],
            self.candidatess[idxs],
            self.actions[idxs],
            self.choices[idxs],            
            self.rewards[idxs],
            self.candidatess[idxs1],
            self.actions[idxs1],
            self.dones[idxs],
        )