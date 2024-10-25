import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn

from tqdm import tqdm

from BaseAlgorithm import BaseAlgorithm
from ..common import ReplayBuffer


class DQN(BaseAlgorithm):
    def __init__(
        self, env: gym.Env, eps: float = 1e-2, gamma: float = 0.999, alpha: float = 3e-4
    ):
        self.env = env

        self.eps = eps
        self.gamma = gamma
        self.alpha = alpha

        self.observation_dim = self.env.observation_space.shape[0]
        self.action_n = self.env.action_space.n

        self.neuron_n = 128
        self.model = nn.Sequential(
            nn.Linear(self.observation_dim, self.neuron_n),
            nn.ReLU(),
            nn.Linear(self.neuron_n, self.neuron_n),
            nn.ReLU(),
            nn.Linear(self.neuron_n, self.action_n)
        )

        self.replay_buffer = ReplayBuffer()

    
    def predict(self, obs):
        greedy_action = torch.argmax(self.model(obs)).numpy()
        probs = self.eps / self.action_n * np.ones(self.action_n)
        probs[greedy_action] = 1 - self.eps + self.eps / self.action_n
        return np.random.choice(np.arange(self.action_n), p=probs)

    def train(self, episode_n, verbose: bool = True):
        for i in tqdm(range(episode_n)):
            rewards = self._fit_trajectory()
            if verbose:
                print(f"Episode: {i}, Total Episode Reward: {np.sum(rewards)}")
            self.eps *= 0.99


    def _fit_trajectory(self):
        """
        One training episode 
        """
        obs, info = self.env.reset()
        done = False
        rewards = []

        while not done:
            action = self.predict(obs)
            next_obs, reward, terminated, truncated, info = self.env.step(action)
            rewards.append(reward)

            self.replay_buffer.get_transition()

            batch = self.replay_buffer.sample_experience()
            self._fit(batch)

            done = terminated or truncated
            obs = next_obs

        return rewards


    def _fit(self, batch):
        return None