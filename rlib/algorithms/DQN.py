import gymnasium as gym
import numpy as np
import torch




import torch.nn as nn

from ..common import ReplayBuffer
from .BaseAlgorithm import BaseAlgorithm


class DQN(BaseAlgorithm):
    def __init__(
        self,
        env: gym.Env,
        eps: float = 1e-2,
        gamma: float = 0.999,
        alpha: float = 3e-4,
        batch_size: int = 64,
    ):
        self.env = env

        self.eps = eps
        self.gamma = gamma
        self.alpha = alpha
        self.batch_size = batch_size

        self.observation_dim = self.env.observation_space.shape[0]
        self.action_n = self.env.action_space.n

        self.neuron_n = 128
        self.model = nn.Sequential(
            nn.Linear(self.observation_dim, self.neuron_n),
            nn.ReLU(),
            nn.Linear(self.neuron_n, self.neuron_n),
            nn.ReLU(),
            nn.Linear(self.neuron_n, self.action_n),
        )

        self.replay_buffer = ReplayBuffer()

        self.optimizer = torch.optim.Adam(self.model.parameters())

    def predict(self, state):
        greedy_action = torch.argmax(self.model(torch.FloatTensor(state)))
        probs = self.eps / self.action_n * np.ones(self.action_n)
        probs[greedy_action] = 1 - self.eps + self.eps / self.action_n
        action = np.random.choice(np.arange(self.action_n), p=probs)
        return action

    def train(self, episode_n, verbose: bool = True):
        total_rewards = []
        for i in range(episode_n):
            rewards = self._fit_trajectory()
            total_rewards.append(np.sum(rewards))
            if verbose:
                print(f"Episode: {i}, Total Episode Reward: {np.sum(rewards)}")
            self.eps *= 0.99

        return total_rewards

    def _fit_trajectory(self):
        """
        One training episode
        """
        state, info = self.env.reset()
        done = False
        rewards = []

        while not done:
            action = self.predict(state)
            next_state, reward, terminated, truncated, info = self.env.step(action)
            rewards.append(reward)

            transition = (state, action, reward, int(terminated), next_state)
            self.replay_buffer.get_tranistion(transition)

            if len(self.replay_buffer.transitions) > self.batch_size:
                batch = self.replay_buffer.sample_expirience(self.batch_size)
                self._fit(batch)

            done = terminated or truncated
            state = next_state

        return rewards

    def _fit(self, batch):
        states, actions, rewards, dones, next_states = map(torch.tensor, batch)

        targets = (
            rewards
            + (1 - dones)
            * self.gamma
            * torch.max(self.model(next_states), dim=1).values
        )

        preds = self.model(states)[(range(self.batch_size), actions)]

        loss = torch.mean((preds - targets.detach()) ** 2)
        loss.backward()

        self.optimizer.step()
        self.optimizer.zero_grad()
