import gymnasium as gym
import numpy as np
from tqdm import tqdm

from .BaseAlgorithm import BaseAlgorithm


class QLearning(BaseAlgorithm):
    def __init__(
        self, env: gym.Env, eps: float = 1e-2, gamma: float = 0.999, alpha: float = 1e-1
    ):
        self.env = env
        self.state_n = self.env.observation_space.n
        self.action_n = self.env.action_space.n

        self.eps = eps
        self.alpha = alpha
        self.gamma = gamma

        self.q_values = np.zeros((self.state_n, self.action_n))

    def predict(self, state: int):
        """
        Get epsilon-greedy action
        """
        action = np.argmax(self.q_values[state, :])
        probs = self.eps / self.action_n * np.ones(self.action_n)
        probs[action] = 1 - self.eps + self.eps / self.action_n
        return np.random.choice(np.arange(self.action_n), p=probs)


    def train(self, episode_n, verbose: bool = True):
        """
        Train function
        """
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

            self._fit(obs, action, reward, terminated, next_obs)

            done = terminated or truncated
            obs = next_obs

        return rewards

    def _fit(self, obs, action, reward, terminated, next_obs):
        """
        Update Q-values 
        """
        self.q_values[obs][action] += self.alpha * (
            reward
            + self.gamma
            * np.max(self.q_values[next_obs, :] - self.q_values[obs, action])
        )
