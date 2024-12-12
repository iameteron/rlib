import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal

from .BaseAlgorithm import BaseAlgorithm


class StochasticPolicy(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(StochasticPolicy, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.shared = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
        )

        self.mean_layer = nn.Linear(128, action_dim)
        self.log_std_layer = nn.Linear(128, action_dim)

    def forward(self, state):
        x = self.shared(state)
        mean = torch.tanh(self.mean_layer(x)) 
        log_std = self.log_std_layer(x)
        log_std = torch.clamp(log_std, min=-20, max=2)
        return mean, log_std


class PPO(BaseAlgorithm):
    def __init__(
        self,
        env: gym.Env,
        pi_model: nn.Module = None,
        v_model: nn.Module = None,
        gamma: float = 0.999,
        epsilon: float = 0.1,
        batch_size: int = 64,
        trajectory_n: int = 20,
        epoch_n: int = 30,
        advantage_type: str = "bellman",
    ):
        self.env = env

        self.gamma = gamma
        self.epsilon = epsilon

        self.state_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.shape[0]

        self.action_low = self.env.action_space.low[0]
        self.action_high = self.env.action_space.high[0]
        self.action_mean = (self.action_low + self.action_high) / 2
        self.action_dev = (self.action_high - self.action_low) / 2

        self.pi_model = StochasticPolicy(self.state_dim, self.action_dim)

        self.v_model = nn.Sequential(
            nn.Linear(self.state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

        self.pi_optimizer = torch.optim.Adam(self.pi_model.parameters())
        self.v_optimizer = torch.optim.Adam(self.v_model.parameters())

        self.batch_size = batch_size
        self.trajectory_n = trajectory_n
        self.epoch_n = epoch_n

        self.advantage_type = advantage_type

    def _get_action(self, state, deterministic: bool = False):
        mean, log_std = self.pi_model.forward(torch.FloatTensor(state))
        dist = Normal(mean, torch.exp(log_std))
        if deterministic:
            action = mean
        else:
            action = dist.sample()

        return action.numpy().reshape(self.action_dim)
    
    def _unscale_action(self, action):
        return self.action_mean + self.action_dev * action

    def predict(self, state, deterministic: bool = False):
        action = self._get_action(state, deterministic)
        return self._unscale_action(action)

    def train(self, episode_n, verbose: bool = True):
        total_rewards = []
        for episode in range(episode_n):
            states, actions, rewards, dones = [], [], [], []

            for _ in range(self.trajectory_n):
                trajectory = self.get_trajectory()

                states.extend(trajectory["states"])
                actions.extend(trajectory["actions"])
                rewards.extend(trajectory["rewards"])
                dones.extend(trajectory["dones"])

                total_rewards.append(np.sum(trajectory["rewards"]))

            if verbose:
                mean_rewards = np.mean(total_rewards[-self.trajectory_n :])
                print(f"Episode: {episode}, ", f"Total Episode Reward: {mean_rewards}")

            self._fit(states, actions, rewards, dones)

        return total_rewards

    def get_trajectory(self):
        trajectory = {"states": [], "actions": [], "rewards": [], "dones": []}

        state, _ = self.env.reset()

        while True:
            trajectory["states"].append(state)

            action = self._get_action(state)
            trajectory["actions"].append(action)

            next_state, reward, terminated, truncated, _ = self.env.step(
                self._unscale_action(action)
            )
            trajectory["rewards"].append(reward)
            trajectory["dones"].append(terminated)

            state = next_state

            done = terminated or truncated
            if done:
                break

        return trajectory

    def _fit(self, states, actions, rewards, dones):
        states, actions, rewards, dones = map(
            np.array, [states, actions, rewards, dones]
        )

        rewards, dones = rewards.reshape(-1, 1), dones.reshape(-1, 1)

        next_states = np.zeros_like(states)
        next_states[:-1] = states[1:]

        rewards_to_go = np.zeros(rewards.shape)
        rewards_to_go[-1] = rewards[-1]
        for t in range(rewards_to_go[0].shape[0] - 2, -1, -1):
            rewards_to_go[t] = rewards[t] + (1 - dones[t]) * self.gamma * rewards_to_go

        states, next_states, actions, rewards, rewards_to_go, dones = map(
            torch.FloatTensor,
            [states, next_states, actions, rewards, rewards_to_go, dones],
        )

        mean, log_std = self.pi_model.forward(states)

        dist = Normal(mean, torch.exp(log_std))
        old_log_probs = dist.log_prob(actions).detach()

        for _ in range(self.epoch_n):
            idxs = np.random.permutation(states.shape[0])

            for i in range(0, states.shape[0], self.batch_size):
                b_idxs = idxs[i : i + self.batch_size]
                b_states = states[b_idxs]
                b_next_states = next_states[b_idxs]
                b_dones = dones[b_idxs]
                b_actions = actions[b_idxs]
                b_rewards = rewards[b_idxs]
                b_returns = rewards_to_go[b_idxs]
                b_old_log_probs = old_log_probs[b_idxs]

                if self.advantage_type == "monte_carlo":
                    b_advantage = b_returns.detach() - self.v_model(b_states)

                if self.advantage_type == "bellman":
                    b_advantage = (
                        b_rewards.detach()
                        + (1 - b_dones.detach())
                        * self.gamma
                        * self.v_model(b_next_states.detach())
                        - self.v_model(b_states)
                    )

                b_mean, b_log_std = self.pi_model.forward(b_states)

                b_dist = Normal(b_mean, torch.exp(b_log_std))
                b_new_log_probs = b_dist.log_prob(b_actions)

                b_ratio = torch.exp(b_new_log_probs - b_old_log_probs)
                pi_loss_1 = b_ratio * b_advantage.detach()
                pi_loss_2 = (
                    torch.clamp(b_ratio, 1.0 - self.epsilon, 1.0 + self.epsilon)
                    * b_advantage.detach()
                )

                pi_loss = -torch.mean(torch.min(pi_loss_1, pi_loss_2))

                pi_loss.backward()
                self.pi_optimizer.step()
                self.pi_optimizer.zero_grad()

                v_loss = torch.mean(b_advantage**2)

                v_loss.backward()
                self.v_optimizer.step()
                self.v_optimizer.zero_grad()
