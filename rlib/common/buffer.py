import numpy as np
import torch

from .utils import get_returns


class RolloutBuffer:
    def __init__(self, gamma=1):
        self.gamma = gamma
        self.clear()

    def clear(self):
        self.observations = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.terminated = []
        self.truncated = []

    def add_transition(self, obs, action, log_prob, reward, terminated, truncated):
        self.observations.append(obs)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.terminated.append(terminated)
        self.truncated.append(truncated)

    def process_field(
        self, value, dtype=torch.float32, reshape=True, reshape_dim=(-1, 1)
    ):
        if reshape:
            return torch.tensor(value, dtype=dtype).reshape(reshape_dim)
        else:
            return torch.tensor(value, dtype=dtype)

    def get_data(self):
        """
        Returns:
            data: Dict[torch.tensor]: Dict of tensors shape (N, dim)
        """
        data = {
            "observations": self.process_field(self.observations, reshape=False),
            "actions": self.process_field(self.actions, reshape=False),
            "rewards": self.process_field(self.rewards),
            "terminated": self.process_field(self.terminated, torch.int8),
            "truncated": self.process_field(self.truncated, torch.int8),
        }

        data["log_probs"] = torch.cat(self.log_probs, dim=0)

        data["dones"] = torch.bitwise_or(data["terminated"], data["truncated"])

        data["q_estimations"] = get_returns(
            data["rewards"], data["dones"], self.gamma
        )

        return data

    def _get_trajectory_from_data(self, data, start, end):
        trajectory = {}

        trajectory["observations"] = data["observations"][start:end]
        trajectory["actions"] = data["actions"][start:end]
        trajectory["rewards"] = data["rewards"][start:end]
        trajectory["q_estimations"] = data["q_estimations"][start:end]

        return trajectory

    def get_trajectories(self, data):
        """
        Returns:
            trajectories: List[Dict[torch.tensor]]: List of trajectory dicts
        """
        trajectories = []
        indices = torch.nonzero(data["dones"].squeeze()).squeeze().numpy()

        trajectory = self._get_trajectory_from_data(data, 0, indices[0] + 1)
        trajectories.append(trajectory)

        for i in range(len(indices) - 1):
            start = indices[i] + 1
            end = indices[i + 1] + 1
            trajectory = self._get_trajectory_from_data(data, start, end)
            trajectories.append(trajectory)

        return trajectories

    def collect_rollouts(self, env, policy, rollout_size=None, trajectories_n=None):
        self.clear()
        trajectories_collected = 0
        steps_collected = 0

        while True:
            obs, _ = env.reset()

            while True:
                action, log_prob_action = policy.predict(obs)
                next_obs, reward, terminated, truncated, _ = env.step(action)

                self.add_transition(
                    obs, action, log_prob_action, reward, terminated, truncated
                )
                obs = next_obs

                steps_collected += 1
                if rollout_size and steps_collected >= rollout_size:
                    return

                if terminated or truncated:
                    break

            trajectories_collected += 1

            if trajectories_n and trajectories_collected >= trajectories_n:
                return


class ReplayBuffer:
    def __init__(
        self,
        obs_dim,
        action_dim,
        max_size: int = 10000,
    ):
        self.max_size = max_size

        self.size = 0
        self.pointer = 0
        self.old_done = 0
        self.new_done = 0
        self.curr_obs = None

        self.observations = np.zeros((max_size, obs_dim))
        self.next_observations = np.zeros((max_size, obs_dim))
        self.actions = np.zeros((max_size, action_dim))
        self.rewards = np.zeros((max_size, 1))
        self.terminated = np.zeros((max_size, 1))
        self.truncated = np.zeros((max_size, 1))

    def add_transition(self, obs, next_obs, action, reward, terminated, truncated):
        self.observations[self.pointer] = obs
        self.next_observations[self.pointer] = next_obs
        self.actions[self.pointer] = action
        self.rewards[self.pointer] = reward
        self.terminated[self.pointer] = terminated
        self.truncated[self.pointer] = truncated

        self.done = terminated or truncated
        if self.done:
            self.old_done = self.new_done
            self.new_done = self.pointer

        self.pointer = (self.pointer + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def collect_transition(self, env, policy):
        if self.curr_obs is None:
            self.curr_obs, _ = env.reset()

        action, _ = policy.predict(self.curr_obs)
        next_obs, reward, terminated, truncated, _ = env.step(action)
        self.add_transition(
            self.curr_obs, next_obs, action, reward, terminated, truncated
        )

        if terminated or truncated:
            next_obs, _ = env.reset()

        self.curr_obs = next_obs

    def get_batch(self, batch_size):
        assert self.size >= batch_size, "Batch size greater than buffer size"

        indices = np.random.choice(range(self.size), batch_size, replace=False)

        batch = {
            "observations": torch.tensor(
                self.observations[indices], dtype=torch.float32
            ),
            "next_observations": torch.tensor(
                self.next_observations[indices], dtype=torch.float32
            ),
            "actions": torch.tensor(self.actions[indices], dtype=torch.float32),
            "rewards": torch.tensor(self.rewards[indices], dtype=torch.float32),
            "terminated": torch.tensor(self.terminated[indices], dtype=torch.bool),
            "truncated": torch.tensor(self.truncated[indices], dtype=torch.bool),
        }

        return batch

    def get_last_trajectory(self):
        trajectories = []

        trajectory = {}
        if self.new_done < self.old_done:
            trajectory["rewards"] = np.concatenate(
                (self.rewards[self.old_done + 1 :], self.rewards[: self.new_done + 1]),
                axis=0,
            )
        else:
            trajectory["rewards"] = self.rewards[self.old_done + 1 : self.new_done + 1]

        trajectories.append(trajectory)

        return trajectories
