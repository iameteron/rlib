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
            data: Dict[torch.tensor]: Tensor shape (N, non-zero)
        """
        data = {
            "observations": self.process_field(self.observations, reshape=False),
            "actions": self.process_field(self.actions, reshape=False),
            "rewards": self.process_field(self.rewards),
            "terminated": self.process_field(self.terminated, torch.bool),
            "truncated": self.process_field(self.truncated, torch.bool),
        }

        data["log_probs"] = torch.cat(self.log_probs, dim=0)
        data["q_estimations"] = get_returns(data["rewards"], data["terminated"])

        return data

    def get_trajectories(self):
        trajectories = []

        dones = [term or trunc for term, trunc in zip(self.terminated, self.truncated)]
        indices = [i for i, value in enumerate(dones) if value]

        trajectory = {}
        trajectory["rewards"] = self.rewards[0: indices[0] + 1]
        trajectories.append(trajectory)

        for i in range(len(indices) - 1):
            trajectory = {}
            trajectory["rewards"] = self.rewards[indices[i] + 1: indices[i + 1] + 1]
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
            