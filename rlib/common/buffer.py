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

    def get_data(self):
        data = {
            "observations": torch.tensor(self.observations, dtype=torch.float32),
            "actions": torch.tensor(self.actions, dtype=torch.float32),
            "log_probs": torch.cat(self.log_probs, dim=0),
            "rewards": torch.tensor(self.rewards, dtype=torch.float32),
            "terminated": torch.tensor(self.terminated, dtype=torch.bool),
            "truncated": torch.tensor(self.truncated, dtype=torch.bool),
        }

        data["q_estimations"] = get_returns(data["rewards"], data["terminated"])

        return data

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