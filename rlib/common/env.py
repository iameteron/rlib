import gymnasium as gym
import torch


class CustomNNRewardWrapper(gym.Wrapper):
    def __init__(self, env, reward_fn):
        super().__init__(env)
        self.reward_fn = reward_fn

    def step(self, action):
        """
        Args:
            action: (np.ndarray): (action_dim,)
        """
        observation, reward, terminated, truncated, info = self.env.step(action)

        observation_t = torch.tensor(observation.reshape(0, -1), dtype=torch.float32)
        action_t = torch.tensor(action.reshape(0, -1), dtype=torch.float32)

        reward_t = self.reward_fn(observation_t, action_t)
        reward = reward_t.item()

        return observation, reward, terminated, truncated, info