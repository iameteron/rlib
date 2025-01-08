import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import animation
from torch import nn
from torch.distributions import Categorical, Normal

env = gym.make("CartPole-v1", render_mode="rgb_array")

class DiscreteStochasticMlpPolicy(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=64):
        super().__init__()
        self.obs_dim = input_size
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.act = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input):
        input = torch.FloatTensor(input.reshape(-1, self.obs_dim))
        output = self.act(self.fc1(input))
        output = self.act(self.fc2(output))
        logits = self.fc3(output)
        probs = self.softmax(logits)

        return probs

    def predict(self, input, deterministic=False):
        probs = self.forward(input)
        dist = Categorical(probs)

        if deterministic:
            action = torch.argmax(probs, dim=1)
        else:
            action = dist.sample()

        log_prob_action = dist.log_prob(action)

        return action[0].item(), log_prob_action

class MlpCritic(nn.Module):
    def __init__(self, input_size, hidden_size=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, input):
        return self.net(input)

class Logger:
    def __init__(self, log_frequency=1000):
        self.log_frequency = log_frequency
        self.next_iteration = self.log_frequency

    def log(self, steps_n, data):
        if steps_n >= self.next_iteration:
            self.next_iteration += self.log_frequency

            dones = data["terminated"] + data["truncated"]
            if sum(dones) == 0:
                dones[-1] = True
            dones_indeces = dones.nonzero()
            last_done_index = dones_indeces[-1][0].item()
            trajectory_n = sum(dones)
            mean_trajectory_rewards = (
                sum(data["rewards"][:last_done_index]) / trajectory_n
            )
            mean_trajectory_length = last_done_index / trajectory_n
            print(f"steps_n: {steps_n}")
            print(f"mean_trajectory_rewards: {mean_trajectory_rewards}")
            print(f"mean_trajectory_length: {mean_trajectory_length}")

def get_returns(rewards, terminated, gamma=1):
    rewards_n = len(rewards)
    returns = torch.zeros_like(rewards)
    returns[-1] = rewards[-1]

    # TODO add truncated, values
    for t in reversed(range(rewards_n - 1)):
        returns[t] = rewards[t] + (~terminated[t]) * gamma * returns[t + 1]

    return returns


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
            "actions": torch.tensor(self.actions, dtype=torch.long),
            "log_probs": torch.stack(self.log_probs).squeeze(),
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


def ppo_loss(
    data,
    actor,
    critic,
    epsilon: float = 0.1,
    advantage_normalization: bool = True,
    rewards_normalization: bool = True,
):
    loss = {}

    old_log_probs = data["log_probs"][:-1]

    observations = data["observations"]
    _, new_log_probs = actor.predict(observations[:-1])

    ratio = torch.exp(new_log_probs - old_log_probs.detach())
    ratio_clipped = torch.clamp(ratio, 1 - epsilon, 1 + epsilon)

    values = critic(observations).squeeze()
    # targets, advantages = advantage_fn(rewards, values, terminated, gamma=0.99)

    targets = data["q_estimations"]
    advantages = targets[:-1].detach() - values[:-1]

    actor_loss_1 = ratio * advantages.detach()
    actor_loss_2 = ratio_clipped * advantages.detach()

    loss["actor"] = -(torch.min(actor_loss_1, actor_loss_2)).mean()
    loss["critic"] = (advantages**2).mean()

    return loss

def ppo(
    env: gym.Env,
    actor: DiscreteStochasticMlpPolicy,
    critic: MlpCritic,
    actor_optimizer,
    critic_optimizer,
    total_timesteps: int = 100_000,
    trajectories_n: int = 20,
    epoch_n: int = 10,
    batch_size: int = 256,
):
    buffer = RolloutBuffer()
    logger = Logger()

    steps_n = 0
    while steps_n < total_timesteps:

        buffer.collect_rollouts(env, actor, trajectories_n=trajectories_n)

        data = buffer.get_data()

        rollout_size = data["observations"].shape[0]
        steps_n += rollout_size

        for _ in range(epoch_n):
            indices = np.random.permutation(range(rollout_size))
            for start in range(0, rollout_size, batch_size):
                end = start + batch_size
                batch_indices = indices[start:end]

                if batch_indices.size <= 1:
                    break

                batch = {key: value[batch_indices] for key, value in data.items()}

                loss = ppo_loss(batch, actor, critic)

                loss["actor"].backward()
                actor_optimizer.step()
                actor_optimizer.zero_grad()

                loss["critic"].backward()
                critic_optimizer.step()
                critic_optimizer.zero_grad()
                
        logger.log(steps_n, data)

discrete = True

input_size = env.observation_space.shape[0]

if discrete:
    output_size = env.action_space.n
else:
    output_size = 2 * env.action_space.shape[0]

print(input_size, output_size)


actor = DiscreteStochasticMlpPolicy(input_size, output_size)
critic = MlpCritic(input_size)

actor_optimizer = torch.optim.Adam(actor.parameters(), lr=1e-3)
critic_optimizer = torch.optim.Adam(critic.parameters(), lr=1e-4)

ppo(env, actor, critic, actor_optimizer, critic_optimizer, total_timesteps=100_000)
