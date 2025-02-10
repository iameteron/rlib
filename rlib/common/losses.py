from typing import Dict

import torch
from torch.distributions import Normal

from .policies import (
    DeterministicMlpPolicy,
    Discriminator,
    MlpQCritic,
    RewardNet,
    StochasticMlpPolicy,
)


def reinforce_loss(
    data: Dict[str, torch.Tensor], returns_normalization=True
) -> Dict[str, torch.Tensor]:
    loss = {}

    returns = data["q_estimations"]
    log_probs = data["log_probs"]

    if returns_normalization:
        mean = returns.mean()
        std = returns.std()
        returns = (returns - mean) / (std + 1e-8)

    print((log_probs * returns).shape)

    loss["actor"] = -(log_probs * returns).mean()

    return loss


def a2c_loss(data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    loss = {}

    log_probs = data["log_probs"]
    advantages = data["advantages"]

    loss["actor"] = -(log_probs[:-1] * advantages.detach()).mean()
    loss["critic"] = (advantages**2).mean()

    return loss


def ppo_loss(
    data: Dict[str, torch.Tensor],
    actor: StochasticMlpPolicy,
    epsilon: float = 0.2,
    entropy_coef: float = 0.01,  # TODO: add entropy coeff
) -> Dict[str, torch.Tensor]:
    loss = {}

    observations = data["observations"]
    old_log_probs = data["log_probs"]
    actions = data["actions"]
    advantages = data["advantages"]

    print(
        observations.shape,
        old_log_probs.shape,
        actions.shape,
        advantages.shape,
    )

    _, new_log_probs = actor.get_action(observations, action=actions)

    ratio = torch.exp(new_log_probs - old_log_probs.detach())
    ratio_clipped = torch.clamp(ratio, 1 - epsilon, 1 + epsilon)

    actor_loss_1 = ratio * advantages.detach()
    actor_loss_2 = ratio_clipped * advantages.detach()

    loss["actor"] = -(torch.min(actor_loss_1, actor_loss_2)).mean()
    loss["critic"] = (advantages**2).mean()

    return loss


def ddpg_loss(
    data: Dict[str, torch.Tensor],
    actor: DeterministicMlpPolicy,
    critic: MlpQCritic,
    actor_target: DeterministicMlpPolicy,
    critic_target: MlpQCritic,
    gamma: float = 0.99,
) -> Dict[str, torch.Tensor]:
    loss = {}

    observations = data["observations"]
    next_observations = data["next_observations"]
    actions = data["actions"]
    rewards = data["rewards"]
    terminated = data["terminated"]

    actor_outputs = actor(observations)
    loss["actor"] = -critic(observations, actor_outputs).mean()

    with torch.no_grad():
        actions_target = actor_target(next_observations)
        targets = rewards + gamma * (1 - terminated) * critic_target(
            next_observations, actions_target
        )

    q_values = critic(observations, actions)
    loss["critic"] = ((q_values - targets) ** 2).mean()

    return loss


def td3_loss(
    data: Dict[str, torch.Tensor],
    actor: DeterministicMlpPolicy,
    critic_1: MlpQCritic,
    critic_2: MlpQCritic,
    actor_target: DeterministicMlpPolicy,
    critic_1_target: MlpQCritic,
    critic_2_target: MlpQCritic,
    gamma=0.99,
    policy_std=0.2,
    policy_clip=0.5,
) -> Dict[str, torch.Tensor]:
    loss = {}

    observations = data["observations"]
    next_observations = data["next_observations"]
    actions = data["actions"]
    rewards = data["rewards"]
    terminated = data["terminated"]

    actor_outputs = actor(observations)

    q_values = torch.min(
        critic_1(observations, actor_outputs),
        critic_2(observations, actor_outputs),
    )
    loss["actor"] = -q_values.mean()

    with torch.no_grad():
        actions_target = actor_target(next_observations)

        loc = torch.zeros_like(actions_target)
        scale = policy_std * torch.ones_like(actions_target)
        dist = Normal(loc, scale)
        epsilon = torch.clamp(dist.sample(), -policy_clip, policy_clip)

        actions_target = actions_target + epsilon

        target_q_values = torch.min(
            critic_1_target(next_observations, actions_target),
            critic_2_target(next_observations, actions_target),
        )
        targets = rewards + gamma * (1 - terminated) * target_q_values

    q_values_1 = critic_1(observations, actions)
    q_values_2 = critic_2(observations, actions)

    loss["critic_1"] = ((q_values_1 - targets) ** 2).mean()
    loss["critic_2"] = ((q_values_2 - targets) ** 2).mean()

    return loss


def sac_loss(
    data: Dict[str, torch.Tensor],
    actor: StochasticMlpPolicy,
    critic_1: MlpQCritic,
    critic_2: MlpQCritic,
    critic_1_target: MlpQCritic,
    critic_2_target: MlpQCritic,
    gamma: float = 0.99,
    alpha: float = 1e-3,
) -> Dict[str, torch.Tensor]:
    loss = {}

    observations = data["observations"]
    next_observations = data["next_observations"]
    actions = data["actions"]
    rewards = data["rewards"]
    terminated = data["terminated"]

    actor_actions, actor_log_probs = actor.get_action(
        observations, sample_gradients=True
    )

    q_actor_values = torch.min(
        critic_1(observations, actor_actions),
        critic_2(observations, actor_actions),
    )

    loss["actor"] = -(q_actor_values - alpha * actor_log_probs).mean()

    with torch.no_grad():
        next_actions, next_log_probs = actor.get_action(next_observations)

        target_q_values = torch.min(
            critic_1_target(next_observations, next_actions),
            critic_2_target(next_observations, next_actions),
        )

        targets = rewards + gamma * (1 - terminated) * (
            target_q_values - alpha * next_log_probs
        )

    q_values_1 = critic_1(observations, actions)
    q_values_2 = critic_2(observations, actions)

    loss["critic_1"] = ((q_values_1 - targets) ** 2).mean()
    loss["critic_2"] = ((q_values_2 - targets) ** 2).mean()

    return loss


def gcl_loss(
    reward_net: RewardNet,
    learning_data: Dict[str, torch.Tensor],
    expert_data: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    loss = {}

    learning_observations = learning_data["observations"]
    learning_actions = learning_data["actions"]

    expert_observations = expert_data["observations"]
    expert_actions = expert_data["actions"]

    loss["learning"] = reward_net(learning_observations, learning_actions).mean()
    loss["expert"] = reward_net(expert_observations, expert_actions).mean()
    loss["reward"] = -(loss["expert"] - loss["learning"])

    return loss


def gail_loss(
    discriminator: Discriminator,
    learning_trajectories: Dict[str, torch.Tensor],
    expert_trajectories: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    loss = {}

    learning_observations = learning_trajectories["observations"]
    learning_actions = learning_trajectories["actions"]
    learning_preds = discriminator(learning_observations, learning_actions)

    expert_observations = expert_trajectories["observations"]
    expert_actions = expert_trajectories["actions"]
    expert_preds = discriminator(expert_observations, expert_actions)

    loss["learning"] = -torch.log(learning_preds + 1e-10).mean()
    loss["expert"] = -torch.log(1 - expert_preds + 1e-10).mean()
    loss["discriminator"] = loss["expert"] + loss["learning"]

    return loss
