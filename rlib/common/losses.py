import torch
from torch.distributions import Normal


def reinforce_loss(data, returns_normalization=True):
    loss = {}

    returns = data["q_estimations"]
    log_probs = data["log_probs"]

    if returns_normalization:
        mean = returns.mean()
        std = returns.std()
        returns = (returns - mean) / (std + 1e-8)

    loss["actor"] = -(log_probs * returns).mean()

    return loss


def a2c_loss(data, critic):
    loss = {}

    observations = data["observations"]
    log_probs = data["log_probs"]
    targets = data["q_estimations"]

    values = critic(observations)
    advantages = targets[:-1].detach() - values[:-1]

    loss["actor"] = -(log_probs[:-1] * advantages.detach()).mean()
    loss["critic"] = (advantages**2).mean()

    return loss


def ppo_loss(
    data,
    actor,
    critic,
    epsilon: float = 0.2,
):
    loss = {}

    observations = data["observations"]
    old_log_probs = data["log_probs"]
    actions = data["actions"]
    targets = data["q_estimations"]

    _, new_log_probs = actor.get_action(observations, action=actions)

    ratio = torch.exp(new_log_probs - old_log_probs.detach())
    ratio_clipped = torch.clamp(ratio, 1 - epsilon, 1 + epsilon)

    values = critic(observations)

    advantages = targets.detach() - values

    actor_loss_1 = ratio * advantages.detach()
    actor_loss_2 = ratio_clipped * advantages.detach()

    loss["actor"] = -(torch.min(actor_loss_1, actor_loss_2)).mean()
    loss["critic"] = (advantages**2).mean()

    return loss


def ddpg_loss(data, actor, critic, actor_target, critic_target, gamma=0.99):
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
        targets = rewards + gamma * (~terminated) * critic_target(
            next_observations, actions_target
        )

    q_values = critic(observations, actions)
    loss["critic"] = ((q_values - targets) ** 2).mean()

    return loss


def td3_loss(
    data,
    actor,
    critic_1,
    critic_2,
    actor_target,
    critic_1_target,
    critic_2_target,
    gamma=0.99,
    policy_std=0.2,
    policy_clip=0.5,
):
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
        targets = rewards + gamma * (~terminated) * target_q_values

    q_values_1 = critic_1(observations, actions)
    q_values_2 = critic_2(observations, actions)

    loss["critic_1"] = ((q_values_1 - targets) ** 2).mean()
    loss["critic_2"] = ((q_values_2 - targets) ** 2).mean()

    return loss


def sac_loss(
    data,
    actor,
    critic_1,
    critic_2,
    critic_1_target,
    critic_2_target,
    gamma=0.99,
    alpha=1e-3,
):
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

        targets = rewards + gamma * (~terminated) * (
            target_q_values - alpha * next_log_probs
        )

    q_values_1 = critic_1(observations, actions)
    q_values_2 = critic_2(observations, actions)

    loss["critic_1"] = ((q_values_1 - targets) ** 2).mean()
    loss["critic_2"] = ((q_values_2 - targets) ** 2).mean()

    return loss
