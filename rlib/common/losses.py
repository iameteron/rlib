import torch


def reinforce_loss(data, returns_normalization=True):
    returns = data["q_estimations"]
    log_probs = data["log_probs"]

    if returns_normalization:
        mean = returns.mean()
        std = returns.std()
        returns = (returns - mean) / (std + 1e-8)

    loss = -(log_probs * returns).mean()

    return loss


def a2c_loss(data, critic):
    loss = {}

    log_probs = data["log_probs"][:-1]

    observations = data["observations"]
    values = critic(observations).squeeze()
    targets = data["q_estimations"]
    advantages = targets[:-1].detach() - values[:-1]

    loss["actor"] = -(log_probs * advantages.detach()).mean()
    loss["critic"] = (advantages**2).mean()

    return loss


def ppo_loss(
    data,
    actor,
    critic,
    epsilon: float = 0.2,
    advantage_normalization: bool = False,
    rewards_normalization: bool = False,
):
    loss = {}

    old_log_probs = data["log_probs"]
    observations = data["observations"]
    actions = data["actions"]

    _, new_log_probs = actor.get_action(observations, action=actions)

    ratio = torch.exp(new_log_probs - old_log_probs.detach())
    ratio_clipped = torch.clamp(ratio, 1 - epsilon, 1 + epsilon)

    values = critic(observations).reshape(ratio.shape)

    targets = data["q_estimations"].reshape(ratio.shape)
    advantages = targets.detach() - values

    if advantage_normalization:
        mean = advantages.mean()
        std = advantages.std()
        advantages = (advantages - mean) / (std + 1e-8)

    actor_loss_1 = ratio * advantages.detach()
    actor_loss_2 = ratio_clipped * advantages.detach()

    loss["actor"] = -(torch.min(actor_loss_1, actor_loss_2)).mean()
    loss["critic"] = (advantages**2).mean()

    return loss