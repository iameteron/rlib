import torch


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
