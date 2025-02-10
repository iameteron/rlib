import torch
from torch import nn


def get_returns(rewards: torch.Tensor, terminated: torch.Tensor, gamma: float = 0.9): 
    """
    Args:
        rewards (torch.Tensor): (N, 1)
        terminated (torch.Tensor): (N, 1)
        gamma (float)

    Returns:
        returns: (torch.Tensor): (N, 1)
    """
    rewards_n = len(rewards)
    returns = torch.zeros_like(rewards)
    returns[-1] = rewards[-1]

    # TODO add truncated, values
    for t in reversed(range(rewards_n - 1)):
        returns[t] = rewards[t] + (1 - terminated[t]) * gamma * returns[t + 1]

    return returns


def get_1_step_td_advantage(
    rewards: torch.Tensor,
    values: torch.Tensor,
    terminated: torch.Tensor,
    gamma: float = 0.9,
):
    """
    Args:
        rewards (torch.Tensor): (N, 1)
        values (torch.Tensor): (N, 1)
        terminated (torch.Tensor): (N, 1)
        gamma (float)

    Returns:
        targets: (torch.Tensor): (N - 1, 1)
        advantages: (torch.Tensor): (N - 1, 1)
    """
    # TODO add truncated, values
    targets = rewards[:-1] + gamma * (1 - terminated[:-1]) * values[1:]
    advantages = targets - values[:-1]
    return targets, advantages


def get_max_step_advantage(
    rewards: torch.Tensor,
    values: torch.Tensor,
    terminated: torch.Tensor,
    gamma: float = 0.9,
):
    """
    Args:
        rewards (torch.Tensor): (N, 1)
        values (torch.Tensor): (N, 1)
        terminated (torch.Tensor): (N, 1)
        gamma (float)

    Returns:
        targets: (torch.Tensor): (N - 1, 1)
        advantages: (torch.Tensor): (N - 1, 1)
    """
    # TODO add truncated, values
    returns = get_returns(rewards, terminated, gamma)
    targets = returns[:-1]
    advantages = targets - values[:-1]
    return targets, advantages


def get_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    terminated: torch.Tensor,
    gamma: float = 0.9,
    lamb: float = 0.95,
):
    """
    Args:
        rewards (torch.Tensor): (N, 1)
        values (torch.Tensor): (N, 1)
        terminated (torch.Tensor): (N, 1)
        gamma (float)
        lamb (float)

    Returns:
        targets: (torch.Tensor): (N - 1, 1)
        advantages: (torch.Tensor): (N - 1, 1)
    """
    # TODO add truncated, values
    advantages = torch.zeros_like(rewards[:-1])  # (N - 1, 1)
    advantages_n = len(advantages)

    targets, td1_advantages = get_1_step_td_advantage(
        rewards, values, terminated, gamma
    )
    advantages[-1] = td1_advantages[-1]

    for t in reversed(range(advantages_n - 1)):
        advantages[t] = (
            td1_advantages[t] + gamma * lamb + (1 - terminated[t]) * advantages[t + 1]
        )

    return targets, advantages


def smooth_update(model: nn.Module, target_model: nn.Module, tau: float = 0.99):
    for param, target_param in zip(model.parameters(), target_model.parameters()):
        new_terget_param = tau * target_param + (1 - tau) * param
        target_param.data.copy_(new_terget_param)

    return target_model
