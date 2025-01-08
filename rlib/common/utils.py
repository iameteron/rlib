import torch


def get_returns(rewards, terminated, gamma=0.9):
    rewards_n = len(rewards)
    returns = torch.zeros_like(rewards)
    returns[-1] = rewards[-1]

    # TODO add truncated, values
    for t in reversed(range(rewards_n - 1)):
        returns[t] = rewards[t] + (~terminated[t]) * gamma * returns[t + 1]

    return returns


def get_1_step_td_advantage(rewards, values, terminated, gamma=0.9):
    targets = rewards[:-1] + gamma * (~terminated[:-1]) * values[1:]
    advantages = targets - values[:-1]
    return targets, advantages


def get_max_step_advantage(rewards, values, terminated, gamma=0.9):
    returns = get_returns(rewards, terminated, gamma)
    targets = returns[:-1]
    advantages = targets - values[:-1]
    return targets, advantages