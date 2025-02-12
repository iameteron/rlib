import gymnasium as gym
from torch.optim import Adam

from ...common.buffer import RolloutBuffer
from ...common.logger import TensorBoardLogger
from ...common.losses import reinforce_loss
from ...common.policies import StochasticMlpPolicy
from ...common.utils import get_returns


def reinforce(
    env: gym.Env,
    policy: StochasticMlpPolicy,
    optimizer: Adam,
    total_timesteps: int = 50_000,
    gamma: float = 0.9,
):
    buffer = RolloutBuffer()
    logger = TensorBoardLogger(log_dir="./tb_logs/reinforce_")

    steps_n = 0
    episode_n = 0

    while steps_n < total_timesteps:
        buffer.collect_rollouts(env, policy, trajectories_n=1)
        data = buffer.get_data()

        rewards = data["rewards"]
        dones = data["dones"]

        data["q_estimations"] = get_returns(rewards, dones, gamma)

        loss = reinforce_loss(data)

        optimizer.zero_grad()
        loss["actor"].backward()
        optimizer.step()

        # Logging
        rollout_size = data["observations"].shape[0]
        steps_n += rollout_size
        episode_n += 1

        trajectories = buffer.get_trajectories(data)
        logger.log_trajectories(trajectories)
        logger.log_scalars(loss, episode_n)

    logger.close()
