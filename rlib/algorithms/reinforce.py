import gymnasium as gym

from ..common.buffer import RolloutBuffer
from ..common.logger import TensorBoardLogger
from ..common.losses import reinforce_loss


def reinforce(
    env: gym.Env, policy, optimizer, total_timesteps: int = 50_000
):
    buffer = RolloutBuffer()
    logger = TensorBoardLogger(log_dir="./tb_logs/reinforce_")

    steps_n = 0
    episode_n = 0

    while steps_n < total_timesteps:

        buffer.collect_rollouts(env, policy, trajectories_n=1)
        data = buffer.get_data()

        loss = reinforce_loss(data)

        loss["actor"].backward()
        optimizer.step()
        optimizer.zero_grad()

        # Logging
        rollout_size = data["observations"].shape[0]
        steps_n += rollout_size
        episode_n += 1

        trajectories = buffer.get_trajectories(data)
        logger.log_trajectories(trajectories)
        logger.log_scalars(loss, episode_n)
        
    logger.close()