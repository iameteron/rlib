import gymnasium as gym

from ..common.buffer import RolloutBuffer
from ..common.logger import TensorBoardLogger
from ..common.losses import a2c_loss


def a2c(
    env: gym.Env,
    actor,
    critic,
    actor_optimizer,
    critic_optimizer,
    total_timesteps: int = 50_000,
    trajectories_n: int = 10,
):
    buffer = RolloutBuffer()
    logger = TensorBoardLogger(log_dir="./tb_logs/a2c_")

    steps_n = 0
    episode_n = 0
    while steps_n < total_timesteps:

        buffer.collect_rollouts(env, actor, trajectories_n=trajectories_n)

        data = buffer.get_data()

        loss = a2c_loss(data, critic)

        loss["actor"].backward()
        actor_optimizer.step()
        actor_optimizer.zero_grad()

        loss["critic"].backward()
        critic_optimizer.step()
        critic_optimizer.zero_grad()

        # Logging
        rollout_size = data["observations"].shape[0]
        steps_n += rollout_size
        episode_n += 1

        trajectories = buffer.get_trajectories()
        logger.log_trajectories(trajectories)
        logger.log_scalars(loss, episode_n)

    logger.close()