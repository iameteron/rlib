import gymnasium as gym
import numpy as np

from ..common.buffer import RolloutBuffer
from ..common.logger import TensorBoardLogger
from ..common.losses import ppo_loss


def ppo(
    env: gym.Env,
    actor,
    critic,
    actor_optimizer,
    critic_optimizer,
    total_timesteps: int = 50_000,
    trajectories_n: int = 20,
    epochs_per_episode: int = 30,
    batch_size: int = 128,
):
    buffer = RolloutBuffer()
    logger = TensorBoardLogger(log_dir="./tb_logs/ppo_")

    steps_n = 0
    gd_step_n = 0

    while steps_n < total_timesteps:

        buffer.collect_rollouts(env, actor, trajectories_n=trajectories_n)

        data = buffer.get_data()
        rollout_size = data["observations"].shape[0]

        for _ in range(epochs_per_episode):
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
                
                # Logging
                gd_step_n += 1
                logger.log_scalars(loss, gd_step_n)

        steps_n += rollout_size

        # Logging
        trajectories = buffer.get_trajectories()
        logger.log_trajectories(trajectories)


    logger.close()