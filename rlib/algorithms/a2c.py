import gymnasium as gym

from ..common.buffer import RolloutBuffer
from ..common.logger import Logger
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
    logger = Logger()

    steps_n = 0
    while steps_n < total_timesteps:

        buffer.collect_rollouts(env, actor, trajectories_n=trajectories_n)

        data = buffer.get_data()

        rollout_size = data["observations"].shape[0]
        steps_n += rollout_size

        loss = a2c_loss(data, critic)

        logger.log(steps_n, data)

        loss["actor"].backward()
        actor_optimizer.step()
        actor_optimizer.zero_grad()

        loss["critic"].backward()
        critic_optimizer.step()
        critic_optimizer.zero_grad()