import gymnasium as gym

from ..common.buffer import RolloutBuffer
from ..common.logger import Logger
from ..common.losses import reinforce_loss


def reinforce(
    env: gym.Env, policy, optimizer, total_timesteps: int = 50_000
):
    buffer = RolloutBuffer()
    logger = Logger()

    steps_n = 0
    while steps_n < total_timesteps:

        buffer.collect_rollouts(env, policy, trajectories_n=1)
        data = buffer.get_data()

        rollout_size = data["observations"].shape[0]
        steps_n += rollout_size

        loss = reinforce_loss(data)

        steps_n += data["observations"].shape[0]
        logger.log(steps_n, data)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()