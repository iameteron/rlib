import gymnasium as gym
from torch.optim import Adam

from ...common.buffer import RolloutBuffer
from ...common.logger import TensorBoardLogger
from ...common.losses import a2c_loss
from ...common.policies import MlpCritic, StochasticMlpPolicy
from ...common.utils import get_1_step_td_advantage


def a2c(
    env: gym.Env,
    actor: StochasticMlpPolicy,
    critic: MlpCritic,
    actor_optimizer: Adam,
    critic_optimizer: Adam,
    total_timesteps: int = 50_000,
    trajectories_n: int = 10,
    gamma: float = 0.99,
):
    buffer = RolloutBuffer()
    logger = TensorBoardLogger(log_dir="./tb_logs/a2c_")

    steps_n = 0
    episode_n = 0
    while steps_n < total_timesteps:
        buffer.collect_rollouts(env, actor, trajectories_n=trajectories_n)

        data = buffer.get_data()

        values = critic(data["observations"])
        _, data["advantages"] = get_1_step_td_advantage(
            data["rewards"], values, data["terminated"], gamma
        )

        loss = a2c_loss(data)

        actor_optimizer.zero_grad()
        loss["actor"].backward()
        actor_optimizer.step()

        critic_optimizer.zero_grad()
        loss["critic"].backward()
        critic_optimizer.step()

        # Logging
        rollout_size = data["observations"].shape[0]
        steps_n += rollout_size
        episode_n += 1

        trajectories = buffer.get_trajectories(data)
        logger.log_trajectories(trajectories)
        logger.log_scalars(loss, episode_n)

    logger.close()
