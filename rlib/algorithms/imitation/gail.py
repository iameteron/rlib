import gymnasium as gym
import torch
from torch.optim import Adam

from ...common.buffer import RolloutBuffer
from ...common.logger import TensorBoardLogger
from ...common.losses import gail_loss, ppo_loss
from ...common.policies import (
    DeterministicMlpPolicy,
    Discriminator,
    DiscriminatorReward,
    MlpCritic,
)
from ...common.utils import get_returns


def gail(
    env: gym.Env,
    expert_data: dict[str, torch.Tensor],
    learning_actor: DeterministicMlpPolicy,
    critic: MlpCritic,
    discriminator: Discriminator,
    actor_optimizer: Adam,
    critic_optimizer: Adam,
    discriminator_optimizer: Adam,
    total_episodes: int = 100,
    trajectories_n: int = 10,
):
    rollout_buffer = RolloutBuffer()

    reward_net = DiscriminatorReward(discriminator)

    logger = TensorBoardLogger(log_dir="./tb_logs/gail_")

    for episode_n in range(total_episodes):
        rollout_buffer.collect_rollouts(
            env, learning_actor, trajectories_n=trajectories_n
        )
        learning_data = rollout_buffer.get_data()

        disc_loss = gail_loss(discriminator, learning_data, expert_data)

        disc_loss["discriminator"].backward()
        discriminator_optimizer.step()
        discriminator_optimizer.zero_grad()

        learning_data["rewards"] = reward_net.forward(
            learning_data["observations"], learning_data["actions"]
        )

        learning_data["q_estimations"] = get_returns(
            learning_data["rewards"], learning_data["dones"]
        )

        actor_loss = ppo_loss(
            learning_data,
            learning_actor,
            critic,
        )

        actor_optimizer.zero_grad()
        actor_loss["actor"].backward()
        actor_optimizer.step()

        critic_optimizer.zero_grad()
        actor_loss["critic"].backward()
        critic_optimizer.step()

        # Logging
        logger.log_scalars(disc_loss, episode_n)
        logger.log_scalars(actor_loss, episode_n)