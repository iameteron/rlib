import gymnasium as gym
import torch
from torch.optim import Adam

from ...common.buffer import RolloutBuffer
from ...common.logger import TensorBoardLogger
from ...common.losses import gcl_loss, ppo_loss
from ...common.policies import DeterministicMlpPolicy, MlpCritic, RewardNet
from ...common.utils import get_returns


def gcl(
    env: gym.Env,
    expert_data: dict[str, torch.Tensor],
    learning_actor: DeterministicMlpPolicy,
    critic: MlpCritic,
    reward_net: RewardNet,
    actor_optimizer: Adam,
    critic_optimizer: Adam,
    reward_optimizer: Adam,
    total_episodes: int = 1000,
    trajectories_n: int = 10,
):
    rollout_buffer = RolloutBuffer()

    logger = TensorBoardLogger(log_dir="./tb_logs/gcl_")

    for episode_n in range(total_episodes):
        rollout_buffer.collect_rollouts(
            env, learning_actor, trajectories_n=trajectories_n
        )
        learning_data = rollout_buffer.get_data()

        reward_loss = gcl_loss(reward_net, learning_data, expert_data)

        reward_optimizer.zero_grad()
        reward_loss["reward"].backward()
        reward_optimizer.step()

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

        actor_loss["actor"].backward()
        actor_optimizer.step()
        actor_optimizer.zero_grad()

        actor_loss["critic"].backward()
        critic_optimizer.step()
        critic_optimizer.zero_grad()

        # Logging
        logger.log_scalars(reward_loss, episode_n)
        logger.log_scalars(actor_loss, episode_n)