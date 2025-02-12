import gymnasium as gym
import numpy as np
import torch
from torch.optim import Adam

from ...common.buffer import RolloutBuffer
from ...common.logger import TensorBoardLogger
from ...common.losses import gcl_loss, ppo_loss
from ...common.policies import DeterministicMlpPolicy, MlpCritic, RewardNet
from ...common.utils import get_gae


def gcl(
    env: gym.Env,
    expert_data: dict[str, torch.Tensor],
    learning_actor: DeterministicMlpPolicy,
    critic: MlpCritic,
    reward_net: RewardNet,
    actor_optimizer: Adam,
    critic_optimizer: Adam,
    reward_optimizer: Adam,
    total_episodes: int = 200,
    trajectories_n: int = 20,
    epochs_per_episode: int = 30,
    batch_size: int = 128,
    gamma: float = 0.9,
    lamb: float = 0.95,
    epsilon: float = 0.2,
):
    rollout_buffer = RolloutBuffer()
    logger = TensorBoardLogger(log_dir="./tb_logs/gcl_")

    steps_n = 0
    gd_step_n = 0

    for episode_n in range(total_episodes):
        rollout_buffer.collect_rollouts(
            env, learning_actor, trajectories_n=trajectories_n
        )

        data = rollout_buffer.get_data()
        rollout_size = data["observations"].shape[0]

        # Reward update
        indices = np.random.permutation(range(rollout_size))
        expert_batch = {key: value[indices] for key, value in expert_data.items()}

        reward_loss = gcl_loss(reward_net, data, expert_batch)

        reward_optimizer.zero_grad()
        reward_loss["reward"].backward()
        reward_optimizer.step()

        logger.log_scalars(reward_loss, episode_n)

        # PPO update
        data["rewards"] = reward_net(
            data["observations"], data["actions"]
        )

        values = critic(data["observations"])

        targets, _ = get_gae(
            data["rewards"], values, data["dones"], gamma, lamb
        )

        data["advantages"] = targets.detach() - values

        observations = data["observations"]
        rewards = data["rewards"]
        dones = data["dones"]

        rollout_size = observations.shape[0]

        values = critic(observations)
        targets, _ = get_gae(rewards, values, dones, gamma, lamb)

        data["q_estimations"] = targets

        for _ in range(epochs_per_episode):
            indices = np.random.permutation(range(rollout_size))

            for start in range(0, rollout_size, batch_size):
                end = start + batch_size
                batch_indices = indices[start:end]

                if batch_indices.size <= 1:
                    break

                batch = {key: value[batch_indices] for key, value in data.items()}

                observations = batch["observations"]
                targets = batch["q_estimations"]

                values = critic(observations)
                batch["advantages"] = targets.detach() - values

                loss = ppo_loss(batch, learning_actor, epsilon)

                actor_optimizer.zero_grad()
                loss["actor"].backward()
                actor_optimizer.step()

                critic_optimizer.zero_grad()
                loss["critic"].backward()
                critic_optimizer.step()

                # Logging
                gd_step_n += 1
                logger.log_scalars(loss, gd_step_n)

        steps_n += rollout_size

        # Logging
        trajectories = rollout_buffer.get_trajectories(data)
        logger.log_trajectories(trajectories)

    logger.close()