from copy import deepcopy

import gymnasium as gym
from torch.optim import Adam

from ..common.buffer import ReplayBuffer
from ..common.logger import TensorBoardLogger
from ..common.losses import td3_loss
from ..common.policies import DeterministicMlpPolicy, MlpQCritic
from ..common.utils import smooth_update


def td3(
    env: gym.Env,
    actor: DeterministicMlpPolicy,
    critic_1: MlpQCritic,
    critic_2: MlpQCritic,
    actor_optimizer: Adam,
    critic_1_optimizer: Adam,
    critic_2_optimizer: Adam,
    training_starts: int = 1000,
    total_timesteps: int = 50_000,
    batch_size: int = 128,
    target_update_frequency: int = 2,
):
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    buffer = ReplayBuffer(obs_dim, action_dim)

    logger = TensorBoardLogger(log_dir="./tb_logs/td3_")

    actor_target = deepcopy(actor)
    critic_1_target = deepcopy(critic_1)
    critic_2_target = deepcopy(critic_2)

    steps_n = 0
    while steps_n < total_timesteps:
        buffer.collect_transition(env, actor)

        if buffer.size < training_starts:
            continue

        batch = buffer.get_batch(batch_size)

        loss = td3_loss(
            batch,
            actor,
            critic_1,
            critic_2,
            actor_target,
            critic_1_target,
            critic_2_target,
        )

        loss["actor"].backward()
        actor_optimizer.step()
        actor_optimizer.zero_grad()

        loss["critic_1"].backward()
        critic_1_optimizer.step()
        critic_1_optimizer.zero_grad()

        loss["critic_2"].backward()
        critic_2_optimizer.step()
        critic_2_optimizer.zero_grad()

        if steps_n % target_update_frequency == 0:
            # actor_target = smooth_update(actor, actor_target)
            critic_1_target = smooth_update(critic_1, critic_1_target)
            critic_2_target = smooth_update(critic_2, critic_2_target)

        # Logging
        steps_n += 1
        logger.log_scalars(loss, steps_n)

        if buffer.done:
            trajectory = buffer.get_last_trajectory()
            logger.log_trajectories(trajectory)