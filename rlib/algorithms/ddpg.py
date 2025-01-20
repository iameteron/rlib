from copy import deepcopy

import gymnasium as gym
from torch.optim import Adam

from ..common.buffer import ReplayBuffer
from ..common.logger import TensorBoardLogger
from ..common.losses import ddpg_loss
from ..common.policies import DeterministicMlpPolicy, MlpQCritic
from ..common.utils import smooth_update


def ddpg(
    env: gym.Env,
    actor: DeterministicMlpPolicy,
    critic: MlpQCritic,
    actor_optimizer: Adam,
    critic_optimizer: Adam,
    training_starts: int = 1000,
    total_timesteps: int = 50_000,
    batch_size: int = 512,
):
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    buffer = ReplayBuffer(obs_dim, action_dim)

    logger = TensorBoardLogger(log_dir="./tb_logs/ddpg_")

    actor_target = deepcopy(actor)
    critic_target = deepcopy(critic)

    steps_n = 0
    while steps_n < total_timesteps:
        buffer.collect_transition(env, actor)

        if buffer.size < training_starts:
            continue

        batch = buffer.get_batch(batch_size)

        loss = ddpg_loss(batch, actor, critic, actor_target, critic_target)

        loss["actor"].backward()
        actor_optimizer.step()
        actor_optimizer.zero_grad()

        loss["critic"].backward()
        critic_optimizer.step()
        critic_optimizer.zero_grad()

        actor_target = smooth_update(actor, actor_target)
        critic_target = smooth_update(critic, critic_target)

        # Logging
        steps_n += 1
        logger.log_scalars(loss, steps_n)

        if buffer.done:
            trajectory = buffer.get_last_trajectory()
            logger.log_trajectories(trajectory)
