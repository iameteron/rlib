# rlib

**rlib** is a lightweight reinforcement learning library that provides clean implementations of various RL algorithms, including model-free, imitation learning, and offline RL methods. It is designed for research and experimentation, with higly customizable components.

## Features

- **Model-Free RL:** A2C, PPO, SAC, TD3, DDPG, REINFORCE
- **Imitation Learning:** GAIL, GCL
- **Offline RL:** Decision Transformer (DT)
- **Common Utilities:** Replay and rollout buffers, logging, evaluation
- **Docker Support:** Easily deployable with Docker and `docker-compose`
- **Notebook Examples:** Pre-built Jupyter notebooks to demonstrate usage

## Installation

You can install the required dependencies with:

```bash
pip install -r requirements.txt
```

## Quick Start

Example usage for training a PPO agent:

```python
import gymnasium as gym
from torch.optim import Adam
from rlib.algorithms.model_free.ppo import ppo
from rlib.common.evaluation import validation
from rlib.common.policies import (
    MlpCritic,
    StochasticMlpPolicy,
)

# Create environment
env = gym.make("Pendulum-v1")

# Initialize PPO agent
actor = StochasticMlpPolicy(obs_dim, action_dim)
critic = MlpCritic(obs_dim)
actor_optimizer = Adam(actor.parameters(), lr=3e-4)
critic_optimizer = Adam(critic.parameters(), lr=3e-4)

# Train agent
ppo(env, actor, critic, actor_optimizer, critic_optimizer, total_timesteps=30_000)
```

## Project Structure

```
rlib/
│-- algorithms/       # RL algorithms
│   ├── model_free/   # A2C, PPO, SAC, etc.
│   ├── imitation/    # GAIL, GCL
│   ├── offline/      # Decision Transformer
│-- common/           # Utilities (buffer, env, logging, evaluation)
│-- examples/         # Jupyter notebooks and pretrained models
│-- tests/            # Unit tests
```

## Running with Docker

You can run the project inside a Docker container:

```bash
docker-compose up --build
```

## License

This project is licensed under the MIT License.