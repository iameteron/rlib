import os
from datetime import datetime

import torch
from torch.utils.tensorboard import SummaryWriter


class TensorBoardLogger:
    def __init__(self, log_dir="./tb_logs/"):
        experiment_name = "experiment_" + datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        path = os.path.abspath(log_dir + experiment_name)

        self.writer = SummaryWriter(path)
        self.env_steps = 0

    def log_scalar(self, tag, value, step):
        self.writer.add_scalar(tag, value, step)

    def log_scalars(self, metrics, step):
        for tag, value in metrics.items():
            self.log_scalar(tag, value, step)

    def log_trajectories(self, trajectories):
        for trajectory in trajectories:
            reward = torch.sum(trajectory["rewards"])
            length = trajectory["rewards"].shape[0]

            self.env_steps += length

            self.log_scalars(
                {
                    "traj_reward": reward,
                    "traj_len": length,
                },
                self.env_steps
            )

    def close(self):
        self.writer.close()