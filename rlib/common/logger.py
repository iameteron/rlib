from datetime import datetime

import numpy as np
from torch.utils.tensorboard import SummaryWriter


class TensorBoardLogger:
    def __init__(self, log_dir="./tb_logs/"):
        experiment_name = "experiment_" + datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        path = log_dir + experiment_name

        self.writer = SummaryWriter(path)
        self.env_steps = 0

    def log_scalar(self, tag, value, step):
        self.writer.add_scalar(tag, value, step)

    def log_scalars(self, metrics, step):
        for tag, value in metrics.items():
            self.log_scalar(tag, value, step)

    def log_trajectories(self, trajectories):
        for trajectory in trajectories:
            reward = np.sum(trajectory["rewards"])
            length = len(trajectory["rewards"])

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


class Logger:
    def __init__(self, log_frequency=1000):
        self.log_frequency = log_frequency
        self.next_iteration = self.log_frequency

    def log(self, steps_n, data):
        if steps_n >= self.next_iteration:
            self.next_iteration += self.log_frequency

            dones = data["terminated"] + data["truncated"]
            if sum(dones) == 0:
                dones[-1] = True
            dones_indeces = dones.nonzero()
            last_done_index = dones_indeces[-1][0].item()
            trajectory_n = sum(dones)
            mean_trajectory_rewards = (
                sum(data["rewards"][:last_done_index]) / trajectory_n
            )
            mean_trajectory_length = last_done_index / trajectory_n
            print(f"steps_n: {steps_n}")
            print(f"mean_trajectory_rewards: {mean_trajectory_rewards.item()}")
            print(f"mean_trajectory_length: {mean_trajectory_length.item()}")
