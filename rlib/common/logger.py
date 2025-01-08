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