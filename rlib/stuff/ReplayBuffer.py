import random


class ReplayBuffer:
    def __init__(self):
        self.max_size = 10_000
        self.transitions = []

    def get_tranistion(self, transition):
        if len(self.transitions) == self.max_size:
            index = random.randint(0, self.max_size - 1)
            self.transitions[index] = transition
        else:
            self.transitions.append(transition)

    def sample_expirience(self, batch_size: int = 64):
        sampled_transitions = random.sample(self.transitions, batch_size)
        batch = list(zip(*sampled_transitions))
        return batch