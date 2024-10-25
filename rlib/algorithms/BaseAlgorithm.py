from abc import ABC, abstractmethod

class BaseAlgorithm(ABC):
    def __init__(self):
        pass
    
    @abstractmethod
    def predict(self, obs):
        pass

    @abstractmethod
    def train(self, total_timesteps: int):
        pass