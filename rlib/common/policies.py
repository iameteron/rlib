import torch
from torch import nn
from torch.distributions import Categorical, Normal


class StochasticMlpPolicy(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_size=128):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.hidden_size = hidden_size

        self.shared_net = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )

        self.mu_layer = nn.Linear(hidden_size, action_dim)
        self.log_std_layer = nn.Linear(hidden_size, action_dim)

    def forward(self, input):
        """
        Args:
            input (torch.Tensor): (B, obs_dim)

        Returns:
            mu: (torch.Tensor): (B, action_dim)
            log_std: (torch.Tensor): (B, action_dim)
        """
        x = self.shared_net(input)
        mu = torch.tanh(self.mu_layer(x))
        log_std = self.log_std_layer(x)
        log_std = torch.clamp(log_std, min=-20, max=2)

        return mu, log_std

    def get_action(
        self, input, action=None, deterministic=False, sample_gradients=False
    ):
        """
        Args:
            input (torch.Tensor): (B, obs_dim)

        Returns:
            action: (torch.Tensor): (B, action_dim)
            log_prob_action: (torch.Tensor): (B, 1)
        """
        mu, log_std = self.forward(input)
        dist = Normal(mu, torch.exp(log_std))

        if action is None:
            if deterministic:
                action = mu
            elif sample_gradients:
                action = dist.rsample()
            else:
                action = dist.sample()

        log_prob_action = dist.log_prob(action).sum(dim=1)
        log_prob_action = log_prob_action.reshape(-1, 1)

        return action, log_prob_action

    def predict(self, observation, action=None, deterministic=False):
        """
        Called for env observation

        Args:
            observation (np.ndarray): (obs_dim,)

        Returns:
            action: (np.ndarray): (action_dim,)
            log_prob_action: (torch.Tensor): (1, 1)
        """

        expected_shape = (self.obs_dim,)
        if observation.shape != (self.obs_dim,):
            raise ValueError(
                f"Expected shape {expected_shape}, but got {observation.shape}"
            )

        input = torch.FloatTensor(observation.reshape(1, self.obs_dim))

        action, log_prob_action = self.get_action(input)
        action = action.detach().numpy()
        action = action.reshape((self.action_dim,))

        return action, log_prob_action


class DiscreteStochasticMlpPolicy(nn.Module):
    def __init__(self, obs_dim, action_size, hidden_size=64):
        super().__init__()
        self.obs_dim = obs_dim
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_size),
            nn.Softmax(dim=1),
        )

    def forward(self, input):
        probs = self.net(input)
        return probs

    def get_action(self, input, action=None, deterministic=False):
        """
        Args:
            input (torch.Tensor): (B, obs_dim)

        Returns:
            action: (np.ndrray): (B,)
            log_prob_action: (torch.Tensor): (B, 1)
        """
        probs = self.forward(input)
        dist = Categorical(probs)

        if action is None:
            if deterministic:
                action = torch.argmax(probs, dim=1)
            else:
                action = dist.sample()

        log_prob_action = dist.log_prob(action)
        log_prob_action = log_prob_action.reshape(-1, 1)

        return action, log_prob_action

    def predict(self, observation, action=None, deterministic=False):
        """
        Called for env observation

        Args:
            observation (np.ndarray): (obs_dim,)

        Returns:
            action: (np.ndarray): (1,)
            log_prob_action: (torch.Tensor): (1, 1)
        """

        expected_shape = (self.obs_dim,)
        if observation.shape != (self.obs_dim,):
            raise ValueError(
                f"Expected shape {expected_shape}, but got {observation.shape}"
            )

        input = torch.FloatTensor(observation.reshape(1, self.obs_dim))

        action, log_prob_action = self.get_action(input)
        action = action.detach().numpy()
        action = action[0]

        return action, log_prob_action


class MlpCritic(nn.Module):
    def __init__(self, input_size, hidden_size=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, input):
        return self.net(input)


class OrnsteinUhlenbeckNoise:
    def __init__(self, shape, alpha: float = 0.15, sigma: float = 0.2):
        self.shape = shape
        self.alpha = alpha
        self.sigma = sigma

        self.loc = torch.zeros(shape, dtype=torch.float32)
        self.scale = sigma * torch.ones(shape, dtype=torch.float32)
        self.dist = Normal(self.loc, self.scale)

        self.reset()

    def sample(self):
        self.state = self.alpha * self.state + self.dist.sample()
        return self.state

    def reset(self):
        self.state = torch.zeros(self.shape)

    
class DeterministicMlpPolicy(nn.Module):
    def __init__(
        self, obs_dim, action_dim, hidden_size=256, noise=OrnsteinUhlenbeckNoise
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.hidden_size = hidden_size

        self.shape = (1, self.action_dim)
        self.noise = noise(self.shape)

        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_dim),
        )

    def forward(self, input):
        """
        Args:
            input (torch.Tensor): (B, obs_dim)

        Returns:
            output: (torch.Tensor): (B, action_dim)
        """
        output = torch.tanh(self.net(input))
        return output

    def get_action(self, input, deterministic=False):
        """
        Args:
            input (torch.Tensor): (B, obs_dim)

        Returns:
            action: (torch.Tensor): (B, action_dim)
            log_prob_action: (torch.Tensor): (B, 1)
        """
        action = self.forward(input)

        if not deterministic:
            epsilon = self.noise.sample()
            action += epsilon

        return action

    def predict(self, observation, action=None, deterministic=False):
        """
        Called for env observation

        Args:
            observation (np.ndarray): (obs_dim,)

        Returns:
            action: (np.ndarray): (action_dim,)
            log_prob_action: (torch.Tensor): (1, 1)
        """

        expected_shape = (self.obs_dim,)
        if observation.shape != (self.obs_dim,):
            raise ValueError(
                f"Expected shape {expected_shape}, but got {observation.shape}"
            )

        input = torch.FloatTensor(observation.reshape(1, self.obs_dim))

        action = self.get_action(input, deterministic)
        action = action.detach().numpy()
        action = action.reshape((self.action_dim,))

        return action, {}
    

class MlpQCritic(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_size=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim + action_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, observations, actions):
        """
        Args:
            observations (torch.Tensor): (B, obs_dim)
            actions (torch.Tensor): (B, action_dim)

        Returns:
            q_values: (torch.Tensor): (B, 1)
        """
        input = torch.cat((observations, actions), dim=1)
        return self.net(input)