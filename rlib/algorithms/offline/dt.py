import gymnasium as gym
import numpy as np
import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset

from ...common.evaluation import save_frames_as_gif
from ...common.logger import TensorBoardLogger


class DecisionTransformer(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        trajectory_len: int,
        embedding_dim: int = 32,
        nhead: int = 1,
        num_layers: int = 1,
    ):
        super().__init__()

        self.R_embedding = nn.Linear(1, embedding_dim)
        self.s_embedding = nn.Linear(obs_dim, embedding_dim)
        self.a_embedding = nn.Linear(action_dim, embedding_dim)

        self.t_embedding = nn.Embedding(trajectory_len, embedding_dim)

        decoder_layer = nn.TransformerEncoderLayer(
            embedding_dim,
            nhead,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(decoder_layer, num_layers)
        self.head = nn.Linear(embedding_dim, action_dim)

    def forward(self, R, s, a, t):
        """
        Args:
            R (torch.Tensor): (B, T, 1)
            s (torch.Tensor): (B, T, obs_dim)
            a (torch.Tensor): (B, T, action_dim)
            t (torch.Tensor): (B, T, 1)

        Returns:
            a_pred: (torch.Tensor): (B, T, action_dim)
        """
        t_emb = self.t_embedding(t.squeeze(-1))

        R_emb = self.R_embedding(R) + t_emb  # (B, T, C)
        s_emb = self.s_embedding(s) + t_emb  # (B, T, C)
        a_emb = self.a_embedding(a) + t_emb  # (B, T, C)

        B, T, C = s_emb.shape

        token_emb = torch.stack((R_emb, s_emb, a_emb), dim=1)  # (B, 3, T, C)
        token_emb = token_emb.permute(0, 2, 1, 3).reshape(B, 3 * T, C)  # (B, 3T, C)

        device = next(self.parameters()).device
        mask = nn.Transformer.generate_square_subsequent_mask(3 * T, device=device)

        hidden_states = self.transformer.forward(token_emb, mask)  # (B, 3T, C)
        hidden_states = hidden_states.reshape(B, T, 3, C).permute(
            0, 2, 1, 3
        )  # (B, 3, T, C)
        a_hidden = hidden_states[:, 1]  # (B, T, C)

        a_pred = torch.tanh(self.head(a_hidden))  # (B, T, action_dim)

        return a_pred


class TrajectoryDataset(Dataset):
    def __init__(self, trajectories, K):
        self.trajectories = trajectories
        self._add_timesteps()

        self.sequences = []

        for traj in self.trajectories:
            traj_len = traj["observations"].shape[0]
            for i in range(0, traj_len - K + 1):
                R = traj["q_estimations"][i : i + K]
                s = traj["observations"][i : i + K]
                a = traj["actions"][i : i + K]
                t = traj["timesteps"][i : i + K]

                self.sequences.append((R, s, a, t))

    def __getitem__(self, i):
        return self.sequences[i]

    def __len__(self):
        return len(self.sequences)

    def _add_timesteps(self):
        for traj in self.trajectories:
            traj_len = traj["observations"].shape[0]
            traj["timesteps"] = torch.arange(0, traj_len).reshape(-1, 1)



def dt_evaluation(
    model: nn.Module,
    env: gym.Env,
    target_return: float = -0.1,
    reward_norm: float = 1000,
    K: int = 30,
    device: str = "cuda",
    visualize: bool = False,
):
    trajectory = {
        "observations": [],
        "actions": [],
        "rewards": [],
        "terminated": [],
        "truncated": [],
    }

    frames = []

    model.eval()
    obs, _ = env.reset()
    action = np.zeros_like(env.action_space.sample())

    R = torch.tensor(target_return, dtype=torch.float32, device=device).reshape(1, 1, 1)
    s = torch.tensor(obs, dtype=torch.float32, device=device).reshape(1, 1, -1)
    a = torch.tensor(action, dtype=torch.float32, device=device).reshape(1, 1, -1)
    t = torch.tensor(0, device=device).reshape(1, 1, 1)

    while True:
        with torch.no_grad():
            a_pred = model.forward(R, s, a, t)[:, -1, :]

        a_pred = a_pred.reshape(1, 1, -1)

        action = a_pred.squeeze().cpu().detach().numpy()

        next_obs, reward, terminated, truncated, _ = env.step(action)

        trajectory["observations"].append(obs)
        trajectory["actions"].append(action)
        trajectory["rewards"].append(reward)
        trajectory["terminated"].append(terminated)
        trajectory["truncated"].append(truncated)

        reward /= reward_norm

        next_obs = torch.tensor(next_obs, dtype=torch.float32, device=device).reshape(
            1, 1, -1
        )
        reward = torch.tensor(reward, dtype=torch.float32, device=device).reshape(
            1, 1, 1
        )

        return_to_go = R[:, -1, :] - reward

        timestep = t[:, -1, :] + 1
        timestep = timestep.reshape(1, 1, 1)

        R = torch.cat((R, return_to_go), dim=1)[:, -K:, :]
        s = torch.cat((s, next_obs), dim=1)[:, -K:, :]
        a = torch.cat((a, a_pred), dim=1)[:, -K:, :]
        t = torch.cat((t, timestep), dim=1)[:, -K:, :]

        if visualize:
            frames.append(env.render())

        if terminated or truncated:
            break

    if visualize:
        print("saving...")
        save_frames_as_gif(frames)

    return trajectory


def dt_train(
    decision_transformer: nn.Module,
    env: gym.Env,
    optimizer: Adam,
    train_dataloader: DataLoader,
    test_dataloader: DataLoader,
    total_epochs: int = 100,
    target_return: float = 1,
    reward_norm: float = 1000,
    K: int = 30,
    device: str = "cuda",
    exp_name: str = "",
):
    logger = TensorBoardLogger(log_dir="./tb_logs/dt_" + exp_name)

    for epoch_n in range(total_epochs):
        train_losses, test_losses = [], []

        decision_transformer.train()
        for R, s, a, t in train_dataloader:
            R, s, a, t = R.to(device), s.to(device), a.to(device), t.to(device)
            a_preds = decision_transformer(R, s, a, t)

            batch_loss = ((a_preds - a) ** 2).mean()
            train_losses.append(batch_loss.item())

            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()

        decision_transformer.eval()
        with torch.no_grad():
            for R, s, a, t in test_dataloader:
                R, s, a, t = R.to(device), s.to(device), a.to(device), t.to(device)
                a_preds = decision_transformer(R, s, a, t)

                batch_loss = ((a_preds - a) ** 2).mean()
                test_losses.append(batch_loss.item())

        loss = {}
        loss["train"] = sum(train_losses) / len(train_losses)
        loss["test"] = sum(test_losses) / len(test_losses)
        logger.log_scalars(loss, epoch_n)

        metrics = {}
        trajectory = dt_evaluation(
            decision_transformer, env, target_return, reward_norm, K
        )
        metrics["traj_reward"] = sum(trajectory["rewards"])
        logger.log_scalars(metrics, epoch_n)