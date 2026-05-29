"""MuZero networks for visual Gym / Atari observations."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    def __init__(self, num_hidden: int):
        super().__init__()
        self.conv1 = nn.Conv2d(num_hidden, num_hidden, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(num_hidden)
        self.conv2 = nn.Conv2d(num_hidden, num_hidden, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(num_hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x += residual
        return F.relu(x)


def _normalize_hidden(x: torch.Tensor) -> torch.Tensor:
    x_flat = x.view(x.size(0), -1)
    x_min = x_flat.min(dim=1, keepdim=True)[0].view(-1, 1, 1, 1)
    x_max = x_flat.max(dim=1, keepdim=True)[0].view(-1, 1, 1, 1)
    scale = x_max - x_min
    scale = torch.where(scale < 1e-5, torch.ones_like(scale), scale)
    return (x - x_min) / scale


class AtariRepresentationNetwork(nn.Module):
    """h(observation) -> hidden_state for stacked grayscale frames."""

    def __init__(self, game, in_channels: int, num_res_blocks: int, num_hidden: int):
        super().__init__()
        self.conv_stack = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, num_hidden, kernel_size=3, stride=1),
            nn.ReLU(),
        )
        self.backbone = nn.ModuleList([ResBlock(num_hidden) for _ in range(num_res_blocks)])
        self.to_latent = nn.Conv2d(num_hidden, num_hidden, kernel_size=3, stride=2)
        self.latent_h = game.row_count
        self.latent_w = game.column_count

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_stack(x)
        for block in self.backbone:
            x = block(x)
        x = self.to_latent(x)
        if x.shape[-2] != self.latent_h or x.shape[-1] != self.latent_w:
            x = F.adaptive_avg_pool2d(x, (self.latent_h, self.latent_w))
        return _normalize_hidden(x)


class AtariDynamicsNetwork(nn.Module):
    """g(hidden_state, action) -> (next_hidden_state, reward)."""

    def __init__(self, game, num_res_blocks: int, num_hidden: int):
        super().__init__()
        self.action_size = game.action_size
        self.latent_h = game.row_count
        self.latent_w = game.column_count

        self.start_block = nn.Sequential(
            nn.Conv2d(num_hidden + game.action_size, num_hidden, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_hidden),
            nn.ReLU(),
        )
        self.backbone = nn.ModuleList([ResBlock(num_hidden) for _ in range(num_res_blocks)])
        self.reward_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(num_hidden, 1),
            nn.Tanh(),
        )

    def forward(self, hidden_state: torch.Tensor, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = hidden_state.size(0)
        action_one_hot = torch.zeros(batch_size, self.action_size, device=hidden_state.device)
        action_one_hot.scatter_(1, action.long().unsqueeze(1), 1.0)
        action_planes = action_one_hot.unsqueeze(-1).unsqueeze(-1).expand(
            -1, -1, self.latent_h, self.latent_w
        )
        x = torch.cat([hidden_state, action_planes], dim=1)
        x = self.start_block(x)
        for block in self.backbone:
            x = block(x)
        return _normalize_hidden(x), self.reward_head(x)


class AtariPredictionNetwork(nn.Module):
    """f(hidden_state) -> (policy, value)."""

    def __init__(self, game, num_hidden: int):
        super().__init__()
        flat = num_hidden * game.row_count * game.column_count
        self.policy_head = nn.Sequential(
            nn.Conv2d(num_hidden, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * game.row_count * game.column_count, game.action_size),
        )
        self.value_head = nn.Sequential(
            nn.Conv2d(num_hidden, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(flat, 1),
            nn.Tanh(),
        )

    def forward(self, hidden_state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.policy_head(hidden_state), self.value_head(hidden_state)


class MuZeroAtariNetwork(nn.Module):
    """MuZero module for Gym / Atari (stacked frame observations)."""

    def __init__(self, game, in_channels: int, num_res_blocks: int = 4, num_hidden: int = 64):
        super().__init__()
        self.representation = AtariRepresentationNetwork(game, in_channels, num_res_blocks, num_hidden)
        self.dynamics = AtariDynamicsNetwork(game, num_res_blocks, num_hidden)
        self.prediction = AtariPredictionNetwork(game, num_hidden)

    def initial_inference(self, observation: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden_state = self.representation(observation)
        policy, value = self.prediction(hidden_state)
        return hidden_state, policy, value

    def recurrent_inference(
        self, hidden_state: torch.Tensor, action: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        next_hidden_state, reward = self.dynamics(hidden_state, action)
        policy, value = self.prediction(next_hidden_state)
        return next_hidden_state, reward, policy, value


class MuZeroVectorNetwork(nn.Module):
    """Compact MuZero for vector Gym envs (e.g. CartPole) — same API as MuZeroAtariNetwork."""

    def __init__(self, game, obs_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.action_size = game.action_size
        latent = hidden_dim

        self.representation = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent),
            nn.ReLU(),
        )
        self.dynamics = nn.Sequential(
            nn.Linear(latent + game.action_size, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent),
            nn.ReLU(),
        )
        self.reward_head = nn.Sequential(nn.Linear(latent, 1), nn.Tanh())
        self.policy_head = nn.Linear(latent, game.action_size)
        self.value_head = nn.Sequential(nn.Linear(latent, 1), nn.Tanh())

    def _latent_to_grid(self, x: torch.Tensor) -> torch.Tensor:
        return x.view(x.size(0), -1, 1, 1).expand(-1, -1, 1, max(4, self.action_size // 2))

    def initial_inference(self, observation: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        flat = observation.view(observation.size(0), -1)
        hidden = self.representation(flat)
        policy = self.policy_head(hidden)
        value = self.value_head(hidden)
        return self._latent_to_grid(hidden), policy, value

    def recurrent_inference(
        self, hidden_state: torch.Tensor, action: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        flat = hidden_state.mean(dim=(2, 3))
        action_one_hot = torch.zeros(flat.size(0), self.action_size, device=flat.device)
        action_one_hot.scatter_(1, action.long().unsqueeze(1), 1.0)
        next_flat = self.dynamics(torch.cat([flat, action_one_hot], dim=1))
        reward = self.reward_head(next_flat)
        policy = self.policy_head(next_flat)
        value = self.value_head(next_flat)
        return self._latent_to_grid(next_flat), reward, policy, value
