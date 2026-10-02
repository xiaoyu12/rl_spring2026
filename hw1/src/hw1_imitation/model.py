"""Model definitions for Push-T imitation policies."""

from __future__ import annotations

import abc
from typing import Literal, TypeAlias

import torch
from torch import nn


class BasePolicy(nn.Module, metaclass=abc.ABCMeta):
    """Base class for action chunking policies."""

    def __init__(self, state_dim: int, action_dim: int, chunk_size: int) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.chunk_size = chunk_size

    @abc.abstractmethod
    def compute_loss(
        self, state: torch.Tensor, action_chunk: torch.Tensor
    ) -> torch.Tensor:
        """Compute training loss for a batch."""

    @abc.abstractmethod
    def sample_actions(
        self,
        state: torch.Tensor,
        *,
        num_steps: int = 10,  # only applicable for flow policy
    ) -> torch.Tensor:
        """Generate a chunk of actions with shape (batch, chunk_size, action_dim)."""


def build_mlp(input_dim: int, output_dim: int, hidden_dims: tuple[int, ...]) -> nn.Sequential:
    """Simple MLP: Linear -> ReLU -> ... -> Linear."""
    layers: list[nn.Module] = []
    in_dim = input_dim
    for h in hidden_dims:
        layers.append(nn.Linear(in_dim, h))
        layers.append(nn.ReLU())
        in_dim = h
    layers.append(nn.Linear(in_dim, output_dim))
    return nn.Sequential(*layers)


class MSEPolicy(BasePolicy):
    """Predicts action chunks with an MSE loss."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        chunk_size: int,
        hidden_dims: tuple[int, ...] = (128, 128),
    ) -> None:
        super().__init__(state_dim, action_dim, chunk_size)
        # Maps a state to a flattened action chunk of size chunk_size * action_dim.
        self.net = build_mlp(state_dim, chunk_size * action_dim, hidden_dims)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        out = self.net(state)
        return out.view(-1, self.chunk_size, self.action_dim)

    def compute_loss(
        self,
        state: torch.Tensor,
        action_chunk: torch.Tensor,
    ) -> torch.Tensor:
        pred = self(state)
        # Squared L2 norm over the whole chunk, averaged over the batch (Eq. 1).
        return ((pred - action_chunk) ** 2).sum(dim=(1, 2)).mean()

    def sample_actions(
        self,
        state: torch.Tensor,
        *,
        num_steps: int = 10,
    ) -> torch.Tensor:
        return self(state)


class FlowMatchingPolicy(BasePolicy):
    """Predicts action chunks with a flow matching loss."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        chunk_size: int,
        hidden_dims: tuple[int, ...] = (128, 128),
    ) -> None:
        super().__init__(state_dim, action_dim, chunk_size)
        # Velocity field v_theta(o, A_tau, tau): input is the state, the flattened
        # noisy chunk, and the scalar flow timestep; output is a flattened velocity.
        self.net = build_mlp(
            state_dim + chunk_size * action_dim + 1, chunk_size * action_dim, hidden_dims
        )

    def forward(
        self, state: torch.Tensor, noisy_chunk: torch.Tensor, tau: torch.Tensor
    ) -> torch.Tensor:
        x = torch.cat([state, noisy_chunk.flatten(1), tau.view(-1, 1)], dim=-1)
        return self.net(x).view(-1, self.chunk_size, self.action_dim)

    def compute_loss(
        self,
        state: torch.Tensor,
        action_chunk: torch.Tensor,
    ) -> torch.Tensor:
        noise = torch.randn_like(action_chunk)
        tau = torch.rand(action_chunk.shape[0], device=action_chunk.device)
        t = tau.view(-1, 1, 1)
        noisy_chunk = t * action_chunk + (1 - t) * noise
        pred = self(state, noisy_chunk, tau)
        # Regress onto the straight-line velocity A - A_0 (Eq. 2).
        return ((pred - (action_chunk - noise)) ** 2).sum(dim=(1, 2)).mean()

    @torch.no_grad()
    def sample_actions(
        self,
        state: torch.Tensor,
        *,
        num_steps: int = 10,
    ) -> torch.Tensor:
        batch_size = state.shape[0]
        chunk = torch.randn(
            batch_size, self.chunk_size, self.action_dim, device=state.device, dtype=state.dtype
        )
        dt = 1.0 / num_steps
        # Euler integration of dA/dtau = v_theta from tau=0 to tau=1 (Eq. 3).
        for i in range(num_steps):
            tau = torch.full((batch_size,), i * dt, device=state.device, dtype=state.dtype)
            chunk = chunk + dt * self(state, chunk, tau)
        return chunk


PolicyType: TypeAlias = Literal["mse", "flow"]


def build_policy(
    policy_type: PolicyType,
    *,
    state_dim: int,
    action_dim: int,
    chunk_size: int,
    hidden_dims: tuple[int, ...] = (128, 128),
) -> BasePolicy:
    if policy_type == "mse":
        return MSEPolicy(
            state_dim=state_dim,
            action_dim=action_dim,
            chunk_size=chunk_size,
            hidden_dims=hidden_dims,
        )
    if policy_type == "flow":
        return FlowMatchingPolicy(
            state_dim=state_dim,
            action_dim=action_dim,
            chunk_size=chunk_size,
            hidden_dims=hidden_dims,
        )
    raise ValueError(f"Unknown policy type: {policy_type}")
