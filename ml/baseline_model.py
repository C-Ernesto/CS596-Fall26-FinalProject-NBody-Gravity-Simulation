"""Small-N fully connected force baseline for pipeline validation."""

from __future__ import annotations

import torch
from torch import nn


class PairwiseForceBaseline(nn.Module):
    """Learn a radial pair interaction and sum it into body accelerations.

    This model is translation and rotation equivariant and linear in source
    mass, but it explicitly evaluates all directed pairs and remains O(N^2).
    It is an accuracy baseline, not the proposed scalable force surrogate.
    """

    def __init__(
        self,
        hidden_dim: int = 64,
        hidden_layers: int = 3,
        target_chunk_size: int = 128,
        minimum_log_kernel: float = -20.0,
        maximum_log_kernel: float = 20.0,
    ) -> None:
        super().__init__()
        if hidden_dim <= 0 or hidden_layers <= 0 or target_chunk_size <= 0:
            raise ValueError("hidden_dim, hidden_layers, and target_chunk_size must be positive")
        layers: list[nn.Module] = []
        input_dim = 2  # log softened distance squared and log epsilon
        for layer_index in range(hidden_layers):
            layers.append(nn.Linear(input_dim if layer_index == 0 else hidden_dim, hidden_dim))
            layers.append(nn.SiLU())
        layers.append(nn.Linear(hidden_dim, 1))
        self.radial_network = nn.Sequential(*layers)
        self.target_chunk_size = target_chunk_size
        self.minimum_log_kernel = minimum_log_kernel
        self.maximum_log_kernel = maximum_log_kernel

    def forward(
        self,
        positions: torch.Tensor,
        masses: torch.Tensor,
        epsilon: torch.Tensor,
        ptr: torch.Tensor,
    ) -> torch.Tensor:
        if positions.ndim != 2 or positions.shape[1] != 3:
            raise ValueError("positions must have shape (total_particles, 3)")
        if masses.ndim == 1:
            masses = masses.unsqueeze(1)
        if masses.ndim != 2 or masses.shape != (len(positions), 1):
            raise ValueError("masses must have shape (total_particles, 1)")
        if ptr.ndim != 1 or len(ptr) != len(epsilon) + 1:
            raise ValueError("ptr must contain one boundary per system plus the final boundary")

        predictions: list[torch.Tensor] = []
        for system_index in range(len(epsilon)):
            start = int(ptr[system_index].item())
            end = int(ptr[system_index + 1].item())
            system_positions = positions[start:end]
            system_masses = masses[start:end, 0]
            count = len(system_positions)
            if count == 0:
                raise ValueError("empty systems are not supported")

            softened_epsilon = epsilon[system_index].clamp_min(1.0e-12)
            log_epsilon = torch.log(softened_epsilon)
            source_indices = torch.arange(count, device=positions.device).unsqueeze(0)
            system_predictions: list[torch.Tensor] = []

            for target_start in range(0, count, self.target_chunk_size):
                target_end = min(target_start + self.target_chunk_size, count)
                targets = system_positions[target_start:target_end]
                displacement = system_positions.unsqueeze(0) - targets.unsqueeze(1)
                softened_distance_squared = (
                    displacement.square().sum(dim=2) + softened_epsilon.square()
                )
                radial_features = torch.stack(
                    [
                        torch.log(softened_distance_squared.clamp_min(1.0e-12)),
                        log_epsilon.expand_as(softened_distance_squared),
                    ],
                    dim=2,
                )
                log_kernel = self.radial_network(radial_features).squeeze(2)
                kernel = torch.exp(
                    log_kernel.clamp(
                        min=self.minimum_log_kernel,
                        max=self.maximum_log_kernel,
                    )
                )
                coefficients = kernel * system_masses.unsqueeze(0)

                target_indices = torch.arange(
                    target_start, target_end, device=positions.device
                ).unsqueeze(1)
                coefficients = coefficients * (target_indices != source_indices)
                acceleration = torch.einsum(
                    "ts,tsd->td", coefficients, displacement
                )
                system_predictions.append(acceleration)

            predictions.append(torch.cat(system_predictions, dim=0))
        return torch.cat(predictions, dim=0)
