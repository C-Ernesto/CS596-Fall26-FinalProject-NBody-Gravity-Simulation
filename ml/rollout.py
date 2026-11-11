"""Reusable force, integration, and diagnostics for normalized N-body rollouts."""

from __future__ import annotations

import torch


def dimensionless_time_step(dt: float, characteristic_radius: float, total_mass: float) -> float:
    """Convert physical dt to units consistent with the prepared dataset."""
    if dt <= 0.0 or characteristic_radius <= 0.0 or total_mass <= 0.0:
        raise ValueError("dt, characteristic_radius, and total_mass must be positive")
    return dt * (total_mass / characteristic_radius**3) ** 0.5


def exact_acceleration(
    positions: torch.Tensor,
    masses: torch.Tensor,
    epsilon: torch.Tensor | float,
) -> torch.Tensor:
    """Direct softened acceleration in normalized units."""
    if masses.ndim == 2:
        masses = masses[:, 0]
    displacement = positions.unsqueeze(0) - positions.unsqueeze(1)
    epsilon_tensor = torch.as_tensor(
        epsilon, dtype=positions.dtype, device=positions.device
    )
    distance_squared = displacement.square().sum(dim=2) + epsilon_tensor.square()
    inverse_distance_cubed = distance_squared.pow(-1.5)
    inverse_distance_cubed.fill_diagonal_(0.0)
    return torch.einsum(
        "ij,ijd,j->id", inverse_distance_cubed, displacement, masses
    )


def velocity_verlet_step(
    positions: torch.Tensor,
    velocities: torch.Tensor,
    accelerations: torch.Tensor,
    dt: float,
    acceleration_function,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    half_velocities = velocities + 0.5 * dt * accelerations
    new_positions = positions + dt * half_velocities
    new_accelerations = acceleration_function(new_positions)
    new_velocities = half_velocities + 0.5 * dt * new_accelerations
    return new_positions, new_velocities, new_accelerations


def softened_total_energy(
    positions: torch.Tensor,
    velocities: torch.Tensor,
    masses: torch.Tensor,
    epsilon: torch.Tensor | float,
) -> torch.Tensor:
    if masses.ndim == 2:
        masses = masses[:, 0]
    kinetic = 0.5 * (masses * velocities.square().sum(dim=1)).sum()
    displacement = positions.unsqueeze(0) - positions.unsqueeze(1)
    epsilon_tensor = torch.as_tensor(
        epsilon, dtype=positions.dtype, device=positions.device
    )
    distances = torch.sqrt(displacement.square().sum(dim=2) + epsilon_tensor.square())
    pair_masses = masses.unsqueeze(0) * masses.unsqueeze(1)
    pair_indices = torch.triu_indices(
        len(positions), len(positions), offset=1, device=positions.device
    )
    potential = -(
        pair_masses[pair_indices[0], pair_indices[1]]
        / distances[pair_indices[0], pair_indices[1]]
    ).sum()
    return kinetic + potential


def total_momentum(velocities: torch.Tensor, masses: torch.Tensor) -> torch.Tensor:
    if masses.ndim == 2:
        masses = masses[:, 0]
    return (masses.unsqueeze(1) * velocities).sum(dim=0)
