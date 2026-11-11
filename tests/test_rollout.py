from __future__ import annotations

import unittest

import torch

from ml.rollout import (
    dimensionless_time_step,
    exact_acceleration,
    softened_total_energy,
    total_momentum,
    velocity_verlet_step,
)


class RolloutTests(unittest.TestCase):
    def test_dimensionless_time_step(self) -> None:
        self.assertAlmostEqual(dimensionless_time_step(0.1, 2.0, 8.0), 0.1)

    def test_two_body_exact_acceleration(self) -> None:
        positions = torch.tensor([[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=torch.float64)
        masses = torch.tensor([[1.0], [1.0]], dtype=torch.float64)
        acceleration = exact_acceleration(positions, masses, 0.0)
        expected = torch.tensor([[0.25, 0.0, 0.0], [-0.25, 0.0, 0.0]], dtype=torch.float64)
        torch.testing.assert_close(acceleration, expected)

    def test_one_body_constant_velocity(self) -> None:
        positions = torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float64)
        velocities = torch.tensor([[0.5, -0.25, 1.0]], dtype=torch.float64)
        accelerations = torch.zeros_like(positions)
        new_positions, new_velocities, new_accelerations = velocity_verlet_step(
            positions,
            velocities,
            accelerations,
            0.1,
            lambda state: torch.zeros_like(state),
        )
        torch.testing.assert_close(
            new_positions, torch.tensor([[1.05, 1.975, 3.1]], dtype=torch.float64)
        )
        torch.testing.assert_close(new_velocities, velocities)
        torch.testing.assert_close(new_accelerations, accelerations)

    def test_energy_and_momentum_are_finite(self) -> None:
        positions = torch.tensor([[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=torch.float64)
        velocities = torch.tensor([[0.0, 0.1, 0.0], [0.0, -0.1, 0.0]], dtype=torch.float64)
        masses = torch.tensor([[0.5], [0.5]], dtype=torch.float64)
        energy = softened_total_energy(positions, velocities, masses, 0.01)
        momentum = total_momentum(velocities, masses)
        self.assertTrue(torch.isfinite(energy))
        torch.testing.assert_close(momentum, torch.zeros(3, dtype=torch.float64))


if __name__ == "__main__":
    unittest.main()
