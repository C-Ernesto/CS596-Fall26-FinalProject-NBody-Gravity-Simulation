from __future__ import annotations

import unittest

import torch

from ml.baseline_model import LocalParticleMLP, PairwiseForceBaseline


class PairwiseForceBaselineTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(7)
        self.model = PairwiseForceBaseline(hidden_dim=16, hidden_layers=2)

    def test_one_body_has_zero_acceleration(self) -> None:
        result = self.model(
            torch.tensor([[1.0, 2.0, 3.0]]),
            torch.tensor([[1.0]]),
            torch.tensor([0.01]),
            torch.tensor([0, 1]),
        )
        torch.testing.assert_close(result, torch.zeros_like(result))

    def test_batch_does_not_create_cross_system_interactions(self) -> None:
        positions = torch.tensor([[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        masses = torch.tensor([[0.5], [0.5]])
        epsilon = torch.tensor([0.01])
        one = self.model(positions, masses, epsilon, torch.tensor([0, 2]))
        combined = self.model(
            torch.cat([positions, positions]),
            torch.cat([masses, masses]),
            torch.tensor([0.01, 0.01]),
            torch.tensor([0, 2, 4]),
        )
        torch.testing.assert_close(combined[:2], one)
        torch.testing.assert_close(combined[2:], one)

    def test_rotation_equivariance(self) -> None:
        positions = torch.tensor(
            [[-1.0, 0.2, 0.0], [0.5, 1.0, -0.3], [0.2, -0.4, 0.8]]
        )
        masses = torch.tensor([[0.2], [0.3], [0.5]])
        rotation = torch.tensor(
            [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
        )
        original = self.model(
            positions, masses, torch.tensor([0.02]), torch.tensor([0, 3])
        )
        rotated = self.model(
            positions @ rotation.T,
            masses,
            torch.tensor([0.02]),
            torch.tensor([0, 3]),
        )
        torch.testing.assert_close(rotated, original @ rotation.T)


class LocalParticleMLPTests(unittest.TestCase):
    def test_output_shape_and_finite_values(self) -> None:
        model = LocalParticleMLP(hidden_dim=16, hidden_layers=2)
        features = torch.randn(23, 4)
        predictions = model(features)
        self.assertEqual(predictions.shape, (23, 3))
        self.assertTrue(torch.isfinite(predictions).all())

    def test_particles_are_independent(self) -> None:
        torch.manual_seed(9)
        model = LocalParticleMLP(hidden_dim=16, hidden_layers=2)
        features = torch.randn(4, 4)
        original = model(features)
        changed = features.clone()
        changed[1] += 10.0
        updated = model(changed)
        torch.testing.assert_close(original[[0, 2, 3]], updated[[0, 2, 3]])


if __name__ == "__main__":
    unittest.main()
