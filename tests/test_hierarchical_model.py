from __future__ import annotations

import unittest

import torch

from ml.baseline_model import PairwiseForceBaseline
from ml.hierarchical_model import HierarchicalForceModel
from ml.rollout import exact_acceleration


class HierarchicalForceModelTests(unittest.TestCase):
    def test_zero_theta_matches_direct_force(self) -> None:
        torch.manual_seed(4)
        positions = torch.randn(17, 3, dtype=torch.float64)
        masses = torch.rand(17, 1, dtype=torch.float64)
        masses /= masses.sum()
        epsilon = torch.tensor([0.01], dtype=torch.float64)
        model = HierarchicalForceModel(
            hidden_dim=8,
            hidden_layers=1,
            theta=0.0,
            leaf_capacity=4,
        ).to(dtype=torch.float64)
        result = model(positions, masses, epsilon, torch.tensor([0, 17]))
        expected = exact_acceleration(positions, masses, epsilon[0])
        torch.testing.assert_close(result, expected, rtol=1.0e-12, atol=1.0e-12)
        self.assertEqual(model.last_interaction_stats["body_interactions"], 17 * 16)
        self.assertEqual(model.last_interaction_stats["aggregate_interactions"], 0)

    def test_positive_theta_reduces_interactions(self) -> None:
        torch.manual_seed(5)
        positions = torch.rand(128, 3)
        masses = torch.full((128, 1), 1.0 / 128)
        model = HierarchicalForceModel(
            hidden_dim=8,
            hidden_layers=1,
            theta=0.8,
            leaf_capacity=8,
        )
        result = model(
            positions,
            masses,
            torch.tensor([0.01]),
            torch.tensor([0, 128]),
        )
        self.assertEqual(result.shape, (128, 3))
        self.assertTrue(torch.isfinite(result).all())
        self.assertLess(model.last_interaction_stats["total_interactions"], 128 * 127)

    def test_two_batched_systems_remain_separate(self) -> None:
        positions = torch.tensor(
            [[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]
        )
        masses = torch.full((4, 1), 0.5)
        model = HierarchicalForceModel(theta=0.0, leaf_capacity=1)
        result = model(
            positions,
            masses,
            torch.tensor([0.0, 0.0]),
            torch.tensor([0, 2, 4]),
        )
        expected = torch.tensor(
            [[0.125, 0.0, 0.0], [-0.125, 0.0, 0.0]]
        )
        torch.testing.assert_close(result[:2], expected)
        torch.testing.assert_close(result[2:], expected)

    def test_aggregate_interactions_train_radial_network(self) -> None:
        torch.manual_seed(7)
        positions = torch.rand(32, 3)
        masses = torch.full((32, 1), 1.0 / 32)
        model = HierarchicalForceModel(
            hidden_dim=8,
            hidden_layers=1,
            theta=1.2,
            leaf_capacity=4,
        )
        predictions = model(
            positions,
            masses,
            torch.tensor([0.01]),
            torch.tensor([0, 32]),
        )
        self.assertGreater(model.last_interaction_stats["aggregate_interactions"], 0)
        predictions.square().mean().backward()
        gradients = [
            parameter.grad
            for parameter in model.radial_network.parameters()
            if parameter.grad is not None
        ]
        self.assertTrue(gradients)
        self.assertTrue(all(torch.isfinite(gradient).all() for gradient in gradients))
        self.assertGreater(sum(float(gradient.abs().sum()) for gradient in gradients), 0.0)

    def test_pairwise_initialization_preserves_predictions(self) -> None:
        torch.manual_seed(9)
        pairwise = PairwiseForceBaseline(hidden_dim=8, hidden_layers=1)
        legacy = HierarchicalForceModel(
            hidden_dim=8,
            hidden_layers=1,
            theta=1.0,
            leaf_capacity=4,
            use_geometry_features=False,
        )
        geometry = HierarchicalForceModel(
            hidden_dim=8,
            hidden_layers=1,
            theta=1.0,
            leaf_capacity=4,
            use_geometry_features=True,
        )
        legacy.load_pairwise_radial_network(pairwise)
        geometry.load_pairwise_radial_network(pairwise)
        positions = torch.rand(40, 3)
        masses = torch.full((40, 1), 1.0 / 40)
        arguments = (positions, masses, torch.tensor([0.01]), torch.tensor([0, 40]))
        torch.testing.assert_close(legacy(*arguments), geometry(*arguments))

    def test_momentum_correction_enforces_zero_net_force(self) -> None:
        torch.manual_seed(11)
        positions = torch.rand(96, 3)
        masses = torch.rand(96, 1)
        masses /= masses.sum()
        model = HierarchicalForceModel(
            hidden_dim=8,
            hidden_layers=1,
            theta=1.0,
            leaf_capacity=8,
            enforce_zero_net_force=True,
        )
        accelerations = model(
            positions,
            masses,
            torch.tensor([0.01]),
            torch.tensor([0, 96]),
        )
        total_force = (masses * accelerations).sum(dim=0)
        torch.testing.assert_close(
            total_force,
            torch.zeros_like(total_force),
            rtol=0.0,
            atol=2.0e-7,
        )


if __name__ == "__main__":
    unittest.main()
