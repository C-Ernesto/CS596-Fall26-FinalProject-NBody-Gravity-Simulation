from __future__ import annotations

import unittest

import torch

from ml.train_hierarchical import (
    maximum_family_regression,
    selection_score,
    snapshot_losses,
)


class HierarchicalTrainingTests(unittest.TestCase):
    def test_snapshot_relative_loss_is_scale_invariant(self) -> None:
        targets = torch.tensor(
            [[1.0, -2.0, 0.5], [0.25, 0.75, -1.0]], dtype=torch.float32
        )
        predictions = targets + 0.1 * targets
        ptr = torch.tensor([0, 2])
        original = snapshot_losses(
            predictions, targets, ptr, beta=1.0, mode="snapshot_relative_huber"
        )
        scaled = snapshot_losses(
            predictions * 100.0,
            targets * 100.0,
            ptr,
            beta=1.0,
            mode="snapshot_relative_huber",
        )
        torch.testing.assert_close(original, scaled)

    def test_snapshot_losses_weight_systems_equally(self) -> None:
        targets = torch.ones((5, 3), dtype=torch.float32)
        predictions = targets.clone()
        predictions[:2] += 0.5
        predictions[2:] += 1.0
        losses = snapshot_losses(
            predictions,
            targets,
            torch.tensor([0, 2, 5]),
            beta=1.0,
            mode="absolute_huber",
        )
        self.assertEqual(losses.shape, (2,))
        self.assertAlmostEqual(float(losses.mean()), 0.3125)

    def test_selection_score_uses_requested_balanced_metric(self) -> None:
        metrics = {
            "loss": 0.1,
            "mean_family_relative_l2": 0.2,
            "worst_family_relative_l2": 0.3,
        }
        self.assertEqual(selection_score(metrics, "loss"), 0.1)
        self.assertEqual(
            selection_score(metrics, "mean_family_relative_l2"), 0.2
        )
        self.assertEqual(
            selection_score(metrics, "worst_family_relative_l2"), 0.3
        )

    def test_maximum_family_regression_finds_worst_family(self) -> None:
        initializer = {
            "uniform": {"relative_l2": 0.01},
            "collision": {"relative_l2": 0.04},
        }
        current = {
            "uniform": {"relative_l2": 0.012},
            "collision": {"relative_l2": 0.02},
        }
        self.assertAlmostEqual(
            maximum_family_regression(current, initializer), 0.2
        )


if __name__ == "__main__":
    unittest.main()
