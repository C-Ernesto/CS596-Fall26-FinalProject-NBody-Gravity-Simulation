from __future__ import annotations

import unittest
from pathlib import Path

import torch

from ml.torch_dataset import (
    NBodyTorchDataset,
    ParticleBudgetBatchSampler,
    collate_nbody_snapshots,
    make_dataloader,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = PROJECT_ROOT / "data" / "processed" / "candidate_v1"


@unittest.skipUnless(
    (DATASET_ROOT / "snapshot_index.parquet").is_file(),
    "candidate dataset has not been generated",
)
class TorchDatasetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.dataset = NBodyTorchDataset(DATASET_ROOT, split="train")

    def test_split_length_and_default_features(self) -> None:
        self.assertEqual(len(self.dataset), 1200)
        sample = self.dataset[0]
        self.assertEqual(sample["particle_count"], 128)
        self.assertEqual(sample["node_features"].shape, (128, 4))
        self.assertEqual(sample["positions"].shape, (128, 3))
        self.assertEqual(sample["masses"].shape, (128, 1))
        self.assertEqual(sample["accelerations"].shape, (128, 3))
        self.assertEqual(sample["node_features"].dtype, torch.float32)
        self.assertTrue(torch.isfinite(sample["node_features"]).all())
        self.assertTrue(torch.isfinite(sample["accelerations"]).all())
        torch.testing.assert_close(sample["node_features"][:, :3], sample["positions"])
        torch.testing.assert_close(sample["node_features"][:, 3:], sample["masses"])

    def test_evaluation_group_lengths(self) -> None:
        expected = {
            "interpolation_counts": 192,
            "extrapolation_count": 96,
            "held_out_family": 96,
        }
        for group, count in expected.items():
            dataset = NBodyTorchDataset(
                DATASET_ROOT,
                split="test",
                evaluation_group=group,
            )
            self.assertEqual(len(dataset), count)

    def test_optional_velocity_features(self) -> None:
        dataset = NBodyTorchDataset(
            DATASET_ROOT,
            split="validation",
            include_velocity_features=True,
        )
        sample = dataset[0]
        self.assertEqual(sample["node_features"].shape[1], 7)
        torch.testing.assert_close(
            sample["node_features"][:, 4:], sample["velocities"]
        )

    def test_variable_size_collation_preserves_boundaries(self) -> None:
        first = self.dataset[0]
        second = self.dataset[80]
        batch = collate_nbody_snapshots([first, second])
        self.assertEqual(batch["particle_counts"].tolist(), [128, 256])
        self.assertEqual(batch["ptr"].tolist(), [0, 128, 384])
        self.assertEqual(batch["positions"].shape, (384, 3))
        self.assertEqual(batch["batch_index"][:128].unique().tolist(), [0])
        self.assertEqual(batch["batch_index"][128:].unique().tolist(), [1])
        torch.testing.assert_close(batch["positions"][:128], first["positions"])
        torch.testing.assert_close(batch["positions"][128:], second["positions"])

    def test_particle_budget_sampler(self) -> None:
        sampler = ParticleBudgetBatchSampler(
            self.dataset,
            max_snapshots=4,
            max_particles=512,
            shuffle=True,
            seed=7,
        )
        batches = list(sampler)
        self.assertGreater(len(batches), 0)
        for indices in batches:
            counts = self.dataset.index.iloc[indices]["particle_count"].astype(int)
            paths = self.dataset.index.iloc[indices]["parquet_path"].unique()
            self.assertLessEqual(len(indices), 4)
            if len(indices) > 1:
                self.assertLessEqual(int(counts.sum()), 512)
            self.assertEqual(len(paths), 1)

        batch_paths = [
            self.dataset.index.iloc[indices[0]]["parquet_path"]
            for indices in batches
        ]
        transitions = sum(
            first != second
            for first, second in zip(batch_paths, batch_paths[1:])
        )
        self.assertEqual(transitions, len(set(batch_paths)) - 1)

    def test_dataloader_and_cuda_transfer(self) -> None:
        loader = make_dataloader(
            DATASET_ROOT,
            split="validation",
            max_snapshots=2,
            max_particles=512,
            shuffle=False,
        )
        batch = next(iter(loader))
        self.assertEqual(batch["ptr"][0].item(), 0)
        self.assertEqual(batch["ptr"][-1].item(), len(batch["positions"]))
        self.assertEqual(len(batch["metadata"]), len(batch["particle_counts"]))
        if torch.cuda.is_available():
            positions = batch["positions"].to("cuda", non_blocking=True)
            self.assertTrue(positions.is_cuda)
            self.assertTrue(torch.isfinite(positions).all())


if __name__ == "__main__":
    unittest.main()
