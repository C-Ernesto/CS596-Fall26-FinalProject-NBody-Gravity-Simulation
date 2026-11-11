"""PyTorch datasets, variable-size collation, and particle-budget batching."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Iterator, Sequence

import torch
from torch.utils.data import DataLoader, Dataset, Sampler

from ml.dataset import SnapshotDataset


class NBodyTorchDataset(Dataset):
    """Convert complete Parquet-backed N-body snapshots to PyTorch tensors."""

    def __init__(
        self,
        root: str | Path,
        split: str | None = None,
        evaluation_group: str | None = None,
        normalize: bool = True,
        include_velocity_features: bool = False,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        if not dtype.is_floating_point:
            raise ValueError("dtype must be a floating-point torch dtype")
        self.snapshots = SnapshotDataset(
            root=root,
            split=split,
            evaluation_group=evaluation_group,
            normalize=normalize,
        )
        if len(self.snapshots) == 0:
            raise ValueError("dataset filters selected no snapshots")
        self.normalize = normalize
        self.include_velocity_features = include_velocity_features
        self.dtype = dtype

    @property
    def index(self):
        """Expose read-only snapshot metadata for samplers and inspection."""
        return self.snapshots.index

    @property
    def input_feature_names(self) -> tuple[str, ...]:
        names = ["x", "y", "z", "mass"]
        if self.include_velocity_features:
            names.extend(["vx", "vy", "vz"])
        return tuple(names)

    def __len__(self) -> int:
        return len(self.snapshots)

    def __getitem__(self, item: int) -> dict:
        sample = self.snapshots[item]
        positions = torch.as_tensor(sample["positions"], dtype=self.dtype).contiguous()
        velocities = torch.as_tensor(sample["velocities"], dtype=self.dtype).contiguous()
        masses = torch.as_tensor(sample["masses"], dtype=self.dtype).reshape(-1, 1).contiguous()
        accelerations = torch.as_tensor(
            sample["accelerations"], dtype=self.dtype
        ).contiguous()
        epsilon = torch.tensor(sample["epsilon"], dtype=self.dtype)

        feature_parts = [positions, masses]
        if self.include_velocity_features:
            feature_parts.append(velocities)
        node_features = torch.cat(feature_parts, dim=1)

        return {
            "node_features": node_features,
            "positions": positions,
            "velocities": velocities,
            "masses": masses,
            "accelerations": accelerations,
            "epsilon": epsilon,
            "particle_count": positions.shape[0],
            "metadata": sample["metadata"],
        }


def collate_nbody_snapshots(samples: Sequence[dict]) -> dict:
    """Concatenate variable-N systems while retaining exact system boundaries."""
    if not samples:
        raise ValueError("cannot collate an empty sample list")

    particle_counts = torch.tensor(
        [int(sample["particle_count"]) for sample in samples], dtype=torch.long
    )
    ptr = torch.zeros(len(samples) + 1, dtype=torch.long)
    ptr[1:] = torch.cumsum(particle_counts, dim=0)
    batch_index = torch.repeat_interleave(
        torch.arange(len(samples), dtype=torch.long), particle_counts
    )

    return {
        "node_features": torch.cat([sample["node_features"] for sample in samples]),
        "positions": torch.cat([sample["positions"] for sample in samples]),
        "velocities": torch.cat([sample["velocities"] for sample in samples]),
        "masses": torch.cat([sample["masses"] for sample in samples]),
        "accelerations": torch.cat([sample["accelerations"] for sample in samples]),
        "epsilon": torch.stack([sample["epsilon"] for sample in samples]),
        "particle_counts": particle_counts,
        "batch_index": batch_index,
        "ptr": ptr,
        "metadata": [sample["metadata"] for sample in samples],
    }


class ParticleBudgetBatchSampler(Sampler[list[int]]):
    """Batch snapshots by shard while limiting snapshots and total particles."""

    def __init__(
        self,
        dataset: NBodyTorchDataset,
        max_snapshots: int = 4,
        max_particles: int = 4096,
        shuffle: bool = False,
        seed: int = 12345,
        drop_last: bool = False,
    ) -> None:
        if max_snapshots <= 0:
            raise ValueError("max_snapshots must be positive")
        if max_particles <= 0:
            raise ValueError("max_particles must be positive")
        self.dataset = dataset
        self.max_snapshots = max_snapshots
        self.max_particles = max_particles
        self.shuffle = shuffle
        self.seed = seed
        self.drop_last = drop_last
        self.epoch = 0

        groups: dict[str, list[int]] = defaultdict(list)
        for index, path in enumerate(dataset.index["parquet_path"].astype(str)):
            groups[path].append(index)
        self.groups = dict(groups)

    def set_epoch(self, epoch: int) -> None:
        """Choose a deterministic but different shuffle for a training epoch."""
        self.epoch = epoch

    def _build_batches(self) -> list[list[int]]:
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        batches: list[list[int]] = []
        particle_counts = self.dataset.index["particle_count"].astype(int)

        group_items = list(self.groups.items())
        if self.shuffle and len(group_items) > 1:
            order = torch.randperm(len(group_items), generator=generator).tolist()
            group_items = [group_items[index] for index in order]

        for _, original_indices in group_items:
            indices = list(original_indices)
            if self.shuffle and len(indices) > 1:
                order = torch.randperm(len(indices), generator=generator).tolist()
                indices = [indices[index] for index in order]

            current: list[int] = []
            current_particles = 0
            for index in indices:
                count = int(particle_counts.iloc[index])
                exceeds_snapshots = len(current) >= self.max_snapshots
                exceeds_particles = current and (
                    current_particles + count > self.max_particles
                )
                if exceeds_snapshots or exceeds_particles:
                    batches.append(current)
                    current = []
                    current_particles = 0
                current.append(index)
                current_particles += count

            if current and (not self.drop_last or len(current) == self.max_snapshots):
                batches.append(current)

        # Do not globally reshuffle the completed batches. Keeping a shard's
        # batches contiguous lets SnapshotDataset reuse its one-shard cache.
        # Shard order and snapshot order within each shard are already shuffled.
        return batches

    def __iter__(self) -> Iterator[list[int]]:
        yield from self._build_batches()

    def __len__(self) -> int:
        return len(self._build_batches())


def make_dataloader(
    root: str | Path,
    split: str,
    evaluation_group: str | None = None,
    normalize: bool = True,
    include_velocity_features: bool = False,
    dtype: torch.dtype = torch.float32,
    max_snapshots: int = 4,
    max_particles: int = 4096,
    shuffle: bool | None = None,
    seed: int = 12345,
    num_workers: int = 0,
    pin_memory: bool | None = None,
) -> DataLoader:
    """Construct a shard-aware DataLoader suitable for training or evaluation."""
    dataset = NBodyTorchDataset(
        root=root,
        split=split,
        evaluation_group=evaluation_group,
        normalize=normalize,
        include_velocity_features=include_velocity_features,
        dtype=dtype,
    )
    if shuffle is None:
        shuffle = split == "train"
    if pin_memory is None:
        pin_memory = torch.cuda.is_available()
    sampler = ParticleBudgetBatchSampler(
        dataset=dataset,
        max_snapshots=max_snapshots,
        max_particles=max_particles,
        shuffle=shuffle,
        seed=seed,
    )
    return DataLoader(
        dataset,
        batch_sampler=sampler,
        collate_fn=collate_nbody_snapshots,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
    )
