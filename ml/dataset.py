"""Snapshot-oriented loader for the prepared N-body force dataset."""

from __future__ import annotations

import json
from collections.abc import Collection
from pathlib import Path

import numpy as np
import pandas as pd


class SnapshotDataset:
    """Loads one complete variable-size N-body snapshot at a time."""

    def __init__(
        self,
        root: str | Path,
        split: str | None = None,
        evaluation_group: str | None = None,
        particle_counts: Collection[int] | None = None,
        normalize: bool = True,
    ) -> None:
        self.root = Path(root)
        self.index = pd.read_parquet(self.root / "snapshot_index.parquet")
        self.particle_counts: set[int] | None = None
        if split is not None:
            self.index = self.index[self.index["split"] == split]
        if evaluation_group is not None:
            self.index = self.index[self.index["evaluation_group"] == evaluation_group]
        if particle_counts is not None:
            allowed_counts = {int(count) for count in particle_counts}
            if not allowed_counts or any(count <= 0 for count in allowed_counts):
                raise ValueError("particle_counts must contain positive integers")
            self.particle_counts = allowed_counts
            self.index = self.index[self.index["particle_count"].isin(allowed_counts)]
        self.index = self.index.reset_index(drop=True)
        self.normalize = normalize
        self.normalization = json.loads((self.root / "normalization.json").read_text(encoding="utf-8"))
        self._cached_path: Path | None = None
        self._cached_frame: pd.DataFrame | None = None

    def __len__(self) -> int:
        return len(self.index)

    def _read_shard(self, relative_path: str) -> pd.DataFrame:
        path = self.root / relative_path
        if path != self._cached_path:
            filters = None
            if self.particle_counts is not None:
                filters = [("particle_count", "in", sorted(self.particle_counts))]
            self._cached_frame = pd.read_parquet(path, filters=filters)
            self._cached_path = path
        assert self._cached_frame is not None
        return self._cached_frame

    def __getitem__(self, item: int) -> dict:
        metadata = self.index.iloc[item]
        shard = self._read_shard(metadata["parquet_path"])
        frame = shard[
            (shard["trajectory_uid"] == metadata["trajectory_uid"])
            & (shard["snapshot_id"] == metadata["snapshot_id"])
        ].sort_values("body_id")

        positions = frame[["x", "y", "z"]].to_numpy(dtype=np.float64, copy=True)
        velocities = frame[["vx", "vy", "vz"]].to_numpy(dtype=np.float64, copy=True)
        masses = frame["mass"].to_numpy(dtype=np.float64, copy=True)
        accelerations = frame[["ax", "ay", "az"]].to_numpy(dtype=np.float64, copy=True)
        epsilon = float(metadata["epsilon"])
        if self.normalize:
            radius = float(metadata["characteristic_radius"])
            total_mass = float(metadata["total_mass"])
            positions /= radius
            velocities /= np.sqrt(total_mass / radius)
            masses /= total_mass
            accelerations /= total_mass / radius**2
            epsilon /= radius

        return {
            "positions": positions,
            "velocities": velocities,
            "masses": masses,
            "accelerations": accelerations,
            "epsilon": epsilon,
            "metadata": metadata.to_dict(),
        }
