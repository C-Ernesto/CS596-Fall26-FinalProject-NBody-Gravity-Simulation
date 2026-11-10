"""Validate a raw N-body CSV and split it by complete trajectories."""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = PROJECT_ROOT / "data" / "raw" / "pilot_cuda.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "processed" / "pilot"
REQUIRED_COLUMNS = {
    "trajectory_id",
    "snapshot_id",
    "time_step",
    "body_id",
    "particle_count",
    "seed",
    "time",
    "dt",
    "epsilon",
    "x",
    "y",
    "z",
    "vx",
    "vy",
    "vz",
    "mass",
    "ax",
    "ay",
    "az",
}


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create leakage-safe train/validation/test trajectory splits."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument("--test-fraction", type=float, default=0.15)
    parser.add_argument("--split-seed", type=int, default=2026)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_and_validate(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise ValueError(f"raw dataset does not exist: {path}")
    data = pd.read_csv(path)
    missing = REQUIRED_COLUMNS - set(data.columns)
    if missing:
        raise ValueError(f"dataset is missing columns: {', '.join(sorted(missing))}")
    if data.empty:
        raise ValueError("dataset contains no rows")
    if data[list(REQUIRED_COLUMNS)].isna().any().any():
        raise ValueError("dataset contains missing required values")

    numeric_columns = [
        "trajectory_id",
        "snapshot_id",
        "time_step",
        "time",
        "dt",
        "body_id",
        "particle_count",
        "seed",
        "epsilon",
        "x",
        "y",
        "z",
        "vx",
        "vy",
        "vz",
        "mass",
        "ax",
        "ay",
        "az",
    ]
    data[numeric_columns] = data[numeric_columns].apply(
        pd.to_numeric,
        errors="raise",
    )
    if not np.isfinite(data[numeric_columns].to_numpy()).all():
        raise ValueError("dataset contains non-finite numeric values")
    if data["particle_count"].le(0).any() or data["mass"].le(0.0).any():
        raise ValueError("particle counts and masses must be positive")
    if data.duplicated(["trajectory_id", "snapshot_id", "body_id"]).any():
        raise ValueError("dataset contains duplicate body rows within a snapshot")
    if data["trajectory_id"].isna().any():
        raise ValueError("dataset contains a missing trajectory_id")

    expected_rows = data.groupby(["trajectory_id", "snapshot_id"])[
        "particle_count"
    ].first()
    actual_rows = data.groupby(["trajectory_id", "snapshot_id"]).size()
    if not expected_rows.equals(actual_rows):
        raise ValueError("one or more snapshots has the wrong number of body rows")

    for column in ("seed", "particle_count", "initial_condition"):
        if column in data.columns:
            inconsistent = data.groupby("trajectory_id")[column].nunique().gt(1)
            if inconsistent.any():
                raise ValueError(f"{column} changes within a trajectory")
    return data


def split_trajectories(
    data: pd.DataFrame,
    validation_fraction: float,
    test_fraction: float,
    split_seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if validation_fraction <= 0.0 or test_fraction <= 0.0:
        raise ValueError("validation and test fractions must be positive")
    if validation_fraction + test_fraction >= 1.0:
        raise ValueError("validation and test fractions must sum to less than one")

    trajectory_ids = sorted(data["trajectory_id"].unique().tolist())
    if len(trajectory_ids) < 3:
        raise ValueError("at least three trajectories are required for three splits")
    random.Random(split_seed).shuffle(trajectory_ids)

    validation_count = max(1, round(len(trajectory_ids) * validation_fraction))
    test_count = max(1, round(len(trajectory_ids) * test_fraction))
    if validation_count + test_count >= len(trajectory_ids):
        validation_count = 1
        test_count = 1

    test_ids = set(trajectory_ids[:test_count])
    validation_ids = set(
        trajectory_ids[test_count : test_count + validation_count]
    )
    train_ids = set(trajectory_ids) - test_ids - validation_ids

    split_map = {trajectory_id: "train" for trajectory_id in train_ids}
    split_map.update({trajectory_id: "validation" for trajectory_id in validation_ids})
    split_map.update({trajectory_id: "test" for trajectory_id in test_ids})

    prepared = data.copy()
    prepared.insert(0, "split", prepared["trajectory_id"].map(split_map))
    manifest = (
        prepared.groupby("trajectory_id", as_index=False)
        .agg(
            split=("split", "first"),
            seed=("seed", "first"),
            particle_count=("particle_count", "first"),
            snapshots=("snapshot_id", "nunique"),
            rows=("body_id", "size"),
        )
        .sort_values("trajectory_id")
    )
    return prepared, manifest


def write_splits(
    prepared: pd.DataFrame,
    manifest: pd.DataFrame,
    output_dir: Path,
    overwrite: bool,
) -> None:
    paths = {
        "train": output_dir / "train.csv",
        "validation": output_dir / "validation.csv",
        "test": output_dir / "test.csv",
    }
    manifest_path = output_dir / "manifest.csv"
    existing = [path for path in [*paths.values(), manifest_path] if path.exists()]
    if existing and not overwrite:
        raise ValueError(
            "processed output exists; use --overwrite: "
            + ", ".join(str(path) for path in existing)
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    for split, path in paths.items():
        prepared[prepared["split"] == split].to_csv(path, index=False)
    manifest.to_csv(manifest_path, index=False)


def main() -> int:
    arguments = parse_arguments()
    try:
        data = load_and_validate(arguments.input)
        prepared, manifest = split_trajectories(
            data,
            arguments.validation_fraction,
            arguments.test_fraction,
            arguments.split_seed,
        )
        write_splits(
            prepared,
            manifest,
            arguments.output_dir,
            arguments.overwrite,
        )

        print("Trajectory split summary:")
        print(
            manifest.groupby("split")
            .agg(trajectories=("trajectory_id", "size"), rows=("rows", "sum"))
            .to_string()
        )
        print(f"\nManifest written to: {arguments.output_dir / 'manifest.csv'}")
        return 0
    except (OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
