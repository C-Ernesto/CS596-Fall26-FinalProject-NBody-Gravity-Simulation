"""Validate raw CUDA shards and convert them to a model-ready Parquet dataset."""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW_MANIFEST = PROJECT_ROOT / "data" / "raw" / "candidate_v1" / "manifest.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "candidate_v1"

KEY_COLUMNS = ["trajectory_id", "snapshot_id", "body_id"]
NUMERIC_COLUMNS = [
    "trajectory_id", "snapshot_id", "time_step", "time", "dt", "body_id",
    "particle_count", "seed", "characteristic_radius", "total_mass", "epsilon",
    "position_scale", "velocity_scale", "cluster_separation", "cluster_speed",
    "disk_thickness", "x", "y", "z", "vx", "vy", "vz", "mass", "ax", "ay", "az",
]
MODEL_COLUMNS = ["x", "y", "z", "vx", "vy", "vz", "mass", "ax", "ay", "az", "epsilon"]


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare the candidate full dataset.")
    parser.add_argument("--raw-manifest", type=Path, default=DEFAULT_RAW_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def normalized_arrays(frame: pd.DataFrame) -> dict[str, np.ndarray]:
    radius = frame["characteristic_radius"].to_numpy(dtype=np.float64)
    total_mass = frame["total_mass"].to_numpy(dtype=np.float64)
    velocity_scale = np.sqrt(total_mass / radius)
    acceleration_scale = total_mass / np.square(radius)
    return {
        "x": frame["x"].to_numpy(dtype=np.float64) / radius,
        "y": frame["y"].to_numpy(dtype=np.float64) / radius,
        "z": frame["z"].to_numpy(dtype=np.float64) / radius,
        "vx": frame["vx"].to_numpy(dtype=np.float64) / velocity_scale,
        "vy": frame["vy"].to_numpy(dtype=np.float64) / velocity_scale,
        "vz": frame["vz"].to_numpy(dtype=np.float64) / velocity_scale,
        "mass": frame["mass"].to_numpy(dtype=np.float64) / total_mass,
        "ax": frame["ax"].to_numpy(dtype=np.float64) / acceleration_scale,
        "ay": frame["ay"].to_numpy(dtype=np.float64) / acceleration_scale,
        "az": frame["az"].to_numpy(dtype=np.float64) / acceleration_scale,
        "epsilon": frame["epsilon"].to_numpy(dtype=np.float64) / radius,
    }


def validate_shard(frame: pd.DataFrame, manifest_row: pd.Series, source: Path) -> None:
    required = set(NUMERIC_COLUMNS + ["initial_condition"])
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"{source} is missing columns: {', '.join(missing)}")
    if len(frame) != int(manifest_row["rows"]):
        raise ValueError(f"{source} has {len(frame)} rows; expected {manifest_row['rows']}")
    if frame[KEY_COLUMNS].duplicated().any():
        raise ValueError(f"{source} contains duplicate trajectory/snapshot/body keys")
    numeric = frame[NUMERIC_COLUMNS].to_numpy(dtype=np.float64)
    if not np.isfinite(numeric).all():
        raise ValueError(f"{source} contains NaN or infinite numeric values")
    if (frame["mass"] <= 0).any() or (frame["characteristic_radius"] <= 0).any():
        raise ValueError(f"{source} contains non-positive masses or scales")
    families = set(frame["initial_condition"].unique())
    if families != {manifest_row["family"]}:
        raise ValueError(f"{source} family values {families} do not match its manifest")

    grouped = frame.groupby(["trajectory_id", "snapshot_id"], sort=False)
    sizes = grouped.size()
    expected_sizes = grouped["particle_count"].first()
    if not sizes.equals(expected_sizes.astype(sizes.dtype)):
        raise ValueError(f"{source} has an incomplete snapshot")
    body_checks = grouped["body_id"].agg(["min", "max", "nunique"])
    if not (
        (body_checks["min"] == 0)
        & (body_checks["max"] == sizes.to_numpy() - 1)
        & (body_checks["nunique"] == sizes.to_numpy())
    ).all():
        raise ValueError(f"{source} body IDs are not exactly 0 through N-1")


def main() -> int:
    arguments = parse_arguments()
    try:
        arguments.raw_manifest = arguments.raw_manifest.resolve()
        arguments.output_dir = arguments.output_dir.resolve()
        if not arguments.raw_manifest.is_file():
            raise ValueError(f"raw manifest does not exist: {arguments.raw_manifest}")
        manifest = pd.read_csv(arguments.raw_manifest)
        if manifest.empty:
            raise ValueError("raw manifest is empty")

        output_dir = arguments.output_dir
        if output_dir.exists() and any(output_dir.iterdir()):
            if not arguments.overwrite:
                raise ValueError(f"output directory is not empty: {output_dir}; pass --overwrite")
            shutil.rmtree(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        statistics = {
            column: {"count": 0, "sum": 0.0, "sum_squares": 0.0}
            for column in MODEL_COLUMNS
        }
        snapshot_frames: list[pd.DataFrame] = []
        processed_rows: list[dict] = []

        for number, row in manifest.iterrows():
            source = PROJECT_ROOT / row["raw_path"]
            print(f"[{number + 1}/{len(manifest)}] Validating {row['shard_id']}")
            frame = pd.read_csv(source)
            validate_shard(frame, row, source)
            frame.insert(0, "trajectory_uid", row["shard_id"] + ":" + frame["trajectory_id"].astype(str))
            frame.insert(0, "shard_id", row["shard_id"])
            frame.insert(0, "evaluation_group", row["evaluation_group"])
            frame.insert(0, "split", row["split"])

            relative_parquet = Path(row["split"]) / f"{row['shard_id']}.parquet"
            parquet_path = output_dir / relative_parquet
            parquet_path.parent.mkdir(parents=True, exist_ok=True)
            frame.to_parquet(parquet_path, engine="pyarrow", compression="zstd", index=False)

            index_columns = [
                "split", "evaluation_group", "shard_id", "trajectory_uid",
                "trajectory_id", "snapshot_id", "time_step", "time", "particle_count",
                "seed", "initial_condition", "characteristic_radius", "total_mass", "epsilon",
            ]
            snapshot_index = frame[index_columns].drop_duplicates(
                subset=["trajectory_uid", "snapshot_id"]
            )
            snapshot_index = snapshot_index.copy()
            snapshot_index["parquet_path"] = relative_parquet.as_posix()
            snapshot_frames.append(snapshot_index)

            if row["split"] == "train":
                for column, values in normalized_arrays(frame).items():
                    statistics[column]["count"] += len(values)
                    statistics[column]["sum"] += float(values.sum(dtype=np.float64))
                    statistics[column]["sum_squares"] += float(np.square(values).sum(dtype=np.float64))

            processed_rows.append({
                **row.to_dict(),
                "processed_path": relative_parquet.as_posix(),
                "processed_bytes": parquet_path.stat().st_size,
            })

        snapshot_index = pd.concat(snapshot_frames, ignore_index=True)
        if snapshot_index[["trajectory_uid", "snapshot_id"]].duplicated().any():
            raise ValueError("snapshot index contains duplicate keys")
        snapshot_index.to_parquet(output_dir / "snapshot_index.parquet", index=False, compression="zstd")
        pd.DataFrame(processed_rows).to_csv(output_dir / "manifest.csv", index=False)

        empirical = {}
        for column, values in statistics.items():
            count = values["count"]
            mean = values["sum"] / count
            variance = max(0.0, values["sum_squares"] / count - mean * mean)
            empirical[column] = {"mean": mean, "std": math.sqrt(variance), "count": count}
        normalization = {
            "fit_split": "train",
            "gravitational_constant": 1.0,
            "physical_scaling": {
                "position": "position / characteristic_radius",
                "velocity": "velocity / sqrt(total_mass / characteristic_radius)",
                "mass": "mass / total_mass",
                "acceleration": "acceleration / (total_mass / characteristic_radius**2)",
                "epsilon": "epsilon / characteristic_radius",
            },
            "empirical_after_physical_scaling": empirical,
        }
        (output_dir / "normalization.json").write_text(
            json.dumps(normalization, indent=2), encoding="utf-8"
        )
        metadata = {
            "status": "pre_serial_validation",
            "serial_validated": False,
            "raw_manifest": str(arguments.raw_manifest.relative_to(PROJECT_ROOT)),
            "rows": int(manifest["rows"].sum()),
            "trajectories": int(manifest["trajectories"].sum()),
            "snapshots": int(manifest["snapshots"].sum()),
            "splits": snapshot_index.groupby("split").size().to_dict(),
        }
        (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        print(f"Prepared dataset written to: {output_dir}")
        print(f"Rows: {metadata['rows']:,}; snapshots: {metadata['snapshots']:,}")
        return 0
    except (OSError, ValueError, KeyError, pd.errors.ParserError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
