"""Generate manifest-driven, sharded CUDA force-label data."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "ml" / "dataset_matrix.json"
DEFAULT_GENERATOR = (
    PROJECT_ROOT
    / "build-cuda-explicit"
    / "Release"
    / "nbody_generate_dataset.exe"
)
DEFAULT_RAW_DIR = PROJECT_ROOT / "data" / "raw" / "candidate_v1"
DEFAULT_MANIFEST = DEFAULT_RAW_DIR / "manifest.csv"


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate the candidate full dataset.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--generator", type=Path, default=DEFAULT_GENERATOR)
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--only-split", choices=["train", "validation", "test"])
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_config(path: Path) -> dict:
    if not path.is_file():
        raise ValueError(f"dataset matrix does not exist: {path}")
    with path.open(encoding="utf-8") as config_file:
        config = json.load(config_file)
    if "global" not in config or not config.get("shards"):
        raise ValueError("dataset matrix requires global settings and shards")
    return config


def validate_seed_ranges(shards: list[dict]) -> None:
    ranges: list[tuple[int, int, str]] = []
    for shard in shards:
        trajectory_count = len(shard["body_counts"]) * shard["trajectories_per_count"]
        start = int(shard["seed_start"])
        end = start + trajectory_count - 1
        name = f"{shard['split']}/{shard['evaluation_group']}/{shard['family']}"
        for other_start, other_end, other_name in ranges:
            if start <= other_end and other_start <= end:
                raise ValueError(f"seed ranges overlap: {name} and {other_name}")
        ranges.append((start, end, name))


def shard_id(shard: dict) -> str:
    return "__".join(
        [shard["split"], shard["evaluation_group"], shard["family"]]
    )


def expected_counts(shard: dict, snapshots: int) -> tuple[int, int, int]:
    trajectories = len(shard["body_counts"]) * shard["trajectories_per_count"]
    snapshot_count = trajectories * snapshots
    rows = (
        sum(shard["body_counts"])
        * shard["trajectories_per_count"]
        * snapshots
    )
    return trajectories, snapshot_count, rows


def count_csv_rows(path: Path) -> int:
    with path.open("rb") as csv_file:
        line_count = sum(chunk.count(b"\n") for chunk in iter(
            lambda: csv_file.read(1024 * 1024),
            b"",
        ))
    return max(0, line_count - 1)


def generator_command(
    executable: Path,
    output: Path,
    shard: dict,
    global_config: dict,
    overwrite: bool,
) -> list[str]:
    parameters = global_config["initial_condition_parameters"]
    command = [
        str(executable),
        "--body-counts",
        ",".join(str(value) for value in shard["body_counts"]),
        "--trajectories-per-count",
        str(shard["trajectories_per_count"]),
        "--snapshots",
        str(global_config["snapshots"]),
        "--stride",
        str(global_config["stride"]),
        "--dt",
        str(global_config["dt"]),
        "--epsilon",
        str(global_config["epsilon"]),
        "--base-seed",
        str(shard["seed_start"]),
        "--block-size",
        str(global_config["block_size"]),
        "--verify-every",
        str(global_config["verify_every"]),
        "--verification-tolerance",
        str(global_config["verification_tolerance"]),
        "--initial-condition",
        shard["family"],
        "--position-scale",
        str(parameters["position_scale"]),
        "--velocity-scale",
        str(parameters["velocity_scale"]),
        "--cluster-separation",
        str(parameters["cluster_separation"]),
        "--cluster-speed",
        str(parameters["cluster_speed"]),
        "--disk-thickness",
        str(parameters["disk_thickness"]),
        "--output",
        str(output),
    ]
    if overwrite:
        command.append("--overwrite")
    return command


def main() -> int:
    arguments = parse_arguments()
    try:
        arguments.config = arguments.config.resolve()
        arguments.generator = arguments.generator.resolve()
        arguments.raw_dir = arguments.raw_dir.resolve()
        arguments.manifest = arguments.manifest.resolve()
        config = load_config(arguments.config)
        shards = [
            shard
            for shard in config["shards"]
            if arguments.only_split is None or shard["split"] == arguments.only_split
        ]
        validate_seed_ranges(config["shards"])
        global_config = config["global"]

        total_rows = 0
        print("Candidate dataset plan:")
        for shard in shards:
            trajectories, snapshots, rows = expected_counts(
                shard,
                global_config["snapshots"],
            )
            total_rows += rows
            print(
                f"  {shard_id(shard)}: trajectories={trajectories}, "
                f"snapshots={snapshots}, rows={rows:,}"
            )
        print(f"Total planned body rows: {total_rows:,}")
        if arguments.dry_run:
            return 0
        if not arguments.generator.is_file():
            raise ValueError(f"dataset generator does not exist: {arguments.generator}")

        arguments.raw_dir.mkdir(parents=True, exist_ok=True)
        manifest_rows = []
        for index, shard in enumerate(shards, start=1):
            identifier = shard_id(shard)
            output_path = arguments.raw_dir / f"{identifier}.csv"
            trajectories, snapshots, expected_rows = expected_counts(
                shard,
                global_config["snapshots"],
            )
            print(f"\n[{index}/{len(shards)}] Generating {identifier}")
            subprocess.run(
                generator_command(
                    arguments.generator,
                    output_path,
                    shard,
                    global_config,
                    arguments.overwrite,
                ),
                check=True,
            )
            actual_rows = count_csv_rows(output_path)
            if actual_rows != expected_rows:
                raise ValueError(
                    f"{identifier} has {actual_rows} rows; expected {expected_rows}"
                )
            manifest_rows.append(
                {
                    "shard_id": identifier,
                    "split": shard["split"],
                    "evaluation_group": shard["evaluation_group"],
                    "family": shard["family"],
                    "body_counts": ";".join(map(str, shard["body_counts"])),
                    "trajectories": trajectories,
                    "snapshots": snapshots,
                    "rows": actual_rows,
                    "seed_start": shard["seed_start"],
                    "seed_end": shard["seed_start"] + trajectories - 1,
                    "label_backend": "tiled_cuda_fp64_verified_naive",
                    "serial_validated": "false",
                    "raw_path": str(output_path.relative_to(PROJECT_ROOT)),
                }
            )

        arguments.manifest.parent.mkdir(parents=True, exist_ok=True)
        with arguments.manifest.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=manifest_rows[0].keys())
            writer.writeheader()
            writer.writerows(manifest_rows)

        metadata = {
            "dataset_name": config["dataset_name"],
            "status": config["status"],
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "serial_validated": False,
            "matrix_config": str(arguments.config.relative_to(PROJECT_ROOT)),
            "manifest": str(arguments.manifest.relative_to(PROJECT_ROOT)),
            "global": global_config,
        }
        metadata_path = arguments.raw_dir / "metadata.json"
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        print(f"\nRaw manifest written to: {arguments.manifest}")
        print(f"Metadata written to: {metadata_path}")
        return 0
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
