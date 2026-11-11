"""Explore and sanity-check the processed candidate N-body dataset."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MATPLOTLIB_CONFIG = PROJECT_ROOT / ".matplotlib"
MATPLOTLIB_CONFIG.mkdir(exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MATPLOTLIB_CONFIG))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


DEFAULT_DATASET = PROJECT_ROOT / "data" / "processed" / "candidate_v1"
DEFAULT_OUTPUT = PROJECT_ROOT / "results" / "dataset_analysis"
DEFAULT_FIGURES = PROJECT_ROOT / "visualization" / "output" / "dataset_analysis"
FAMILY_ORDER = [
    "uniform_cube",
    "plummer_sphere",
    "two_cluster_collision",
    "rotating_disk",
]


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze the processed candidate N-body dataset."
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--figure-dir", type=Path, default=DEFAULT_FIGURES)
    parser.add_argument(
        "--sample-per-shard",
        type=int,
        default=25000,
        help="Maximum body rows sampled from each shard for plots.",
    )
    return parser.parse_args()


def vector_magnitude(frame: pd.DataFrame, columns: list[str]) -> np.ndarray:
    return np.sqrt(np.square(frame[columns].to_numpy(dtype=np.float64)).sum(axis=1))


def add_derived_columns(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    radius = frame["characteristic_radius"].to_numpy(dtype=np.float64)
    total_mass = frame["total_mass"].to_numpy(dtype=np.float64)
    acceleration_scale = total_mass / np.square(radius)
    velocity_scale = np.sqrt(total_mass / radius)

    frame["position_magnitude"] = vector_magnitude(frame, ["x", "y", "z"])
    frame["velocity_magnitude"] = vector_magnitude(frame, ["vx", "vy", "vz"])
    frame["acceleration_magnitude"] = vector_magnitude(frame, ["ax", "ay", "az"])
    frame["normalized_position_magnitude"] = frame["position_magnitude"] / radius
    frame["normalized_velocity_magnitude"] = frame["velocity_magnitude"] / velocity_scale
    frame["normalized_acceleration_magnitude"] = (
        frame["acceleration_magnitude"] / acceleration_scale
    )
    frame["normalized_epsilon"] = frame["epsilon"] / radius
    return frame


def snapshot_quality(frame: pd.DataFrame) -> pd.DataFrame:
    work = frame[
        [
            "trajectory_uid", "snapshot_id", "split", "evaluation_group",
            "initial_condition", "particle_count", "seed", "time_step",
            "characteristic_radius", "mass", "x", "y", "z", "vx", "vy",
            "vz", "ax", "ay", "az",
        ]
    ].copy()
    for axis in "xyz":
        work[f"m_{axis}"] = work["mass"] * work[axis]
        work[f"m_v{axis}"] = work["mass"] * work[f"v{axis}"]
        work[f"m_a{axis}"] = work["mass"] * work[f"a{axis}"]
    work["m_velocity_magnitude"] = work["mass"] * vector_magnitude(
        work, ["vx", "vy", "vz"]
    )
    work["m_acceleration_magnitude"] = work["mass"] * vector_magnitude(
        work, ["ax", "ay", "az"]
    )

    keys = [
        "trajectory_uid", "snapshot_id", "split", "evaluation_group",
        "initial_condition", "particle_count", "seed", "time_step",
        "characteristic_radius",
    ]
    sums = work.groupby(keys, as_index=False).agg(
        total_mass=("mass", "sum"),
        mx=("m_x", "sum"),
        my=("m_y", "sum"),
        mz=("m_z", "sum"),
        momentum_x=("m_vx", "sum"),
        momentum_y=("m_vy", "sum"),
        momentum_z=("m_vz", "sum"),
        net_force_x=("m_ax", "sum"),
        net_force_y=("m_ay", "sum"),
        net_force_z=("m_az", "sum"),
        momentum_scale=("m_velocity_magnitude", "sum"),
        force_scale=("m_acceleration_magnitude", "sum"),
    )
    sums["center_of_mass_residual"] = (
        np.sqrt(np.square(sums[["mx", "my", "mz"]]).sum(axis=1))
        / sums["total_mass"]
        / sums["characteristic_radius"]
    )
    momentum_norm = np.sqrt(
        np.square(sums[["momentum_x", "momentum_y", "momentum_z"]]).sum(axis=1)
    )
    force_norm = np.sqrt(
        np.square(sums[["net_force_x", "net_force_y", "net_force_z"]]).sum(axis=1)
    )
    sums["relative_momentum_residual"] = momentum_norm / sums["momentum_scale"].clip(
        lower=np.finfo(np.float64).tiny
    )
    sums["relative_net_force_residual"] = force_norm / sums["force_scale"].clip(
        lower=np.finfo(np.float64).tiny
    )
    return sums


def minimum_pair_distance(positions: np.ndarray, block_size: int = 512) -> float:
    minimum_squared = np.inf
    count = len(positions)
    for first_start in range(0, count, block_size):
        first = positions[first_start : first_start + block_size]
        for second_start in range(first_start, count, block_size):
            second = positions[second_start : second_start + block_size]
            differences = first[:, None, :] - second[None, :, :]
            squared = np.einsum("ijk,ijk->ij", differences, differences)
            if first_start == second_start:
                np.fill_diagonal(squared, np.inf)
            candidate = float(squared.min())
            minimum_squared = min(minimum_squared, candidate)
    return float(np.sqrt(minimum_squared))


def close_encounter_rows(frame: pd.DataFrame) -> list[dict]:
    rows: list[dict] = []
    grouping = ["split", "evaluation_group", "initial_condition", "particle_count"]
    representatives = (
        frame.groupby(grouping, as_index=False)[["trajectory_uid", "snapshot_id"]]
        .first()
    )
    representatives["selection"] = "representative"
    maximum_indices = frame.groupby(grouping)[
        "normalized_acceleration_magnitude"
    ].idxmax()
    maximum_snapshots = frame.loc[
        maximum_indices,
        grouping + ["trajectory_uid", "snapshot_id"],
    ].copy()
    maximum_snapshots["selection"] = "maximum_acceleration"
    selected = pd.concat([representatives, maximum_snapshots], ignore_index=True)
    selected = selected.drop_duplicates(
        subset=["trajectory_uid", "snapshot_id"], keep="last"
    )

    for representative in selected.itertuples(index=False):
        snapshot = frame[
            (frame["trajectory_uid"] == representative.trajectory_uid)
            & (frame["snapshot_id"] == representative.snapshot_id)
        ]
        minimum_distance = minimum_pair_distance(
            snapshot[["x", "y", "z"]].to_numpy(dtype=np.float64)
        )
        epsilon = float(snapshot["epsilon"].iloc[0])
        rows.append(
            {
                "split": representative.split,
                "evaluation_group": representative.evaluation_group,
                "initial_condition": representative.initial_condition,
                "particle_count": representative.particle_count,
                "trajectory_uid": representative.trajectory_uid,
                "snapshot_id": representative.snapshot_id,
                "selection": representative.selection,
                "minimum_pair_distance": minimum_distance,
                "epsilon": epsilon,
                "minimum_distance_over_epsilon": minimum_distance / epsilon,
                "maximum_normalized_acceleration": float(
                    snapshot["normalized_acceleration_magnitude"].max()
                ),
            }
        )
    return rows


def distribution_summary(frame: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "normalized_position_magnitude",
        "normalized_velocity_magnitude",
        "normalized_acceleration_magnitude",
        "normalized_epsilon",
    ]
    groups = ["split", "evaluation_group", "initial_condition", "particle_count"]
    rows: list[dict] = []
    for keys, group in frame.groupby(groups, sort=False):
        base = dict(zip(groups, keys))
        for metric in metrics:
            values = group[metric].to_numpy(dtype=np.float64)
            quantiles = np.quantile(values, [0.0, 0.5, 0.9, 0.99, 0.999, 1.0])
            rows.append(
                {
                    **base,
                    "metric": metric,
                    "count": len(values),
                    "mean": float(values.mean()),
                    "std": float(values.std()),
                    "min": quantiles[0],
                    "p50": quantiles[1],
                    "p90": quantiles[2],
                    "p99": quantiles[3],
                    "p999": quantiles[4],
                    "max": quantiles[5],
                }
            )
    return pd.DataFrame(rows)


def create_plots(
    sampled: pd.DataFrame,
    quality: pd.DataFrame,
    close_encounters: pd.DataFrame,
    snapshot_index: pd.DataFrame,
    figure_dir: Path,
) -> None:
    figure_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="notebook")

    counts = (
        snapshot_index.groupby(["split", "initial_condition"], as_index=False)
        .size()
        .rename(columns={"size": "snapshots"})
    )
    figure, axis = plt.subplots(figsize=(10, 5.5))
    sns.barplot(
        data=counts,
        x="initial_condition",
        y="snapshots",
        hue="split",
        order=FAMILY_ORDER,
        ax=axis,
    )
    axis.set_xlabel("Initial-condition family")
    axis.set_ylabel("Complete snapshots")
    axis.set_title("Dataset composition")
    axis.tick_params(axis="x", rotation=20)
    figure.tight_layout()
    figure.savefig(figure_dir / "dataset_composition.png", dpi=200)
    plt.close(figure)

    positive = sampled[sampled["normalized_acceleration_magnitude"] > 0].copy()
    figure, axis = plt.subplots(figsize=(10, 5.5))
    sns.ecdfplot(
        data=positive,
        x="normalized_acceleration_magnitude",
        hue="initial_condition",
        hue_order=FAMILY_ORDER,
        ax=axis,
    )
    axis.set_xscale("log")
    axis.set_xlabel("Normalized acceleration magnitude")
    axis.set_ylabel("Cumulative fraction of body rows")
    axis.set_title("Acceleration-label distributions")
    figure.tight_layout()
    figure.savefig(figure_dir / "normalized_acceleration_ecdf.png", dpi=200)
    plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    for axis, metric, title in [
        (axes[0], "center_of_mass_residual", "Center-of-mass residual"),
        (axes[1], "relative_net_force_residual", "Relative net-force residual"),
    ]:
        sns.boxplot(
            data=quality,
            x="initial_condition",
            y=metric,
            order=FAMILY_ORDER,
            showfliers=False,
            ax=axis,
        )
        axis.set_yscale("log")
        axis.set_xlabel("")
        axis.set_ylabel(metric.replace("_", " "))
        axis.set_title(title)
        axis.tick_params(axis="x", rotation=25)
    figure.suptitle("Snapshot-level physical sanity checks")
    figure.tight_layout()
    figure.savefig(figure_dir / "snapshot_residuals.png", dpi=200)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(9, 5.5))
    sns.scatterplot(
        data=close_encounters,
        x="minimum_distance_over_epsilon",
        y="maximum_normalized_acceleration",
        hue="initial_condition",
        style="initial_condition",
        hue_order=FAMILY_ORDER,
        s=80,
        ax=axis,
    )
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xlabel("Minimum pair distance / epsilon")
    axis.set_ylabel("Maximum normalized acceleration")
    axis.set_title("Representative close encounters")
    figure.tight_layout()
    figure.savefig(figure_dir / "close_encounters.png", dpi=200)
    plt.close(figure)


def write_report(
    output_path: Path,
    metadata: dict,
    distribution: pd.DataFrame,
    quality: pd.DataFrame,
    close_encounters: pd.DataFrame,
) -> None:
    acceleration = distribution[
        distribution["metric"] == "normalized_acceleration_magnitude"
    ]
    family_summary = (
        acceleration.groupby("initial_condition", as_index=False)
        .agg(p50=("p50", "median"), p99=("p99", "max"), maximum=("max", "max"))
        .sort_values("initial_condition")
    )
    family_lines = [
        "| Family | Median p50 | Maximum p99 | Maximum |",
        "|---|---:|---:|---:|",
    ]
    for row in family_summary.itertuples(index=False):
        family_lines.append(
            f"| {row.initial_condition} | {row.p50:.6g} | "
            f"{row.p99:.6g} | {row.maximum:.6g} |"
        )
    report = [
        "# Candidate dataset analysis",
        "",
        "## Dataset size",
        "",
        f"- Body rows: {metadata['rows']:,}",
        f"- Trajectories: {metadata['trajectories']:,}",
        f"- Snapshots: {metadata['snapshots']:,}",
        f"- Status: `{metadata['status']}`",
        f"- Serial validated: `{str(metadata['serial_validated']).lower()}`",
        "",
        "## Normalized acceleration by family",
        "",
        *family_lines,
        "",
        "## Snapshot residual maxima",
        "",
        f"- Center-of-mass residual: {quality['center_of_mass_residual'].max():.6e}",
        f"- Relative momentum residual: {quality['relative_momentum_residual'].max():.6e}",
        f"- Relative net-force residual: {quality['relative_net_force_residual'].max():.6e}",
        "",
        "## Representative close encounters",
        "",
        f"- Smallest sampled distance/epsilon: {close_encounters['minimum_distance_over_epsilon'].min():.6g}",
        f"- Largest sampled normalized acceleration: {close_encounters['maximum_normalized_acceleration'].max():.6g}",
        "",
        "The close-encounter calculation uses one deterministic representative snapshot and",
        "the maximum-acceleration snapshot per split/evaluation/family/particle-count",
        "combination. It is diagnostic, not an exhaustive all-snapshot nearest-neighbor search.",
    ]
    output_path.write_text("\n".join(report) + "\n", encoding="utf-8")


def main() -> int:
    arguments = parse_arguments()
    try:
        dataset = arguments.dataset.resolve()
        output_dir = arguments.output_dir.resolve()
        figure_dir = arguments.figure_dir.resolve()
        if arguments.sample_per_shard <= 0:
            raise ValueError("--sample-per-shard must be positive")
        for required in ["manifest.csv", "metadata.json", "snapshot_index.parquet"]:
            if not (dataset / required).is_file():
                raise ValueError(f"missing processed dataset file: {dataset / required}")

        manifest = pd.read_csv(dataset / "manifest.csv")
        metadata = json.loads((dataset / "metadata.json").read_text(encoding="utf-8"))
        snapshot_index = pd.read_parquet(dataset / "snapshot_index.parquet")
        output_dir.mkdir(parents=True, exist_ok=True)

        sampled_frames: list[pd.DataFrame] = []
        distribution_frames: list[pd.DataFrame] = []
        quality_frames: list[pd.DataFrame] = []
        close_rows: list[dict] = []
        total_rows = 0

        for index, row in manifest.iterrows():
            print(f"[{index + 1}/{len(manifest)}] Analyzing {row['shard_id']}")
            frame = pd.read_parquet(dataset / row["processed_path"])
            if len(frame) != int(row["rows"]):
                raise ValueError(
                    f"{row['shard_id']} has {len(frame)} rows; expected {row['rows']}"
                )
            total_rows += len(frame)
            numeric = frame.select_dtypes(include=[np.number]).to_numpy(dtype=np.float64)
            if not np.isfinite(numeric).all():
                raise ValueError(f"{row['shard_id']} contains non-finite numeric values")

            derived = add_derived_columns(frame)
            distribution_frames.append(distribution_summary(derived))
            quality_frames.append(snapshot_quality(derived))
            close_rows.extend(close_encounter_rows(derived))
            sample_size = min(arguments.sample_per_shard, len(derived))
            sampled_frames.append(
                derived.sample(n=sample_size, random_state=20261111 + index)
            )

        if total_rows != int(metadata["rows"]):
            raise ValueError(f"analyzed {total_rows} rows; metadata says {metadata['rows']}")

        sampled = pd.concat(sampled_frames, ignore_index=True)
        distribution = pd.concat(distribution_frames, ignore_index=True)
        quality = pd.concat(quality_frames, ignore_index=True)
        close_encounters = pd.DataFrame(close_rows)

        distribution.to_csv(output_dir / "distribution_summary.csv", index=False)
        quality.to_csv(output_dir / "snapshot_quality.csv", index=False)
        close_encounters.to_csv(output_dir / "close_encounters.csv", index=False)
        create_plots(sampled, quality, close_encounters, snapshot_index, figure_dir)
        write_report(
            output_dir / "report.md",
            metadata,
            distribution,
            quality,
            close_encounters,
        )

        print(f"Analyzed body rows: {total_rows:,}")
        print(f"Analyzed snapshots: {len(quality):,}")
        print(f"Maximum center-of-mass residual: {quality['center_of_mass_residual'].max():.3e}")
        print(f"Maximum relative net-force residual: {quality['relative_net_force_residual'].max():.3e}")
        print(f"Smallest sampled distance/epsilon: {close_encounters['minimum_distance_over_epsilon'].min():.3e}")
        print(f"Tables and report: {output_dir}")
        print(f"Figures: {figure_dir}")
        return 0
    except (OSError, ValueError, KeyError, pd.errors.ParserError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
