"""Validate the large-N crossover across initial-condition families and seeds."""

from __future__ import annotations

import argparse
import gc
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".matplotlib"))

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch

from ml.benchmark_hierarchical import load_model, run_once
from ml.benchmark_large_n import DEFAULT_CHECKPOINT, DEFAULT_EXACT, DEFAULT_GENERATOR
from ml.train_baseline import choose_device, parse_body_counts


FAMILIES = [
    "uniform_cube",
    "plummer_sphere",
    "two_cluster_collision",
    "rotating_disk",
]
DEFAULT_RAW_DIR = PROJECT_ROOT / "data" / "raw" / "crossover_matrix"
DEFAULT_OUTPUT = PROJECT_ROOT / "results" / "crossover_matrix"
DEFAULT_FIGURE = PROJECT_ROOT / "visualization" / "output" / "crossover_matrix.png"


def parse_families(value: str) -> list[str]:
    values = [item.strip() for item in value.split(",") if item.strip()]
    if not values or any(item not in FAMILIES for item in values):
        raise argparse.ArgumentTypeError(
            "families must be comma-separated values from: " + ", ".join(FAMILIES)
        )
    return values


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Repeat the paired crossover across families and random seeds."
    )
    parser.add_argument("--body-counts", type=parse_body_counts, default=[32768, 65536])
    parser.add_argument("--families", type=parse_families, default=FAMILIES)
    parser.add_argument("--seeds-per-case", type=int, default=3)
    parser.add_argument("--base-seed", type=int, default=700000)
    parser.add_argument("--family-seed-stride", type=int, default=100000)
    parser.add_argument("--generator", type=Path, default=DEFAULT_GENERATOR)
    parser.add_argument("--exact-benchmark", type=Path, default=DEFAULT_EXACT)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--figure", type=Path, default=DEFAULT_FIGURE)
    parser.add_argument("--theta", type=float, default=0.7)
    parser.add_argument("--leaf-capacity", type=int, default=16)
    parser.add_argument("--block-size", type=int, default=64)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--device", choices=["auto", "cuda"], default="auto")
    parser.add_argument("--regenerate", action="store_true")
    return parser.parse_args()


def family_seed(arguments: argparse.Namespace, family_index: int) -> int:
    return arguments.base_seed + family_index * arguments.family_seed_stride


def generate_family(
    arguments: argparse.Namespace,
    family: str,
    family_index: int,
) -> tuple[Path, Path]:
    raw_path = arguments.raw_dir / f"{family}.csv"
    exact_path = arguments.output_dir / f"exact_{family}_per_run.csv"
    seed = family_seed(arguments, family_index)
    counts = ",".join(map(str, arguments.body_counts))
    if arguments.regenerate or not raw_path.exists():
        subprocess.run(
            [
                str(arguments.generator),
                "--body-counts", counts,
                "--trajectories-per-count", str(arguments.seeds_per_case),
                "--snapshots", "1",
                "--stride", "1",
                "--dt", "0.001",
                "--epsilon", "0.01",
                "--base-seed", str(seed),
                "--block-size", str(arguments.block_size),
                "--verify-every", "0",
                "--initial-condition", family,
                "--output", str(raw_path),
                "--overwrite",
            ],
            check=True,
        )
    if arguments.regenerate or not exact_path.exists():
        subprocess.run(
            [
                str(arguments.exact_benchmark),
                "--body-counts", counts,
                "--trajectories-per-count", str(arguments.seeds_per_case),
                "--base-seed", str(seed),
                "--repetitions", str(arguments.repetitions),
                "--epsilon", "0.01",
                "--block-size", str(arguments.block_size),
                "--initial-condition", family,
                "--output", str(exact_path),
            ],
            check=True,
        )
    return raw_path, exact_path


def samples_from_family(path: Path, expected_counts: list[int]) -> list[dict]:
    columns = [
        "initial_condition",
        "particle_count",
        "seed",
        "trajectory_id",
        "time_step",
        "body_id",
        "characteristic_radius",
        "total_mass",
        "epsilon",
        "x",
        "y",
        "z",
        "mass",
        "ax",
        "ay",
        "az",
    ]
    float_columns = ["x", "y", "z", "mass", "ax", "ay", "az"]
    dtypes = {column: "float32" for column in float_columns}
    dtypes.update(
        {
            "initial_condition": "category",
            "particle_count": "int32",
            "seed": "int64",
            "trajectory_id": "int32",
            "time_step": "int32",
            "body_id": "int32",
            "characteristic_radius": "float64",
            "total_mass": "float64",
            "epsilon": "float64",
        }
    )
    frame = pd.read_csv(path, usecols=columns, dtype=dtypes)
    samples: list[dict] = []
    grouped = frame.groupby(
        ["initial_condition", "particle_count", "seed", "trajectory_id"],
        sort=True,
        observed=True,
    )
    for (family, count, seed, trajectory_id), rows in grouped:
        count = int(count)
        if count not in expected_counts:
            continue
        rows = rows.sort_values("body_id")
        if len(rows) != count:
            raise ValueError(f"{family} N={count} seed={seed} has {len(rows)} rows")
        radius = float(rows["characteristic_radius"].iloc[0])
        total_mass = float(rows["total_mass"].iloc[0])
        positions = torch.tensor(rows[["x", "y", "z"]].to_numpy(), dtype=torch.float32)
        masses = torch.tensor(rows["mass"].to_numpy(), dtype=torch.float32).reshape(-1, 1)
        accelerations = torch.tensor(
            rows[["ax", "ay", "az"]].to_numpy(), dtype=torch.float32
        )
        positions /= radius
        masses /= total_mass
        accelerations /= total_mass / radius**2
        samples.append(
            {
                "positions": positions,
                "masses": masses,
                "accelerations": accelerations,
                "epsilon": torch.tensor(float(rows["epsilon"].iloc[0]) / radius),
                "particle_count": count,
                "metadata": {
                    "split": "crossover_matrix",
                    "evaluation_group": "family_seed_crossover",
                    "initial_condition": str(family),
                    "trajectory_id": str(int(trajectory_id)),
                    "time_step": int(rows["time_step"].iloc[0]),
                    "seed": int(seed),
                },
            }
        )
    return samples


def summarize_learned(rows: pd.DataFrame) -> pd.DataFrame:
    keys = ["initial_condition", "particle_count", "seed"]
    metrics = [
        column
        for column in rows.columns
        if column.endswith("_ms")
        or column
        in {
            "body_interactions",
            "aggregate_interactions",
            "total_interactions",
            "direct_interactions",
            "interaction_ratio",
            "relative_l2",
            "rmse",
        }
    ]
    summary = rows.groupby(keys, as_index=False)[metrics].median()
    runs = rows.groupby(keys, as_index=False).size().rename(columns={"size": "runs"})
    return summary.merge(runs, on=keys)


def summarize_exact(rows: pd.DataFrame) -> pd.DataFrame:
    keys = ["initial_condition", "bodies", "seed"]
    metrics = ["allocation_ms", "h2d_ms", "kernel_ms", "d2h_ms", "cleanup_ms", "total_ms"]
    summary = rows.groupby(keys, as_index=False)[metrics].median()
    return summary.rename(
        columns={"bodies": "particle_count", "total_ms": "exact_total_ms"}
    )


def pair_results(learned: pd.DataFrame, exact: pd.DataFrame) -> pd.DataFrame:
    keys = ["initial_condition", "particle_count", "seed"]
    paired = learned.merge(exact, on=keys, validate="one_to_one")
    paired = paired.rename(columns={"end_to_end_ms": "learned_total_ms"})
    paired["learned_over_exact"] = paired["learned_total_ms"] / paired["exact_total_ms"]
    paired["exact_over_learned"] = paired["exact_total_ms"] / paired["learned_total_ms"]
    paired["ml_faster"] = paired["learned_over_exact"] < 1.0
    return paired


def family_summary(paired: pd.DataFrame) -> pd.DataFrame:
    keys = ["initial_condition", "particle_count"]
    metrics = [
        "learned_total_ms",
        "exact_total_ms",
        "learned_over_exact",
        "exact_over_learned",
        "interaction_ratio",
        "relative_l2",
        "rmse",
    ]
    summary = paired.groupby(keys, as_index=False)[metrics].median()
    wins = paired.groupby(keys, as_index=False).agg(
        seeds=("seed", "nunique"),
        ml_win_fraction=("ml_faster", "mean"),
    )
    return summary.merge(wins, on=keys)


def overall_summary(paired: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "learned_total_ms",
        "exact_total_ms",
        "learned_over_exact",
        "exact_over_learned",
        "interaction_ratio",
        "relative_l2",
        "rmse",
    ]
    summary = paired.groupby("particle_count", as_index=False)[metrics].median()
    counts = paired.groupby("particle_count", as_index=False).agg(
        cases=("seed", "size"),
        ml_win_fraction=("ml_faster", "mean"),
    )
    return summary.merge(counts, on="particle_count")


def plot_results(summary: pd.DataFrame, destination: Path) -> None:
    sns.set_theme(style="whitegrid")
    summary = summary.copy()
    labels = {
        "uniform_cube": "Uniform cube",
        "plummer_sphere": "Plummer sphere",
        "two_cluster_collision": "Two-cluster collision",
        "rotating_disk": "Rotating disk",
    }
    summary["family"] = summary["initial_condition"].map(labels)
    figure, axes = plt.subplots(1, 3, figsize=(17, 5))
    sns.lineplot(
        data=summary,
        x="particle_count",
        y="learned_over_exact",
        hue="family",
        marker="o",
        ax=axes[0],
    )
    axes[0].axhline(1.0, color="black", linestyle="--", linewidth=1)
    axes[0].set(
        xscale="log",
        yscale="log",
        xlabel="Bodies",
        ylabel="Sparse ML / exact time",
        title="Crossover by family",
    )
    sns.barplot(
        data=summary,
        x="family",
        y="ml_win_fraction",
        hue="particle_count",
        ax=axes[1],
    )
    axes[1].set_ylim(0.0, 1.05)
    axes[1].tick_params(axis="x", rotation=25)
    axes[1].set(xlabel="Initial condition", ylabel="Fraction of seeds where ML wins")
    axes[1].set_title("Crossover consistency")
    sns.scatterplot(
        data=summary,
        x="interaction_ratio",
        y="relative_l2",
        hue="family",
        style="particle_count",
        s=110,
        ax=axes[2],
    )
    axes[2].set(xscale="log", yscale="log")
    axes[2].set_xlabel("Retained interaction ratio")
    axes[2].set_ylabel("Relative L2 force error")
    axes[2].set_title("Accuracy versus retained work")
    figure.tight_layout()
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=180, bbox_inches="tight")
    plt.close(figure)


def main() -> int:
    arguments = parse_arguments()
    try:
        if arguments.seeds_per_case <= 0 or arguments.repetitions <= 0:
            raise ValueError("seed and repetition counts must be positive")
        if arguments.base_seed < 0 or arguments.family_seed_stride <= 0:
            raise ValueError("invalid seed configuration")
        if arguments.theta <= 0.0 or arguments.warmups < 0:
            raise ValueError("invalid model benchmark configuration")
        if not arguments.generator.is_file() or not arguments.exact_benchmark.is_file():
            raise FileNotFoundError("build the dataset generator and force benchmark first")
        arguments.raw_dir.mkdir(parents=True, exist_ok=True)
        arguments.output_dir.mkdir(parents=True, exist_ok=True)

        raw_paths: list[Path] = []
        exact_frames: list[pd.DataFrame] = []
        for family_index, family in enumerate(arguments.families):
            raw_path, exact_path = generate_family(
                arguments, family, family_index
            )
            raw_paths.append(raw_path)
            exact_frames.append(pd.read_csv(exact_path))

        device = choose_device(arguments.device)
        if device.type != "cuda":
            raise RuntimeError("crossover matrix requires CUDA")
        model, checkpoint = load_model(
            arguments.checkpoint,
            device,
            arguments.theta,
            arguments.leaf_capacity,
            "cuda",
        )
        expected_cases = (
            len(arguments.families)
            * len(arguments.body_counts)
            * arguments.seeds_per_case
        )
        records: list[dict[str, float | int | str | bool]] = []
        case_index = 0
        with torch.inference_mode():
            for raw_path in raw_paths:
                family_samples = samples_from_family(
                    raw_path, arguments.body_counts
                )
                for sample in family_samples:
                    case_index += 1
                    for _ in range(arguments.warmups):
                        run_once(model, sample, device)
                    target = sample["accelerations"]
                    for repetition in range(1, arguments.repetitions + 1):
                        row, prediction = run_once(model, sample, device)
                        difference = prediction - target
                        squared_error = float(difference.square().sum().item())
                        target_squared = float(target.square().sum().item())
                        count = int(sample["particle_count"])
                        row.update(
                            {
                                "seed": int(sample["metadata"]["seed"]),
                                "repetition": repetition,
                                "rmse": (squared_error / (3 * count)) ** 0.5,
                                "relative_l2": (
                                    squared_error / max(target_squared, 1.0e-30)
                                )
                                ** 0.5,
                            }
                        )
                        records.append(row)
                    metadata = sample["metadata"]
                    print(
                        f"[{case_index}/{expected_cases}] "
                        f"{metadata['initial_condition']} "
                        f"N={sample['particle_count']} seed={metadata['seed']}",
                        flush=True,
                    )
                del family_samples
                gc.collect()
        if case_index != expected_cases:
            raise ValueError(f"loaded {case_index} cases; expected {expected_cases}")

        learned_rows = pd.DataFrame(records)
        exact_rows = pd.concat(exact_frames, ignore_index=True)
        learned = summarize_learned(learned_rows)
        exact = summarize_exact(exact_rows)
        paired = pair_results(learned, exact)
        by_family = family_summary(paired)
        overall = overall_summary(paired)
        learned_rows.to_csv(arguments.output_dir / "learned_per_run.csv", index=False)
        exact_rows.to_csv(arguments.output_dir / "exact_per_run.csv", index=False)
        paired.to_csv(arguments.output_dir / "paired_per_case.csv", index=False)
        by_family.to_csv(arguments.output_dir / "family_summary.csv", index=False)
        overall.to_csv(arguments.output_dir / "overall_summary.csv", index=False)
        plot_results(by_family, arguments.figure)

        configuration = {
            "body_counts": arguments.body_counts,
            "families": arguments.families,
            "seeds_per_case": arguments.seeds_per_case,
            "base_seed": arguments.base_seed,
            "family_seed_stride": arguments.family_seed_stride,
            "theta": arguments.theta,
            "leaf_capacity": arguments.leaf_capacity,
            "block_size": arguments.block_size,
            "warmups": arguments.warmups,
            "repetitions": arguments.repetitions,
            "checkpoint": str(arguments.checkpoint),
            "checkpoint_epoch": checkpoint.get("epoch"),
            "device": str(device),
            "gpu": torch.cuda.get_device_name(device),
            "torch_version": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "platform": platform.platform(),
            "comparison_scope": "paired force evaluation by family, count, and seed",
        }
        (arguments.output_dir / "config.json").write_text(
            json.dumps(configuration, indent=2), encoding="utf-8"
        )
        print("\nOverall crossover summary:")
        print(overall.to_string(index=False, float_format=lambda value: f"{value:.6g}"))
        print(f"wrote crossover matrix to {arguments.output_dir}")
        return 0
    except (
        FileNotFoundError,
        KeyError,
        OSError,
        RuntimeError,
        ValueError,
        subprocess.CalledProcessError,
    ) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
