"""Run a paired large-N CUDA force and sparse learned-force crossover study."""

from __future__ import annotations

import argparse
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

from ml.benchmark_hierarchical import load_model, median_summary, run_once
from ml.train_baseline import choose_device, parse_body_counts


DEFAULT_GENERATOR = PROJECT_ROOT / "build-cuda-explicit" / "Release" / "nbody_generate_dataset.exe"
DEFAULT_EXACT = PROJECT_ROOT / "build-cuda-explicit" / "Release" / "nbody_benchmark_force.exe"
DEFAULT_CHECKPOINT = PROJECT_ROOT / "results" / "ml_hierarchical_geometry" / "best_model.pt"
DEFAULT_RAW = PROJECT_ROOT / "data" / "raw" / "large_n_crossover.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "results" / "large_n_crossover"
DEFAULT_FIGURE = PROJECT_ROOT / "visualization" / "output" / "large_n_crossover.png"


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark exact tiled CUDA and sparse learned force on matching large-N states."
    )
    parser.add_argument("--body-counts", type=parse_body_counts, default=[4096, 8192, 16384])
    parser.add_argument("--base-seed", type=int, default=600000)
    parser.add_argument("--generator", type=Path, default=DEFAULT_GENERATOR)
    parser.add_argument("--exact-benchmark", type=Path, default=DEFAULT_EXACT)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--raw-snapshots", type=Path, default=DEFAULT_RAW)
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


def generate_snapshots(arguments: argparse.Namespace) -> None:
    if arguments.raw_snapshots.exists() and not arguments.regenerate:
        return
    if not arguments.generator.is_file():
        raise FileNotFoundError(f"dataset generator is missing: {arguments.generator}")
    arguments.raw_snapshots.parent.mkdir(parents=True, exist_ok=True)
    command = [
        str(arguments.generator),
        "--body-counts", ",".join(map(str, arguments.body_counts)),
        "--trajectories-per-count", "1",
        "--snapshots", "1",
        "--stride", "1",
        "--dt", "0.001",
        "--epsilon", "0.01",
        "--base-seed", str(arguments.base_seed),
        "--block-size", str(arguments.block_size),
        "--verify-every", "0",
        "--initial-condition", "uniform_cube",
        "--output", str(arguments.raw_snapshots),
        "--overwrite",
    ]
    subprocess.run(command, check=True)


def run_exact_benchmark(arguments: argparse.Namespace, output: Path) -> None:
    if not arguments.exact_benchmark.is_file():
        raise FileNotFoundError(f"force benchmark is missing: {arguments.exact_benchmark}")
    subprocess.run(
        [
            str(arguments.exact_benchmark),
            "--body-counts", ",".join(map(str, arguments.body_counts)),
            "--base-seed", str(arguments.base_seed),
            "--repetitions", str(arguments.repetitions),
            "--epsilon", "0.01",
            "--block-size", str(arguments.block_size),
            "--initial-condition", "uniform_cube",
            "--output", str(output),
        ],
        check=True,
    )


def samples_from_csv(path: Path, body_counts: list[int]) -> list[dict]:
    frame = pd.read_csv(path)
    samples: list[dict] = []
    for count in body_counts:
        rows = frame[frame["particle_count"] == count].sort_values("body_id")
        if len(rows) != count:
            raise ValueError(f"expected {count} rows for N={count}, found {len(rows)}")
        radius = float(rows["characteristic_radius"].iloc[0])
        total_mass = float(rows["total_mass"].iloc[0])
        acceleration_scale = total_mass / radius**2
        positions = torch.tensor(rows[["x", "y", "z"]].to_numpy(), dtype=torch.float32)
        masses = torch.tensor(rows["mass"].to_numpy(), dtype=torch.float32).reshape(-1, 1)
        accelerations = torch.tensor(rows[["ax", "ay", "az"]].to_numpy(), dtype=torch.float32)
        positions /= radius
        masses /= total_mass
        accelerations /= acceleration_scale
        samples.append(
            {
                "positions": positions,
                "masses": masses,
                "accelerations": accelerations,
                "epsilon": torch.tensor(float(rows["epsilon"].iloc[0]) / radius),
                "particle_count": count,
                "metadata": {
                    "split": "large_n_benchmark",
                    "evaluation_group": "extrapolation_count",
                    "initial_condition": str(rows["initial_condition"].iloc[0]),
                    "trajectory_id": str(int(rows["trajectory_id"].iloc[0])),
                    "time_step": int(rows["time_step"].iloc[0]),
                },
            }
        )
    return samples


def exact_summary(path: Path) -> pd.DataFrame:
    rows = pd.read_csv(path)
    metrics = ["allocation_ms", "h2d_ms", "kernel_ms", "d2h_ms", "cleanup_ms", "total_ms"]
    result = rows.groupby(["bodies", "seed"], as_index=False)[metrics].median()
    return result.rename(columns={"bodies": "particle_count", "total_ms": "exact_total_ms"})


def paired_summary(learned: pd.DataFrame, exact: pd.DataFrame) -> pd.DataFrame:
    paired = learned.merge(exact, on="particle_count", validate="one_to_one")
    paired = paired.rename(columns={"end_to_end_ms": "learned_total_ms"})
    paired["learned_over_exact"] = paired["learned_total_ms"] / paired["exact_total_ms"]
    paired["exact_over_learned"] = paired["exact_total_ms"] / paired["learned_total_ms"]
    return paired


def plot_results(summary: pd.DataFrame, destination: Path) -> None:
    sns.set_theme(style="whitegrid")
    figure, axes = plt.subplots(1, 3, figsize=(16, 5))
    runtime = summary[["particle_count", "learned_total_ms", "exact_total_ms", "kernel_ms"]].melt(
        id_vars="particle_count", var_name="method", value_name="milliseconds"
    )
    labels = {
        "learned_total_ms": "Sparse ML end-to-end",
        "exact_total_ms": "Tiled CUDA force end-to-end",
        "kernel_ms": "Tiled CUDA kernel only",
    }
    runtime["method"] = runtime["method"].map(labels)
    sns.lineplot(data=runtime, x="particle_count", y="milliseconds", hue="method", marker="o", ax=axes[0])
    axes[0].set(xscale="log", yscale="log", xlabel="Bodies", ylabel="Median time (ms)")
    axes[0].set_title("Paired force-evaluation runtime")

    sns.lineplot(data=summary, x="particle_count", y="learned_over_exact", marker="o", ax=axes[1])
    axes[1].axhline(1.0, color="black", linestyle="--", linewidth=1)
    axes[1].set(xscale="log", yscale="log", xlabel="Bodies", ylabel="Sparse ML / exact time")
    axes[1].set_title("Crossover ratio (<1 means ML wins)")

    axes[2].plot(summary["particle_count"], summary["interaction_ratio"], "o-", color="tab:blue")
    axes[2].set_xscale("log")
    axes[2].set_yscale("log")
    axes[2].set_xlabel("Bodies")
    axes[2].set_ylabel("Retained interaction ratio", color="tab:blue")
    error_axis = axes[2].twinx()
    error_axis.plot(summary["particle_count"], summary["relative_l2"], "s--", color="tab:red")
    error_axis.set_yscale("log")
    error_axis.set_ylabel("Relative L2 force error", color="tab:red")
    axes[2].set_title("Work and accuracy")
    figure.tight_layout()
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=180, bbox_inches="tight")
    plt.close(figure)


def main() -> int:
    arguments = parse_arguments()
    try:
        if arguments.base_seed < 0 or arguments.theta <= 0.0:
            raise ValueError("base-seed must be nonnegative and theta must be positive")
        if arguments.warmups < 0 or arguments.repetitions <= 0:
            raise ValueError("invalid warmup or repetition count")
        arguments.output_dir.mkdir(parents=True, exist_ok=True)
        generate_snapshots(arguments)
        exact_path = arguments.output_dir / "exact_per_run.csv"
        run_exact_benchmark(arguments, exact_path)

        device = choose_device(arguments.device)
        if device.type != "cuda":
            raise RuntimeError("large-N crossover benchmark requires CUDA")
        model, checkpoint = load_model(
            arguments.checkpoint,
            device,
            arguments.theta,
            arguments.leaf_capacity,
            "cuda",
        )
        samples = samples_from_csv(arguments.raw_snapshots, arguments.body_counts)
        records: list[dict[str, float | int | str]] = []
        with torch.inference_mode():
            for sample in samples:
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
                            "repetition": repetition,
                            "rmse": (squared_error / (3 * count)) ** 0.5,
                            "relative_l2": (squared_error / max(target_squared, 1.0e-30)) ** 0.5,
                        }
                    )
                    records.append(row)
                print(f"benchmarked sparse learned force at N={sample['particle_count']}")

        learned_rows = pd.DataFrame(records)
        learned = median_summary(learned_rows)
        exact = exact_summary(exact_path)
        paired = paired_summary(learned, exact)
        learned_rows.to_csv(arguments.output_dir / "learned_per_run.csv", index=False)
        learned.to_csv(arguments.output_dir / "learned_summary.csv", index=False)
        exact.to_csv(arguments.output_dir / "exact_summary.csv", index=False)
        paired.to_csv(arguments.output_dir / "paired_summary.csv", index=False)
        plot_results(paired, arguments.figure)
        configuration = {
            "body_counts": arguments.body_counts,
            "base_seed": arguments.base_seed,
            "theta": arguments.theta,
            "leaf_capacity": arguments.leaf_capacity,
            "block_size": arguments.block_size,
            "warmups": arguments.warmups,
            "repetitions": arguments.repetitions,
            "initial_condition": "uniform_cube",
            "checkpoint": str(arguments.checkpoint),
            "checkpoint_epoch": checkpoint.get("epoch"),
            "raw_snapshots": str(arguments.raw_snapshots),
            "device": str(device),
            "gpu": torch.cuda.get_device_name(device),
            "torch_version": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "platform": platform.platform(),
            "comparison_scope": "same initial state and one force evaluation",
        }
        (arguments.output_dir / "config.json").write_text(
            json.dumps(configuration, indent=2), encoding="utf-8"
        )
        print(paired.to_string(index=False, float_format=lambda value: f"{value:.6g}"))
        print(f"wrote paired benchmark to {arguments.output_dir}")
        return 0
    except (FileNotFoundError, KeyError, OSError, RuntimeError, ValueError, subprocess.CalledProcessError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
