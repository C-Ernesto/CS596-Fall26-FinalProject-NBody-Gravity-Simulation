"""Benchmark the sparse learned-force pipeline by stage and particle count."""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".matplotlib"))

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch

from ml.hierarchical_model import HierarchicalForceModel
from ml.torch_dataset import NBodyTorchDataset
from ml.train_baseline import choose_device, parse_body_counts


DEFAULT_DATASET = PROJECT_ROOT / "data" / "processed" / "candidate_v1"
DEFAULT_CHECKPOINT = PROJECT_ROOT / "results" / "ml_hierarchical_geometry" / "best_model.pt"
DEFAULT_CUDA_RESULTS = PROJECT_ROOT / "results" / "cuda_optimized_scaling.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "results" / "hierarchical_benchmark"
DEFAULT_FIGURE = PROJECT_ROOT / "visualization" / "output" / "hierarchical_benchmark.png"
DEFAULT_COUNTS = [128, 256, 512, 768, 1024, 1536, 2048, 3072]


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Time sparse exact-near/learned-far force prediction by stage."
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--cuda-results", type=Path, default=DEFAULT_CUDA_RESULTS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--figure", type=Path, default=DEFAULT_FIGURE)
    parser.add_argument("--body-counts", type=parse_body_counts, default=DEFAULT_COUNTS)
    parser.add_argument("--theta", type=float, default=0.7)
    parser.add_argument("--leaf-capacity", type=int, default=16)
    parser.add_argument(
        "--backend", choices=["python", "cpp", "auto"], default="auto"
    )
    parser.add_argument(
        "--builder-threads",
        type=int,
        default=0,
        help="OpenMP threads for the compiled builder; 0 uses its runtime default.",
    )
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    return parser.parse_args()


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def load_model(
    path: Path,
    device: torch.device,
    theta: float,
    leaf_capacity: int,
    backend: str,
):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    configuration = checkpoint["configuration"]
    model = HierarchicalForceModel(
        hidden_dim=int(configuration["hidden_dim"]),
        hidden_layers=int(configuration["hidden_layers"]),
        theta=theta,
        leaf_capacity=leaf_capacity,
        interaction_backend=backend,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint


def select_samples(dataset: NBodyTorchDataset, requested: list[int]) -> list[dict]:
    """Choose one deterministic uniform-cube snapshot for every requested count."""
    frame = dataset.index.reset_index(drop=True)
    samples: list[dict] = []
    missing: list[int] = []
    for count in requested:
        candidates = frame[
            (frame["particle_count"].astype(int) == count)
            & (frame["initial_condition"] == "uniform_cube")
        ]
        if candidates.empty:
            candidates = frame[frame["particle_count"].astype(int) == count]
        if candidates.empty:
            missing.append(count)
            continue
        samples.append(dataset[int(candidates.index[0])])
    if missing:
        print(f"Skipping unavailable particle counts: {missing}")
    if not samples:
        raise ValueError("none of the requested particle counts exist in the dataset")
    return samples


def run_once(
    model: HierarchicalForceModel,
    sample: dict,
    device: torch.device,
) -> tuple[dict[str, float | int | str], torch.Tensor]:
    synchronize(device)
    total_started = time.perf_counter()
    transfer_started = time.perf_counter()
    positions = sample["positions"].to(device)
    masses = sample["masses"].to(device)
    epsilon = sample["epsilon"].reshape(1).to(device)
    count = int(sample["particle_count"])
    ptr = torch.tensor([0, count], dtype=torch.long, device=device)
    synchronize(device)
    state_transfer_seconds = time.perf_counter() - transfer_started

    predictions = model(positions, masses, epsilon, ptr)
    synchronize(device)
    end_to_end_seconds = time.perf_counter() - total_started
    stats = model.last_interaction_stats
    direct_interactions = count * (count - 1)
    metadata = sample["metadata"]
    row: dict[str, float | int | str] = {
        "particle_count": count,
        "split": str(metadata["split"]),
        "evaluation_group": str(metadata["evaluation_group"]),
        "initial_condition": str(metadata["initial_condition"]),
        "trajectory_id": str(metadata["trajectory_id"]),
        "time_step": int(metadata["time_step"]),
        "state_transfer_ms": 1000.0 * state_transfer_seconds,
        "tree_input_copy_ms": 1000.0 * float(stats["host_copy_seconds"]),
        "tree_build_ms": 1000.0 * float(stats["tree_seconds"]),
        "interaction_transfer_ms": 1000.0 * float(
            stats["interaction_transfer_seconds"]
        ),
        "force_evaluation_ms": 1000.0 * float(stats["interaction_seconds"]),
        "momentum_projection_ms": 1000.0 * float(
            stats["momentum_projection_seconds"]
        ),
        "model_total_ms": 1000.0
        * (
            float(stats["build_seconds"])
            + float(stats["interaction_seconds"])
            + float(stats["momentum_projection_seconds"])
        ),
        "end_to_end_ms": 1000.0 * end_to_end_seconds,
        "body_interactions": int(stats["body_interactions"]),
        "aggregate_interactions": int(stats["aggregate_interactions"]),
        "total_interactions": int(stats["total_interactions"]),
        "direct_interactions": direct_interactions,
        "interaction_ratio": float(stats["total_interactions"]) / direct_interactions,
    }
    return row, predictions.detach().cpu()


def median_summary(rows: pd.DataFrame) -> pd.DataFrame:
    timing = [column for column in rows.columns if column.endswith("_ms")]
    fixed = [
        "body_interactions",
        "aggregate_interactions",
        "total_interactions",
        "direct_interactions",
        "interaction_ratio",
        "relative_l2",
        "rmse",
    ]
    summary = rows.groupby("particle_count", as_index=False)[timing + fixed].median()
    runs = rows.groupby("particle_count", as_index=False).size().rename(columns={"size": "runs"})
    return summary.merge(runs, on="particle_count")


def combine_cuda(
    summary: pd.DataFrame, cuda_path: Path, interaction_backend: str
) -> pd.DataFrame:
    ml_rows = summary[["particle_count", "end_to_end_ms"]].rename(
        columns={"end_to_end_ms": "median_ms"}
    )
    ml_rows["method"] = (
        f"Sparse learned force ({interaction_backend} builder)"
    )
    ml_rows["measurement_scope"] = "current run; one force evaluation"
    if not cuda_path.exists():
        return ml_rows
    cuda = pd.read_csv(cuda_path)
    required = {"implementation", "bodies", "steps", "time_ms"}
    if not required.issubset(cuda.columns):
        raise ValueError(
            f"CUDA results must contain columns: {sorted(required)}"
        )
    cuda = cuda[cuda["implementation"] == "cuda_tiled"].copy()
    cuda["amortized_step_ms"] = cuda["time_ms"] / cuda["steps"]
    cuda = (
        cuda.groupby("bodies", as_index=False)["amortized_step_ms"]
        .median()
        .rename(
            columns={
                "bodies": "particle_count",
                "amortized_step_ms": "median_ms",
            }
        )
    )
    cuda["method"] = "Tiled CUDA (amortized per step)"
    cuda["measurement_scope"] = "existing independent 10-step benchmark"
    return pd.concat([ml_rows, cuda], ignore_index=True)


def plot_results(summary: pd.DataFrame, combined: pd.DataFrame, destination: Path) -> None:
    sns.set_theme(style="whitegrid")
    figure, axes = plt.subplots(1, 3, figsize=(17, 5))

    sns.lineplot(
        data=combined,
        x="particle_count",
        y="median_ms",
        hue="method",
        marker="o",
        ax=axes[0],
    )
    axes[0].set(xscale="log", yscale="log", xlabel="Bodies", ylabel="Median time (ms)")
    axes[0].set_title("Runtime scaling")

    stages = summary.melt(
        id_vars="particle_count",
        value_vars=[
            "state_transfer_ms",
            "tree_input_copy_ms",
            "tree_build_ms",
            "interaction_transfer_ms",
            "force_evaluation_ms",
            "momentum_projection_ms",
        ],
        var_name="stage",
        value_name="median_ms",
    )
    stages["stage"] = stages["stage"].str.replace("_ms", "", regex=False).str.replace("_", " ")
    sns.lineplot(
        data=stages,
        x="particle_count",
        y="median_ms",
        hue="stage",
        marker="o",
        ax=axes[1],
    )
    axes[1].set(xscale="log", yscale="log", xlabel="Bodies", ylabel="Median time (ms)")
    axes[1].set_title("Sparse pipeline stages")

    axes[2].plot(summary["particle_count"], summary["interaction_ratio"], "o-", color="tab:blue")
    axes[2].set_xscale("log")
    axes[2].set_yscale("log")
    axes[2].set_xlabel("Bodies")
    axes[2].set_ylabel("Retained / all-pairs interactions", color="tab:blue")
    axes[2].tick_params(axis="y", labelcolor="tab:blue")
    error_axis = axes[2].twinx()
    error_axis.plot(summary["particle_count"], summary["relative_l2"], "s--", color="tab:red")
    error_axis.set_yscale("log")
    error_axis.set_ylabel("Relative L2 force error", color="tab:red")
    error_axis.tick_params(axis="y", labelcolor="tab:red")
    axes[2].set_title("Work–accuracy scaling")

    figure.tight_layout()
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=180, bbox_inches="tight")
    plt.close(figure)


def main() -> int:
    arguments = parse_arguments()
    try:
        if arguments.theta <= 0.0 or arguments.leaf_capacity <= 0:
            raise ValueError("theta and leaf-capacity must be positive")
        if arguments.warmups < 0 or arguments.repetitions <= 0:
            raise ValueError("warmups must be nonnegative and repetitions must be positive")
        if arguments.builder_threads < 0:
            raise ValueError("builder-threads cannot be negative")
        builder_threads = None
        if arguments.backend in {"cpp", "auto"}:
            try:
                from ml import _hierarchical_cpp
            except ImportError:
                if arguments.backend == "cpp":
                    raise RuntimeError(
                        "compiled interaction backend is unavailable; build it first"
                    ) from None
            else:
                if arguments.builder_threads > 0:
                    _hierarchical_cpp.set_num_threads(arguments.builder_threads)
                builder_threads = int(_hierarchical_cpp.max_threads())
        device = choose_device(arguments.device)
        model, checkpoint = load_model(
            arguments.checkpoint,
            device,
            arguments.theta,
            arguments.leaf_capacity,
            arguments.backend,
        )
        dataset = NBodyTorchDataset(
            root=arguments.dataset, particle_counts=arguments.body_counts
        )
        samples = select_samples(dataset, arguments.body_counts)
        records: list[dict[str, float | int | str]] = []
        print(
            f"Benchmarking on {device} with theta={arguments.theta}, "
            f"backend={arguments.backend}, builder_threads={builder_threads}"
        )
        with torch.inference_mode():
            for sample in samples:
                for _ in range(arguments.warmups):
                    run_once(model, sample, device)
                target = sample["accelerations"]
                for repetition in range(1, arguments.repetitions + 1):
                    row, predictions = run_once(model, sample, device)
                    difference = predictions - target
                    squared_error = float(difference.square().sum().item())
                    target_squared = float(target.square().sum().item())
                    count = int(sample["particle_count"])
                    row.update(
                        {
                            "repetition": repetition,
                            "rmse": (squared_error / (3 * count)) ** 0.5,
                            "relative_l2": (
                                squared_error / max(target_squared, 1.0e-30)
                            )
                            ** 0.5,
                        }
                    )
                    records.append(row)
                count_rows = [r for r in records if r["particle_count"] == count]
                print(
                    f"N={count}: median {statistics.median(float(r['end_to_end_ms']) for r in count_rows):.3f} ms, "
                    f"interactions={int(count_rows[0]['total_interactions']):,}"
                )

        rows = pd.DataFrame(records)
        summary = median_summary(rows)
        resolved_backend = str(model.last_interaction_stats["interaction_backend"])
        combined = combine_cuda(summary, arguments.cuda_results, resolved_backend)
        arguments.output_dir.mkdir(parents=True, exist_ok=True)
        rows.to_csv(arguments.output_dir / "per_run.csv", index=False)
        summary.to_csv(arguments.output_dir / "summary.csv", index=False)
        combined.to_csv(arguments.output_dir / "combined_methods.csv", index=False)
        configuration = {
            "dataset": str(arguments.dataset),
            "checkpoint": str(arguments.checkpoint),
            "checkpoint_epoch": checkpoint.get("epoch"),
            "cuda_results": str(arguments.cuda_results),
            "body_counts": arguments.body_counts,
            "theta": arguments.theta,
            "leaf_capacity": arguments.leaf_capacity,
            "interaction_backend_requested": arguments.backend,
            "interaction_backend_used": resolved_backend,
            "interaction_builder_threads": builder_threads,
            "warmups": arguments.warmups,
            "repetitions": arguments.repetitions,
            "device_requested": arguments.device,
            "device_used": str(device),
            "torch_version": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "platform": platform.platform(),
            "comparison_note": (
                "CUDA rows are existing 10-step runtimes divided by step count and "
                "are not paired to the selected ML snapshots."
            ),
        }
        (arguments.output_dir / "config.json").write_text(
            json.dumps(configuration, indent=2), encoding="utf-8"
        )
        plot_results(summary, combined, arguments.figure)
        print(f"Wrote benchmark results to {arguments.output_dir}")
        print(f"Wrote figure to {arguments.figure}")
        return 0
    except (FileNotFoundError, KeyError, RuntimeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
