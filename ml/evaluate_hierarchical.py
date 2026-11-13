"""Sweep the hierarchical opening angle and measure accuracy versus work."""

from __future__ import annotations

import argparse
import json
import os
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

from ml.baseline_model import PairwiseForceBaseline
from ml.hierarchical_model import HierarchicalForceModel
from ml.torch_dataset import NBodyTorchDataset
from ml.train_baseline import choose_device, parse_body_counts


DEFAULT_DATASET = PROJECT_ROOT / "data" / "processed" / "candidate_v1"
DEFAULT_CHECKPOINT = PROJECT_ROOT / "results" / "ml_baseline" / "best_model.pt"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "hierarchical_geometry_validation"
DEFAULT_FIGURE = PROJECT_ROOT / "visualization" / "output" / "hierarchical_geometry_validation.png"


def parse_float_list(value: str) -> list[float]:
    values = [float(item.strip()) for item in value.split(",") if item.strip()]
    if not values or any(item < 0.0 for item in values):
        raise argparse.ArgumentTypeError("provide comma-separated nonnegative values")
    return values


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate exact-near/learned-far hierarchical force prediction."
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument(
        "--hierarchical-checkpoint",
        type=Path,
        help="Optional fine-tuned hierarchical checkpoint; otherwise use pairwise weights.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--figure", type=Path, default=DEFAULT_FIGURE)
    parser.add_argument("--split", choices=["train", "validation", "test"], default="validation")
    parser.add_argument("--evaluation-group")
    parser.add_argument("--body-counts", type=parse_body_counts, default=[128])
    parser.add_argument("--theta", type=parse_float_list, default=[0.0, 0.3, 0.5, 0.7, 1.0])
    parser.add_argument("--snapshots-per-case", type=int, default=2)
    parser.add_argument("--leaf-capacity", type=int, default=16)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    return parser.parse_args()


def load_radial_model(
    checkpoint_path: Path, device: torch.device
) -> tuple[PairwiseForceBaseline, dict]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    configuration = checkpoint["configuration"]
    if configuration.get("model", "pairwise") != "pairwise":
        raise ValueError("the hierarchical model requires a pairwise baseline checkpoint")
    model = PairwiseForceBaseline(
        hidden_dim=int(configuration["hidden_dim"]),
        hidden_layers=int(configuration["hidden_layers"]),
        target_chunk_size=int(configuration["target_chunk_size"]),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint


def selected_indices(dataset: NBodyTorchDataset, snapshots_per_case: int) -> list[int]:
    if snapshots_per_case <= 0:
        raise ValueError("snapshots-per-case must be positive")
    frame = dataset.index.reset_index(drop=True)
    columns = ["initial_condition", "particle_count"]
    selected: list[int] = []
    for _, group in frame.groupby(columns, sort=True):
        selected.extend(group.index[:snapshots_per_case].tolist())
    return selected


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def sample_row(
    model: HierarchicalForceModel,
    sample: dict,
    theta: float,
    device: torch.device,
) -> dict[str, float | int | str]:
    positions = sample["positions"].to(device)
    masses = sample["masses"].to(device)
    epsilon = sample["epsilon"].reshape(1).to(device)
    targets = sample["accelerations"].to(device)
    count = int(sample["particle_count"])
    ptr = torch.tensor([0, count], dtype=torch.long, device=device)

    synchronize(device)
    started = time.perf_counter()
    predictions = model(positions, masses, epsilon, ptr)
    synchronize(device)
    elapsed = time.perf_counter() - started

    difference = predictions - targets
    squared_error = float(difference.square().sum().item())
    target_squared = float(targets.square().sum().item())
    relative_per_body = difference.norm(dim=1) / targets.norm(dim=1).clamp_min(1.0e-12)
    cosine = torch.nn.functional.cosine_similarity(predictions, targets, dim=1)
    stats = model.last_interaction_stats
    direct_interactions = count * (count - 1)
    metadata = sample["metadata"]
    return {
        "theta": theta,
        "split": str(metadata["split"]),
        "evaluation_group": str(metadata["evaluation_group"]),
        "initial_condition": str(metadata["initial_condition"]),
        "particle_count": count,
        "trajectory_id": str(metadata["trajectory_id"]),
        "time_step": int(metadata["time_step"]),
        "mse": squared_error / (3 * count),
        "rmse": (squared_error / (3 * count)) ** 0.5,
        "relative_l2": (squared_error / max(target_squared, 1.0e-30)) ** 0.5,
        "mean_relative_vector_error": float(relative_per_body.mean().item()),
        "mean_cosine_similarity": float(cosine.mean().item()),
        "body_interactions": int(stats["body_interactions"]),
        "aggregate_interactions": int(stats["aggregate_interactions"]),
        "total_interactions": int(stats["total_interactions"]),
        "direct_interactions": direct_interactions,
        "interaction_ratio": float(stats["total_interactions"]) / direct_interactions,
        "tree_build_ms": 1000.0 * float(stats["build_seconds"]),
        "force_evaluation_ms": 1000.0 * float(stats["interaction_seconds"]),
        "end_to_end_ms": 1000.0 * elapsed,
    }


def summarize(rows: pd.DataFrame) -> pd.DataFrame:
    keys = ["theta", "initial_condition", "particle_count"]
    metrics = [
        "mse", "rmse", "relative_l2", "mean_relative_vector_error",
        "mean_cosine_similarity", "body_interactions", "aggregate_interactions",
        "total_interactions", "direct_interactions", "interaction_ratio",
        "tree_build_ms", "force_evaluation_ms", "end_to_end_ms",
    ]
    result = rows.groupby(keys, as_index=False)[metrics].mean()
    counts = rows.groupby(keys, as_index=False).size().rename(columns={"size": "snapshots"})
    return result.merge(counts, on=keys)


def plot_tradeoff(summary: pd.DataFrame, destination: Path) -> None:
    sns.set_theme(style="whitegrid")
    figure, axes = plt.subplots(1, 2, figsize=(12, 5))
    plot_data = summary.copy()
    plot_data["case"] = (
        plot_data["initial_condition"].astype(str)
        + " N=" + plot_data["particle_count"].astype(str)
    )
    sns.lineplot(
        data=plot_data,
        x="interaction_ratio",
        y="relative_l2",
        hue="case",
        marker="o",
        ax=axes[0],
    )
    axes[0].set_yscale("log")
    axes[0].set_xlabel("Evaluated interactions / direct all-pairs interactions")
    axes[0].set_ylabel("Relative L2 acceleration error")
    axes[0].set_title("Accuracy–work tradeoff")
    sns.lineplot(
        data=plot_data,
        x="theta",
        y="interaction_ratio",
        hue="case",
        marker="o",
        legend=False,
        ax=axes[1],
    )
    axes[1].set_xlabel("Hierarchical opening angle θ")
    axes[1].set_ylabel("Interaction ratio")
    axes[1].set_title("Work removed by hierarchy")
    figure.tight_layout()
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=180, bbox_inches="tight")
    plt.close(figure)


def main() -> int:
    arguments = parse_arguments()
    try:
        device = choose_device(arguments.device)
        pairwise, checkpoint = load_radial_model(arguments.checkpoint, device)
        hierarchical_checkpoint = None
        if arguments.hierarchical_checkpoint is not None:
            hierarchical_checkpoint = torch.load(
                arguments.hierarchical_checkpoint,
                map_location=device,
                weights_only=False,
            )
        dataset = NBodyTorchDataset(
            root=arguments.dataset,
            split=arguments.split,
            evaluation_group=arguments.evaluation_group,
            particle_counts=arguments.body_counts,
        )
        indices = selected_indices(dataset, arguments.snapshots_per_case)
        configuration = checkpoint["configuration"]
        records: list[dict[str, float | int | str]] = []
        with torch.inference_mode():
            for theta in arguments.theta:
                model = HierarchicalForceModel(
                    hidden_dim=int(configuration["hidden_dim"]),
                    hidden_layers=int(configuration["hidden_layers"]),
                    theta=theta,
                    leaf_capacity=arguments.leaf_capacity,
                ).to(device)
                model.load_pairwise_radial_network(pairwise)
                if hierarchical_checkpoint is not None:
                    model.load_state_dict(hierarchical_checkpoint["model_state_dict"])
                model.eval()
                # Exclude one-time CUDA/library initialization from reported timing.
                sample_row(model, dataset[indices[0]], theta, device)
                for index in indices:
                    records.append(sample_row(model, dataset[index], theta, device))

        raw = pd.DataFrame(records)
        summary = summarize(raw)
        arguments.output_dir.mkdir(parents=True, exist_ok=True)
        raw_path = arguments.output_dir / "per_snapshot.csv"
        summary_path = arguments.output_dir / "summary.csv"
        raw.to_csv(raw_path, index=False)
        summary.to_csv(summary_path, index=False)
        plot_tradeoff(summary, arguments.figure)
        metadata = {
            "checkpoint": str(arguments.checkpoint),
            "checkpoint_epoch": int(checkpoint["epoch"]),
            "hierarchical_checkpoint": (
                str(arguments.hierarchical_checkpoint)
                if arguments.hierarchical_checkpoint is not None
                else None
            ),
            "device": str(device),
            "split": arguments.split,
            "evaluation_group": arguments.evaluation_group,
            "body_counts": arguments.body_counts,
            "theta": arguments.theta,
            "snapshots": len(indices),
            "leaf_capacity": arguments.leaf_capacity,
        }
        (arguments.output_dir / "run.json").write_text(
            json.dumps(metadata, indent=2), encoding="utf-8"
        )
        print(summary.to_string(index=False, float_format=lambda value: f"{value:.6g}"))
        print(f"\nDetailed results: {raw_path}")
        print(f"Tradeoff figure: {arguments.figure}")
        return 0
    except (OSError, ValueError, RuntimeError, KeyError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
