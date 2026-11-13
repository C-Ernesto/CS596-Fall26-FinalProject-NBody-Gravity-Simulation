"""Run learned or exact velocity-Verlet rollouts against CUDA checkpoints."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
MATPLOTLIB_CONFIG = PROJECT_ROOT / ".matplotlib"
MATPLOTLIB_CONFIG.mkdir(exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MATPLOTLIB_CONFIG))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch

from ml.baseline_model import PairwiseForceBaseline
from ml.hierarchical_model import HierarchicalForceModel
from ml.rollout import (
    dimensionless_time_step,
    exact_acceleration,
    softened_total_energy,
    total_momentum,
    velocity_verlet_step,
)
from ml.torch_dataset import NBodyTorchDataset
from ml.train_baseline import choose_device, parse_body_counts


DEFAULT_DATASET = PROJECT_ROOT / "data" / "processed" / "candidate_v1"
DEFAULT_CHECKPOINT = PROJECT_ROOT / "results" / "ml_baseline" / "best_model.pt"


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate autoregressive N-body rollouts.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument(
        "--predictor",
        choices=["learned", "hierarchical", "exact"],
        default="learned",
    )
    parser.add_argument("--theta", type=float)
    parser.add_argument("--leaf-capacity", type=int)
    parser.add_argument("--split", choices=["train", "validation", "test"], default="validation")
    parser.add_argument("--evaluation-group")
    parser.add_argument("--body-counts", type=parse_body_counts, default=[128])
    parser.add_argument("--max-trajectories-per-family", type=int, default=0)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--figure-dir", type=Path)
    return parser.parse_args()


def load_model(
    checkpoint_path: Path,
    device: torch.device,
    theta_override: float | None = None,
    leaf_capacity_override: int | None = None,
):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    configuration = checkpoint["configuration"]
    model_name = configuration.get("model", "pairwise")
    if model_name == "pairwise":
        model = PairwiseForceBaseline(
            hidden_dim=int(configuration["hidden_dim"]),
            hidden_layers=int(configuration["hidden_layers"]),
            target_chunk_size=int(configuration["target_chunk_size"]),
        )
    elif model_name == "hierarchical":
        model = HierarchicalForceModel(
            hidden_dim=int(configuration["hidden_dim"]),
            hidden_layers=int(configuration["hidden_layers"]),
            theta=(
                theta_override
                if theta_override is not None
                else float(configuration["theta"])
            ),
            leaf_capacity=(
                leaf_capacity_override
                if leaf_capacity_override is not None
                else int(configuration["leaf_capacity"])
            ),
        )
    else:
        raise ValueError(f"unsupported checkpoint model: {model_name}")
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()
    return model, model_name, checkpoint


def model_acceleration(
    model,
    model_name: str,
    positions: torch.Tensor,
    masses: torch.Tensor,
    epsilon: torch.Tensor,
) -> torch.Tensor:
    if model_name in {"pairwise", "hierarchical"}:
        return model(
            positions,
            masses,
            epsilon.reshape(1),
            torch.tensor([0, len(positions)], device=positions.device),
        )
    node_features = torch.cat([positions, masses], dim=1)
    return model(node_features)


def select_trajectories(index: pd.DataFrame, maximum_per_family: int) -> list[str]:
    trajectories = index[
        ["trajectory_uid", "initial_condition", "seed"]
    ].drop_duplicates("trajectory_uid")
    trajectories = trajectories.sort_values(["initial_condition", "seed"])
    if maximum_per_family > 0:
        trajectories = trajectories.groupby("initial_condition", as_index=False).head(
            maximum_per_family
        )
    return trajectories["trajectory_uid"].tolist()


def create_plots(metrics: pd.DataFrame, final_states: dict, figure_dir: Path) -> None:
    figure_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="notebook")
    summary = (
        metrics.groupby(["initial_condition", "time_step"], as_index=False)
        .agg(
            position_rmse=("position_rmse", "mean"),
            velocity_rmse=("velocity_rmse", "mean"),
            relative_energy_drift=("relative_energy_drift", "mean"),
        )
    )
    figure, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    for axis, metric, title in [
        (axes[0], "position_rmse", "Position RMSE"),
        (axes[1], "velocity_rmse", "Velocity RMSE"),
        (axes[2], "relative_energy_drift", "Relative energy drift"),
    ]:
        # Step zero is exactly identical by construction and cannot be shown on
        # a logarithmic axis, so error curves begin at the first saved checkpoint.
        plot_data = summary[summary["time_step"] > 0].copy()
        sns.lineplot(
            data=plot_data,
            x="time_step",
            y=metric,
            hue="initial_condition",
            marker="o",
            ax=axis,
            legend=axis is axes[0],
        )
        axis.set_yscale("log")
        axis.set_xlabel("Integration step")
        axis.set_ylabel(metric.replace("_", " "))
        axis.set_title(title)
        if axis is not axes[0] and axis.get_legend() is not None:
            axis.get_legend().remove()
    figure.suptitle("Autoregressive rollout error")
    figure.tight_layout()
    figure.savefig(figure_dir / "rollout_errors.png", dpi=200)
    plt.close(figure)

    families = sorted(final_states)
    figure, axes = plt.subplots(1, len(families), figsize=(5 * len(families), 4.8))
    if len(families) == 1:
        axes = [axes]
    for axis, family in zip(axes, families):
        predicted, exact = final_states[family]
        axis.scatter(exact[:, 0], exact[:, 1], s=13, alpha=0.65, label="Exact CUDA")
        axis.scatter(
            predicted[:, 0], predicted[:, 1], s=13, alpha=0.65, marker="x", label="Rollout"
        )
        axis.set_title(family)
        axis.set_xlabel("Normalized x")
        axis.set_ylabel("Normalized y")
        axis.set_aspect("equal", adjustable="box")
    axes[0].legend()
    figure.suptitle("Final saved checkpoint: exact versus rollout")
    figure.tight_layout()
    figure.savefig(figure_dir / "rollout_final_positions.png", dpi=200)
    plt.close(figure)


def main() -> int:
    arguments = parse_arguments()
    try:
        if arguments.max_trajectories_per_family < 0:
            raise ValueError("max trajectories per family cannot be negative")
        device = choose_device(arguments.device)
        dtype = torch.float64 if arguments.predictor == "exact" else torch.float32
        dataset = NBodyTorchDataset(
            arguments.dataset,
            split=arguments.split,
            evaluation_group=arguments.evaluation_group,
            particle_counts=arguments.body_counts,
            normalize=True,
            dtype=dtype,
        )
        model = None
        model_name = "exact"
        checkpoint = None
        if arguments.predictor in {"learned", "hierarchical"}:
            model, model_name, checkpoint = load_model(
                arguments.checkpoint,
                device,
                arguments.theta,
                arguments.leaf_capacity,
            )
            if arguments.predictor == "hierarchical" and model_name != "hierarchical":
                raise ValueError("hierarchical predictor requires a hierarchical checkpoint")

        output_dir = arguments.output_dir or (
            PROJECT_ROOT / "results" / f"rollout_{arguments.predictor}"
        )
        figure_dir = arguments.figure_dir or (
            PROJECT_ROOT / "visualization" / "output" / f"rollout_{arguments.predictor}"
        )
        output_dir.mkdir(parents=True, exist_ok=True)

        trajectories = select_trajectories(
            dataset.index, arguments.max_trajectories_per_family
        )
        metric_rows: list[dict] = []
        final_states: dict[str, tuple[np.ndarray, np.ndarray]] = {}

        for trajectory_number, trajectory_uid in enumerate(trajectories, start=1):
            rows = dataset.index[dataset.index["trajectory_uid"] == trajectory_uid]
            rows = rows.sort_values("time_step")
            dataset_indices = rows.index.tolist()
            snapshots = [dataset[index] for index in dataset_indices]
            initial = snapshots[0]
            metadata = initial["metadata"]
            radius = float(metadata["characteristic_radius"])
            total_mass_value = float(metadata["total_mass"])
            dt = dimensionless_time_step(float(metadata["dt"]), radius, total_mass_value)
            family = str(metadata["initial_condition"])

            positions = initial["positions"].to(device)
            velocities = initial["velocities"].to(device)
            masses = initial["masses"].to(device)
            epsilon = initial["epsilon"].to(device)

            def acceleration_function(state_positions: torch.Tensor) -> torch.Tensor:
                if arguments.predictor == "exact":
                    return exact_acceleration(state_positions, masses, epsilon)
                assert model is not None
                return model_acceleration(
                    model, model_name, state_positions, masses, epsilon
                )

            with torch.inference_mode():
                accelerations = acceleration_function(positions)
                initial_energy = softened_total_energy(
                    positions, velocities, masses, epsilon
                )
                initial_momentum = total_momentum(velocities, masses)
                saved_by_step = {
                    int(snapshot["metadata"]["time_step"]): snapshot
                    for snapshot in snapshots
                }
                maximum_step = max(saved_by_step)

                for time_step in range(maximum_step + 1):
                    if time_step in saved_by_step:
                        exact_snapshot = saved_by_step[time_step]
                        exact_positions = exact_snapshot["positions"].to(device)
                        exact_velocities = exact_snapshot["velocities"].to(device)
                        exact_accelerations = exact_snapshot["accelerations"].to(device)
                        force_on_exact = acceleration_function(exact_positions)
                        energy = softened_total_energy(
                            positions, velocities, masses, epsilon
                        )
                        momentum = total_momentum(velocities, masses)
                        metric_rows.append(
                            {
                                "predictor": arguments.predictor,
                                "trajectory_uid": trajectory_uid,
                                "initial_condition": family,
                                "particle_count": len(positions),
                                "seed": int(metadata["seed"]),
                                "time_step": time_step,
                                "normalized_time": time_step * dt,
                                "position_rmse": float(
                                    torch.sqrt(
                                        (positions - exact_positions).square().mean()
                                    ).item()
                                ),
                                "velocity_rmse": float(
                                    torch.sqrt(
                                        (velocities - exact_velocities).square().mean()
                                    ).item()
                                ),
                                "force_rmse_on_exact_state": float(
                                    torch.sqrt(
                                        (force_on_exact - exact_accelerations)
                                        .square()
                                        .mean()
                                    ).item()
                                ),
                                "relative_energy_drift": float(
                                    (energy - initial_energy).abs().item()
                                    / max(abs(float(initial_energy.item())), 1.0e-15)
                                ),
                                "momentum_drift": float(
                                    torch.linalg.vector_norm(
                                        momentum - initial_momentum
                                    ).item()
                                ),
                            }
                        )
                        if time_step == maximum_step and family not in final_states:
                            final_states[family] = (
                                positions.detach().cpu().numpy(),
                                exact_positions.detach().cpu().numpy(),
                            )
                    if time_step < maximum_step:
                        positions, velocities, accelerations = velocity_verlet_step(
                            positions,
                            velocities,
                            accelerations,
                            dt,
                            acceleration_function,
                        )
                        if not (
                            torch.isfinite(positions).all()
                            and torch.isfinite(velocities).all()
                            and torch.isfinite(accelerations).all()
                        ):
                            raise ValueError(
                                f"non-finite rollout state for {trajectory_uid} at step {time_step + 1}"
                            )
            print(
                f"[{trajectory_number}/{len(trajectories)}] {trajectory_uid}: "
                f"family={family}, final position RMSE={metric_rows[-1]['position_rmse']:.6g}"
            )

        metrics = pd.DataFrame(metric_rows)
        metrics.to_csv(output_dir / "trajectory_metrics.csv", index=False)
        summary = (
            metrics.groupby(["initial_condition", "time_step"], as_index=False)
            .agg(
                trajectories=("trajectory_uid", "nunique"),
                position_rmse_mean=("position_rmse", "mean"),
                position_rmse_max=("position_rmse", "max"),
                velocity_rmse_mean=("velocity_rmse", "mean"),
                force_rmse_mean=("force_rmse_on_exact_state", "mean"),
                relative_energy_drift_mean=("relative_energy_drift", "mean"),
                relative_energy_drift_max=("relative_energy_drift", "max"),
                momentum_drift_mean=("momentum_drift", "mean"),
            )
        )
        summary.to_csv(output_dir / "summary.csv", index=False)
        configuration = {
            "predictor": arguments.predictor,
            "checkpoint": str(arguments.checkpoint) if checkpoint else None,
            "checkpoint_epoch": checkpoint["epoch"] if checkpoint else None,
            "split": arguments.split,
            "evaluation_group": arguments.evaluation_group,
            "body_counts": arguments.body_counts,
            "trajectories": len(trajectories),
            "device": str(device),
            "dtype": str(dtype),
            "theta": getattr(model, "theta", None),
            "leaf_capacity": getattr(model, "leaf_capacity", None),
        }
        (output_dir / "config.json").write_text(
            json.dumps(configuration, indent=2), encoding="utf-8"
        )
        create_plots(metrics, final_states, figure_dir)
        print(f"Metrics written to: {output_dir}")
        print(f"Figures written to: {figure_dir}")
        return 0
    except (OSError, ValueError, RuntimeError, KeyError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
