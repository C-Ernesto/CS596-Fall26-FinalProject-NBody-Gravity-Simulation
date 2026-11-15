"""Run and summarize hierarchical rollout configurations on trained and held-out families."""

from __future__ import annotations

import argparse
import json
import os
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

from ml.train_baseline import parse_body_counts


ROLLOUT_SCRIPT = PROJECT_ROOT / "ml" / "rollout_baseline.py"
DEFAULT_OUTPUT = PROJECT_ROOT / "results" / "rollout_matrix"
DEFAULT_FIGURE = PROJECT_ROOT / "visualization" / "output" / "rollout_matrix.png"
CONFIGURATIONS = {
    "original_theta_0.7": (
        PROJECT_ROOT / "results" / "ml_hierarchical_geometry" / "best_model.pt",
        0.7,
    ),
    "original_theta_1.0": (
        PROJECT_ROOT / "results" / "ml_hierarchical_geometry" / "best_model.pt",
        1.0,
    ),
    "guarded_theta_0.7": (
        PROJECT_ROOT
        / "results"
        / "ml_hierarchical_balanced_guarded"
        / "best_model.pt",
        0.7,
    ),
    "guarded_theta_1.0": (
        PROJECT_ROOT
        / "results"
        / "ml_hierarchical_balanced_guarded"
        / "best_model.pt",
        1.0,
    ),
}
EVALUATIONS = {
    "validation": ("validation", None),
    "held_out_disk": ("test", "held_out_family"),
}


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--body-counts", type=parse_body_counts, default=[1024, 2048])
    parser.add_argument("--max-trajectories-per-family", type=int, default=1)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--figure", type=Path, default=DEFAULT_FIGURE)
    parser.add_argument("--device", choices=["auto", "cuda"], default="auto")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def run_configuration(
    name: str,
    checkpoint: Path,
    theta: float,
    evaluation_name: str,
    split: str,
    evaluation_group: str | None,
    arguments: argparse.Namespace,
) -> Path:
    destination = arguments.output_dir / name / evaluation_name
    metrics_path = destination / "trajectory_metrics.csv"
    runtime_path = destination / "trajectory_runtime.csv"
    if metrics_path.is_file() and runtime_path.is_file() and not arguments.overwrite:
        print(f"Reusing {name} / {evaluation_name}", flush=True)
        return destination
    command = [
        sys.executable,
        str(ROLLOUT_SCRIPT),
        "--predictor",
        "hierarchical",
        "--checkpoint",
        str(checkpoint),
        "--theta",
        str(theta),
        "--interaction-backend",
        "cuda",
        "--split",
        split,
        "--body-counts",
        ",".join(map(str, arguments.body_counts)),
        "--max-trajectories-per-family",
        str(arguments.max_trajectories_per_family),
        "--device",
        arguments.device,
        "--output-dir",
        str(destination),
        "--figure-dir",
        str(destination / "figures"),
    ]
    if evaluation_group is not None:
        command.extend(["--evaluation-group", evaluation_group])
    print(f"Running {name} / {evaluation_name}", flush=True)
    subprocess.run(command, check=True)
    return destination


def plot_final(summary: pd.DataFrame, destination: Path) -> None:
    sns.set_theme(style="whitegrid")
    metrics = [
        ("position_rmse", "Final position RMSE"),
        ("velocity_rmse", "Final velocity RMSE"),
        ("relative_energy_drift", "Final relative energy drift"),
        ("simulation_ms", "175-step simulation time (ms)"),
    ]
    counts = sorted(summary["particle_count"].unique())
    figure, axes = plt.subplots(
        len(counts), len(metrics), figsize=(5 * len(metrics), 4.6 * len(counts))
    )
    if len(counts) == 1:
        axes = [axes]
    for row, count in enumerate(counts):
        data = summary[summary["particle_count"] == count]
        for column, (metric, title) in enumerate(metrics):
            axis = axes[row][column]
            sns.barplot(data=data, x="initial_condition", y=metric, hue="configuration", ax=axis)
            axis.set_yscale("log")
            axis.set_title(f"{title}, N={count}")
            axis.set_xlabel("")
            axis.tick_params(axis="x", rotation=25)
            if not (row == 0 and column == 0) and axis.get_legend() is not None:
                axis.get_legend().remove()
    figure.tight_layout()
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=180, bbox_inches="tight")
    plt.close(figure)


def main() -> int:
    arguments = parse_arguments()
    try:
        if arguments.max_trajectories_per_family <= 0:
            raise ValueError("max trajectories per family must be positive")
        for name, (checkpoint, _) in CONFIGURATIONS.items():
            if not checkpoint.is_file():
                raise FileNotFoundError(f"{name}: {checkpoint}")
        arguments.output_dir.mkdir(parents=True, exist_ok=True)

        metric_frames: list[pd.DataFrame] = []
        runtime_frames: list[pd.DataFrame] = []
        for name, (checkpoint, theta) in CONFIGURATIONS.items():
            for evaluation_name, (split, evaluation_group) in EVALUATIONS.items():
                destination = run_configuration(
                    name,
                    checkpoint,
                    theta,
                    evaluation_name,
                    split,
                    evaluation_group,
                    arguments,
                )
                metrics = pd.read_csv(destination / "trajectory_metrics.csv")
                runtimes = pd.read_csv(destination / "trajectory_runtime.csv")
                for frame in (metrics, runtimes):
                    frame["configuration"] = name
                    frame["theta"] = theta
                    frame["evaluation"] = evaluation_name
                metric_frames.append(metrics)
                runtime_frames.append(runtimes)

        all_metrics = pd.concat(metric_frames, ignore_index=True)
        all_runtimes = pd.concat(runtime_frames, ignore_index=True)
        final_metrics = (
            all_metrics.sort_values("time_step")
            .groupby(["configuration", "trajectory_uid"], as_index=False)
            .tail(1)
        )
        final_metrics = final_metrics.merge(
            all_runtimes,
            on=[
                "configuration",
                "theta",
                "evaluation",
                "predictor",
                "trajectory_uid",
                "initial_condition",
                "particle_count",
                "seed",
            ],
            validate="one_to_one",
        )
        keys = ["configuration", "theta", "evaluation", "initial_condition", "particle_count"]
        summary = final_metrics.groupby(keys, as_index=False).agg(
            trajectories=("trajectory_uid", "nunique"),
            position_rmse=("position_rmse", "mean"),
            velocity_rmse=("velocity_rmse", "mean"),
            force_rmse=("force_rmse_on_exact_state", "mean"),
            relative_energy_drift=("relative_energy_drift", "mean"),
            momentum_drift=("momentum_drift", "mean"),
            simulation_ms=("simulation_ms", "mean"),
            rollout_force_ms=("rollout_force_ms", "mean"),
            mean_rollout_force_ms=("mean_rollout_force_ms", "mean"),
        )
        all_metrics.to_csv(arguments.output_dir / "trajectory_metrics.csv", index=False)
        all_runtimes.to_csv(arguments.output_dir / "trajectory_runtime.csv", index=False)
        final_metrics.to_csv(arguments.output_dir / "final_per_trajectory.csv", index=False)
        summary.to_csv(arguments.output_dir / "final_summary.csv", index=False)
        plot_final(summary, arguments.figure)
        configuration = {
            "body_counts": arguments.body_counts,
            "max_trajectories_per_family": arguments.max_trajectories_per_family,
            "configurations": {
                name: {"checkpoint": str(checkpoint), "theta": theta}
                for name, (checkpoint, theta) in CONFIGURATIONS.items()
            },
            "reference": "saved exact tiled-CUDA trajectory checkpoints",
        }
        (arguments.output_dir / "config.json").write_text(
            json.dumps(configuration, indent=2), encoding="utf-8"
        )
        print("\nFinal rollout summary:")
        print(summary.to_string(index=False, float_format=lambda value: f"{value:.6g}"))
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
