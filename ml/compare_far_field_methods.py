"""Compare learned aggregate forces with matched analytical Barnes-Hut cells."""

from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".matplotlib"))

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


ONE_STEP_CASES = {
    "Validation N=128": {
        "Learned": "far_field_learned_validation",
        "Classical Barnes-Hut": "far_field_analytic_validation",
    },
    "Unseen N=768": {
        "Learned": "far_field_learned_768",
        "Classical Barnes-Hut": "far_field_analytic_768",
    },
    "Held-out disk N=256": {
        "Learned": "far_field_learned_disk",
        "Classical Barnes-Hut": "far_field_analytic_disk",
    },
}
ROLLOUT_CASES = {
    "Validation N=128": {
        "Learned": "rollout_hierarchical_conservative",
        "Classical Barnes-Hut": "rollout_barnes_hut",
    },
    "Held-out disk N=256": {
        "Learned": "rollout_hierarchical_conservative_disk",
        "Classical Barnes-Hut": "rollout_barnes_hut_disk",
    },
}


def one_step_summary() -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for evaluation, methods in ONE_STEP_CASES.items():
        for method, directory in methods.items():
            frame = pd.read_csv(PROJECT_ROOT / "results" / directory / "summary.csv")
            frame["evaluation"] = evaluation
            frame["method"] = method
            frames.append(frame)
    data = pd.concat(frames, ignore_index=True)
    return (
        data.groupby(["evaluation", "method", "theta"], as_index=False)
        .agg(
            relative_l2_mean=("relative_l2", "mean"),
            interaction_ratio=("interaction_ratio", "mean"),
            force_evaluation_ms=("force_evaluation_ms", "mean"),
            end_to_end_ms=("end_to_end_ms", "mean"),
        )
    )


def rollout_summary() -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for evaluation, methods in ROLLOUT_CASES.items():
        for method, directory in methods.items():
            frame = pd.read_csv(
                PROJECT_ROOT / "results" / directory / "trajectory_metrics.csv"
            )
            frame["evaluation"] = evaluation
            frame["method"] = method
            frames.append(frame)
    data = pd.concat(frames, ignore_index=True)
    return (
        data.groupby(["evaluation", "method", "time_step"], as_index=False)
        .agg(
            trajectories=("trajectory_uid", "nunique"),
            position_rmse=("position_rmse", "mean"),
            velocity_rmse=("velocity_rmse", "mean"),
            force_rmse=("force_rmse_on_exact_state", "mean"),
            energy_drift=("relative_energy_drift", "mean"),
            momentum_drift=("momentum_drift", "mean"),
        )
    )


def main() -> int:
    sns.set_theme(style="whitegrid")
    one_step = one_step_summary()
    one_step_path = PROJECT_ROOT / "results/far_field_method_comparison.csv"
    one_step.to_csv(one_step_path, index=False)
    figure, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    for axis, evaluation in zip(axes, ONE_STEP_CASES):
        data = one_step[one_step["evaluation"] == evaluation]
        sns.lineplot(
            data=data,
            x="interaction_ratio",
            y="relative_l2_mean",
            hue="method",
            marker="o",
            ax=axis,
            legend=axis is axes[0],
        )
        axis.set_yscale("log")
        axis.set_title(evaluation)
        axis.set_xlabel("Interaction ratio")
        axis.set_ylabel("Mean relative L2 force error")
        if axis is not axes[0] and axis.get_legend() is not None:
            axis.get_legend().remove()
    figure.suptitle("Matched far-field comparison")
    figure.tight_layout()
    one_step_figure = PROJECT_ROOT / "visualization/output/far_field_method_comparison.png"
    figure.savefig(one_step_figure, dpi=200, bbox_inches="tight")
    plt.close(figure)

    rollout = rollout_summary()
    rollout_path = PROJECT_ROOT / "results/far_field_rollout_comparison.csv"
    rollout.to_csv(rollout_path, index=False)
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
    for axis, evaluation in zip(axes, ROLLOUT_CASES):
        data = rollout[
            (rollout["evaluation"] == evaluation) & (rollout["time_step"] > 0)
        ]
        sns.lineplot(
            data=data,
            x="time_step",
            y="position_rmse",
            hue="method",
            marker="o",
            ax=axis,
        )
        axis.set_yscale("log")
        axis.set_title(evaluation)
        axis.set_xlabel("Integration step")
        axis.set_ylabel("Mean position RMSE")
    figure.suptitle("Learned versus classical Barnes-Hut rollout")
    figure.tight_layout()
    rollout_figure = PROJECT_ROOT / "visualization/output/far_field_rollout_comparison.png"
    figure.savefig(rollout_figure, dpi=200, bbox_inches="tight")
    plt.close(figure)

    print(f"One-step summary: {one_step_path}")
    print(f"Rollout summary: {rollout_path}")
    print(f"Figures: {one_step_figure}, {rollout_figure}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
