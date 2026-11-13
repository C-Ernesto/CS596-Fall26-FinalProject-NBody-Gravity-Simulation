"""Plot momentum drift before and after the zero-net-force correction."""

from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".matplotlib"))

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


CASES = {
    "Validation N=128": {
        "Uncorrected": PROJECT_ROOT
        / "results/rollout_hierarchical_geometry/trajectory_metrics.csv",
        "Corrected": PROJECT_ROOT
        / "results/rollout_hierarchical_conservative/trajectory_metrics.csv",
    },
    "Held-out disk N=256": {
        "Uncorrected": PROJECT_ROOT
        / "results/rollout_hierarchical_geometry_disk/trajectory_metrics.csv",
        "Corrected": PROJECT_ROOT
        / "results/rollout_hierarchical_conservative_disk/trajectory_metrics.csv",
    },
}
OUTPUT_CSV = PROJECT_ROOT / "results/conservation_comparison.csv"
OUTPUT_FIGURE = PROJECT_ROOT / "visualization/output/conservation_comparison.png"


def main() -> int:
    frames: list[pd.DataFrame] = []
    for evaluation, versions in CASES.items():
        for version, path in versions.items():
            frame = pd.read_csv(path)
            frame["evaluation"] = evaluation
            frame["version"] = version
            frames.append(frame)
    data = pd.concat(frames, ignore_index=True)
    summary = (
        data.groupby(["evaluation", "version", "time_step"], as_index=False)
        .agg(
            trajectories=("trajectory_uid", "nunique"),
            momentum_drift_mean=("momentum_drift", "mean"),
            position_rmse_mean=("position_rmse", "mean"),
        )
    )
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(OUTPUT_CSV, index=False)

    sns.set_theme(style="whitegrid")
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
    for axis, evaluation in zip(axes, CASES):
        plot_data = summary[
            (summary["evaluation"] == evaluation) & (summary["time_step"] > 0)
        ]
        sns.lineplot(
            data=plot_data,
            x="time_step",
            y="momentum_drift_mean",
            hue="version",
            marker="o",
            ax=axis,
        )
        axis.set_yscale("log")
        axis.set_title(evaluation)
        axis.set_xlabel("Integration step")
        axis.set_ylabel("Mean momentum drift")
    figure.suptitle("Zero-net-force correction restores momentum conservation")
    figure.tight_layout()
    OUTPUT_FIGURE.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT_FIGURE, dpi=200, bbox_inches="tight")
    plt.close(figure)
    print(f"Summary: {OUTPUT_CSV}")
    print(f"Figure: {OUTPUT_FIGURE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
