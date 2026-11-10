"""Summarize and visualize detailed CUDA N-body benchmarks."""

from __future__ import annotations

import argparse
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
import pandas as pd
import seaborn as sns


DEFAULT_INPUT = PROJECT_ROOT / "results" / "cuda_detailed_scaling.csv"
DEFAULT_SCALING_PLOT = (
    PROJECT_ROOT / "visualization" / "output" / "cuda_detailed_scaling.png"
)
DEFAULT_BREAKDOWN_PLOT = (
    PROJECT_ROOT / "visualization" / "output" / "cuda_timing_breakdown.png"
)
DEFAULT_SPEEDUP_PLOT = (
    PROJECT_ROOT / "visualization" / "output" / "cuda_tiled_speedup.png"
)
DEFAULT_SUMMARY = PROJECT_ROOT / "results" / "cuda_detailed_scaling_summary.csv"

TIMING_COLUMNS = [
    "allocation_ms",
    "h2d_ms",
    "kernel_ms",
    "d2h_ms",
    "cleanup_ms",
    "total_ms",
]
REQUIRED_COLUMNS = {
    "implementation",
    "bodies",
    "time_ms",
    *TIMING_COLUMNS,
}


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize and plot detailed N-body CUDA benchmarks."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_SCALING_PLOT,
        help="End-to-end scaling plot path.",
    )
    parser.add_argument(
        "--breakdown-output",
        type=Path,
        default=DEFAULT_BREAKDOWN_PLOT,
        help="Stacked timing-breakdown plot path.",
    )
    parser.add_argument(
        "--speedup-output",
        type=Path,
        default=DEFAULT_SPEEDUP_PLOT,
        help="Tiled-versus-naive speedup plot path.",
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=DEFAULT_SUMMARY,
        help="Per-configuration summary CSV path.",
    )
    return parser.parse_args()


def load_results(path: Path) -> pd.DataFrame:
    """Load the raw repetitions and validate the detailed timing schema."""
    if not path.is_file():
        raise ValueError(f"benchmark CSV does not exist: {path}")

    results = pd.read_csv(path)
    missing_columns = REQUIRED_COLUMNS - set(results.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"benchmark CSV is missing columns: {missing}")

    results["implementation"] = results["implementation"].astype(str).str.strip()
    numeric_columns = ["bodies", "time_ms", *TIMING_COLUMNS]
    results[numeric_columns] = results[numeric_columns].apply(
        pd.to_numeric,
        errors="raise",
    )

    if results.empty:
        raise ValueError(f"benchmark CSV contains no data rows: {path}")
    if results["implementation"].eq("").any():
        raise ValueError("implementation cannot be empty")
    if results["bodies"].le(0).any():
        raise ValueError("bodies must be positive")
    if results[["time_ms", *TIMING_COLUMNS]].le(0).any().any():
        raise ValueError("all timing measurements must be positive")

    component_sum = results[
        ["allocation_ms", "h2d_ms", "kernel_ms", "d2h_ms", "cleanup_ms"]
    ].sum(axis=1)
    if (component_sum > results["total_ms"] + 0.1).any():
        raise ValueError("timing components exceed CUDA total time")
    if (results["total_ms"] > results["time_ms"] + 0.1).any():
        raise ValueError("CUDA total time exceeds outer end-to-end time")

    results["other_cuda_ms"] = (results["total_ms"] - component_sum).clip(
        lower=0.0
    )
    results["outer_overhead_ms"] = (
        results["time_ms"] - results["total_ms"]
    ).clip(lower=0.0)
    return results


def summarize(results: pd.DataFrame) -> pd.DataFrame:
    """Calculate per-configuration medians and end-to-end variability."""
    summary = (
        results.groupby(["implementation", "bodies"], as_index=False)
        .agg(
            runs=("time_ms", "size"),
            median_ms=("time_ms", "median"),
            min_ms=("time_ms", "min"),
            max_ms=("time_ms", "max"),
            stddev_ms=("time_ms", "std"),
            allocation_ms=("allocation_ms", "median"),
            h2d_ms=("h2d_ms", "median"),
            kernel_ms=("kernel_ms", "median"),
            d2h_ms=("d2h_ms", "median"),
            cleanup_ms=("cleanup_ms", "median"),
            other_cuda_ms=("other_cuda_ms", "median"),
            cuda_total_ms=("total_ms", "median"),
            outer_overhead_ms=("outer_overhead_ms", "median"),
        )
        .sort_values(["implementation", "bodies"])
    )
    summary["stddev_ms"] = summary["stddev_ms"].fillna(0.0)
    summary["kernel_percent"] = (
        100.0 * summary["kernel_ms"] / summary["cuda_total_ms"]
    )
    return summary


def warn_about_run_counts(summary: pd.DataFrame) -> None:
    run_counts = sorted(summary["runs"].unique())
    if len(run_counts) > 1:
        values = ", ".join(str(count) for count in run_counts)
        print(
            f"warning: configurations have different repetition counts: {values}",
            file=sys.stderr,
        )


def create_scaling_plot(summary: pd.DataFrame, output_path: Path) -> None:
    """Plot median end-to-end runtime with min/max error bars."""
    sns.set_theme(style="whitegrid", context="notebook")
    implementations = sorted(summary["implementation"].unique())
    colors = sns.color_palette("colorblind", n_colors=len(implementations))
    palette = dict(zip(implementations, colors))

    figure, axis = plt.subplots(figsize=(9, 5.5))
    sns.lineplot(
        data=summary,
        x="bodies",
        y="median_ms",
        hue="implementation",
        style="implementation",
        markers=True,
        dashes=False,
        palette=palette,
        errorbar=None,
        ax=axis,
    )

    for implementation in implementations:
        rows = summary[summary["implementation"] == implementation]
        axis.errorbar(
            rows["bodies"],
            rows["median_ms"],
            yerr=[
                rows["median_ms"] - rows["min_ms"],
                rows["max_ms"] - rows["median_ms"],
            ],
            fmt="none",
            color=palette[implementation],
            capsize=4,
        )

    body_counts = sorted(summary["bodies"].unique())
    axis.set_xscale("log", base=2)
    axis.set_yscale("log")
    axis.set_xticks(body_counts)
    axis.set_xticklabels([str(count) for count in body_counts])
    axis.set_xlabel("Number of bodies")
    axis.set_ylabel("Median end-to-end simulation time (ms)")
    axis.set_title("CUDA N-body runtime scaling")
    axis.legend(title="Implementation")

    save_figure(figure, output_path)


def create_breakdown_plot(summary: pd.DataFrame, output_path: Path) -> None:
    """Plot each CUDA wrapper component as a percentage of total time."""
    components = {
        "Allocation": "allocation_ms",
        "Host to device": "h2d_ms",
        "Kernels": "kernel_ms",
        "Device to host": "d2h_ms",
        "Cleanup": "cleanup_ms",
        "Other CUDA wrapper": "other_cuda_ms",
    }
    implementations = sorted(summary["implementation"].unique())
    colors = sns.color_palette("colorblind", n_colors=len(components))

    figure, axes = plt.subplots(
        1,
        len(implementations),
        figsize=(13, 5.5),
        sharey=True,
    )
    if len(implementations) == 1:
        axes = [axes]

    for axis, implementation in zip(axes, implementations):
        rows = summary[summary["implementation"] == implementation]
        bottom = pd.Series(0.0, index=rows.index)

        for (label, column), color in zip(components.items(), colors):
            percentage = 100.0 * rows[column] / rows["cuda_total_ms"]
            axis.bar(
                rows["bodies"].astype(str),
                percentage,
                bottom=bottom,
                label=label,
                color=color,
            )
            bottom += percentage

        axis.set_title(implementation)
        axis.set_xlabel("Number of bodies")
        axis.tick_params(axis="x", rotation=45)
        axis.set_ylim(0.0, 100.0)

    axes[0].set_ylabel("Share of CUDA wrapper time (%)")
    figure.suptitle("CUDA simulation timing breakdown")
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="lower center",
        ncol=3,
        frameon=False,
    )
    figure.tight_layout(rect=(0.0, 0.12, 1.0, 0.94))
    save_figure(figure, output_path, apply_tight_layout=False)


def create_speedup_plot(summary: pd.DataFrame, output_path: Path) -> None:
    """Plot naïve runtime divided by tiled runtime; values above one favor tiling."""
    naive = summary[summary["implementation"] == "cuda_naive"].set_index(
        "bodies"
    )
    tiled = summary[summary["implementation"] == "cuda_tiled"].set_index(
        "bodies"
    )
    common_counts = naive.index.intersection(tiled.index).sort_values()
    if common_counts.empty:
        raise ValueError("speedup plot requires cuda_naive and cuda_tiled rows")

    speedup = pd.DataFrame(
        {
            "bodies": common_counts,
            "End-to-end": (
                naive.loc[common_counts, "median_ms"].to_numpy()
                / tiled.loc[common_counts, "median_ms"].to_numpy()
            ),
            "Kernels": (
                naive.loc[common_counts, "kernel_ms"].to_numpy()
                / tiled.loc[common_counts, "kernel_ms"].to_numpy()
            ),
        }
    ).melt(
        id_vars="bodies",
        var_name="Measurement",
        value_name="Speedup",
    )

    figure, axis = plt.subplots(figsize=(9, 5.5))
    sns.lineplot(
        data=speedup,
        x="bodies",
        y="Speedup",
        hue="Measurement",
        style="Measurement",
        markers=True,
        dashes=False,
        ax=axis,
    )
    axis.axhline(1.0, color="black", linestyle="--", linewidth=1.2)
    axis.set_xscale("log", base=2)
    axis.set_xticks(common_counts)
    axis.set_xticklabels([str(count) for count in common_counts])
    axis.set_xlabel("Number of bodies")
    axis.set_ylabel("Naïve time / tiled time")
    axis.set_title("Speedup from shared-memory tiling")
    axis.legend(title="Measurement")

    save_figure(figure, output_path)


def save_figure(
    figure: plt.Figure,
    output_path: Path,
    *,
    apply_tight_layout: bool = True,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if apply_tight_layout:
        figure.tight_layout()
    figure.savefig(output_path, dpi=200)
    plt.close(figure)


def main() -> int:
    arguments = parse_arguments()

    try:
        results = load_results(arguments.input)
        summary = summarize(results)
        warn_about_run_counts(summary)

        arguments.summary_output.parent.mkdir(parents=True, exist_ok=True)
        summary.to_csv(arguments.summary_output, index=False)
        print(summary.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
        print(f"Summary written to: {arguments.summary_output}")

        create_scaling_plot(summary, arguments.output)
        create_breakdown_plot(summary, arguments.breakdown_output)
        create_speedup_plot(summary, arguments.speedup_output)
        print(f"Scaling plot written to: {arguments.output}")
        print(f"Breakdown plot written to: {arguments.breakdown_output}")
        print(f"Speedup plot written to: {arguments.speedup_output}")
    except (OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
