"""Summarize and plot the final paired CPU-CUDA force benchmark."""

from __future__ import annotations

import argparse
import os
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


DEFAULT_INPUT = PROJECT_ROOT / "results" / "cpu_cuda_crossover.csv"
DEFAULT_SUMMARY = PROJECT_ROOT / "results" / "cpu_cuda_crossover_summary.csv"
DEFAULT_FIGURE = (
    PROJECT_ROOT / "visualization" / "output" / "cpu_cuda_crossover.png"
)

IMPLEMENTATION_LABELS = {
    "serial_force": "Serial CPU",
    "openmp_force": "OpenMP CPU (16 threads)",
    "cuda_tiled_force": "Tiled CUDA",
}
IMPLEMENTATION_ORDER = list(IMPLEMENTATION_LABELS)
REQUIRED_COLUMNS = {
    "implementation",
    "initial_condition",
    "bodies",
    "seed",
    "repetition",
    "threads",
    "total_ms",
}


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize and plot the final paired CPU-CUDA benchmark."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--summary-output", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--figure-output", type=Path, default=DEFAULT_FIGURE)
    return parser.parse_args()


def load_results(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise ValueError(f"benchmark CSV does not exist: {path}")

    results = pd.read_csv(path)
    missing = REQUIRED_COLUMNS - set(results.columns)
    if missing:
        raise ValueError(
            "benchmark CSV is missing columns: " + ", ".join(sorted(missing))
        )
    if results.empty:
        raise ValueError("benchmark CSV contains no measurements")

    results = results.copy()
    results["implementation"] = results["implementation"].astype(str).str.strip()
    numeric = ["bodies", "seed", "repetition", "threads", "total_ms"]
    results[numeric] = results[numeric].apply(pd.to_numeric, errors="raise")

    unexpected = set(results["implementation"]) - set(IMPLEMENTATION_ORDER)
    if unexpected:
        raise ValueError(
            "unexpected implementations: " + ", ".join(sorted(unexpected))
        )
    if results["bodies"].le(0).any() or results["total_ms"].le(0).any():
        raise ValueError("body counts and timings must be positive")
    if results.isna().any().any():
        raise ValueError("benchmark CSV contains missing values")

    return results


def summarize(results: pd.DataFrame) -> pd.DataFrame:
    summary = (
        results.groupby(["implementation", "bodies"], as_index=False)
        .agg(
            measurements=("total_ms", "size"),
            families=("initial_condition", "nunique"),
            seeds=("seed", "nunique"),
            median_ms=("total_ms", "median"),
            q1_ms=("total_ms", lambda values: values.quantile(0.25)),
            q3_ms=("total_ms", lambda values: values.quantile(0.75)),
            min_ms=("total_ms", "min"),
            max_ms=("total_ms", "max"),
        )
        .sort_values(["implementation", "bodies"])
    )
    return summary


def crossover_bounds(summary: pd.DataFrame) -> tuple[int, int]:
    medians = summary.pivot(
        index="bodies", columns="implementation", values="median_ms"
    ).sort_index()
    gpu_wins = medians["cuda_tiled_force"] < medians["openmp_force"]
    winning_counts = medians.index[gpu_wins]
    if winning_counts.empty:
        raise ValueError("tiled CUDA never beats OpenMP in the measured range")

    first_gpu_win = int(winning_counts[0])
    earlier_counts = medians.index[medians.index < first_gpu_win]
    if earlier_counts.empty:
        raise ValueError("crossover occurs before the first measured body count")
    return int(earlier_counts[-1]), first_gpu_win


def create_figure(
    summary: pd.DataFrame,
    output_path: Path,
    lower_crossover: int,
    upper_crossover: int,
) -> None:
    sns.set_theme(style="whitegrid", context="notebook")
    colors = sns.color_palette("colorblind", n_colors=3)
    markers = {"serial_force": "o", "openmp_force": "s", "cuda_tiled_force": "^"}

    figure, axis = plt.subplots(figsize=(9.5, 6.0))
    for implementation, color in zip(IMPLEMENTATION_ORDER, colors):
        rows = summary[summary["implementation"] == implementation]
        axis.plot(
            rows["bodies"],
            rows["median_ms"],
            label=IMPLEMENTATION_LABELS[implementation],
            color=color,
            marker=markers[implementation],
            linewidth=2.2,
            markersize=7,
        )
        axis.fill_between(
            rows["bodies"],
            rows["q1_ms"],
            rows["q3_ms"],
            color=color,
            alpha=0.14,
            linewidth=0,
        )

    axis.axvspan(
        lower_crossover,
        upper_crossover,
        color="0.55",
        alpha=0.14,
        zorder=0,
    )
    crossover_y = summary["median_ms"].max() * 0.43
    axis.annotate(
        "Overall crossover region",
        xy=((lower_crossover * upper_crossover) ** 0.5, crossover_y),
        xytext=(2600, crossover_y * 1.9),
        arrowprops={"arrowstyle": "->", "color": "0.3", "linewidth": 1.2},
        color="0.25",
        ha="left",
    )

    body_counts = sorted(summary["bodies"].unique())
    axis.set_xscale("log", base=2)
    axis.set_yscale("log")
    axis.set_xticks(body_counts)
    axis.set_xticklabels([f"{count:,}" for count in body_counts])
    axis.set_xlabel("Number of bodies")
    axis.set_ylabel("Median force-evaluation time (ms)")
    axis.set_title("Paired exact-force CPU-CUDA crossover")
    axis.legend(title="Implementation", frameon=True)
    figure.text(
        0.11,
        0.015,
        "Lines show medians; shaded bands show the interquartile range\n"
        "across four initial-condition families, three seeds, and five repetitions.",
        fontsize=9,
        color="0.35",
        va="bottom",
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout(rect=(0.0, 0.07, 1.0, 1.0))
    figure.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def main() -> int:
    arguments = parse_arguments()
    try:
        results = load_results(arguments.input)
        summary = summarize(results)
        lower, upper = crossover_bounds(summary)

        arguments.summary_output.parent.mkdir(parents=True, exist_ok=True)
        summary.to_csv(arguments.summary_output, index=False)
        create_figure(summary, arguments.figure_output, lower, upper)

        print(summary.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
        print(f"Overall OpenMP-to-CUDA crossover: N={lower} to N={upper}")
        print(f"Summary written to: {arguments.summary_output}")
        print(f"Figure written to: {arguments.figure_output}")
    except (OSError, ValueError) as error:
        print(f"error: {error}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
