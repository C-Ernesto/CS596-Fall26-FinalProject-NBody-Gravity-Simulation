"""Analyze and plot the CUDA block-size sweep."""

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


DEFAULT_INPUT = PROJECT_ROOT / "results" / "cuda_block_sweep.csv"
DEFAULT_SUMMARY = PROJECT_ROOT / "results" / "cuda_block_sweep_summary.csv"
DEFAULT_BEST = PROJECT_ROOT / "results" / "cuda_block_sweep_best.csv"
DEFAULT_PLOT = PROJECT_ROOT / "visualization" / "output" / "cuda_block_sweep.png"
REQUIRED_COLUMNS = {
    "implementation",
    "bodies",
    "block_size",
    "time_ms",
    "kernel_ms",
}


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze a CUDA block-size sweep.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--summary-output", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--best-output", type=Path, default=DEFAULT_BEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_PLOT)
    return parser.parse_args()


def load_results(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise ValueError(f"block-sweep CSV does not exist: {path}")

    results = pd.read_csv(path)
    missing_columns = REQUIRED_COLUMNS - set(results.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"block-sweep CSV is missing columns: {missing}")

    results["implementation"] = results["implementation"].astype(str).str.strip()
    numeric_columns = ["bodies", "block_size", "time_ms", "kernel_ms"]
    results[numeric_columns] = results[numeric_columns].apply(
        pd.to_numeric,
        errors="raise",
    )

    if results.empty:
        raise ValueError(f"block-sweep CSV contains no rows: {path}")
    if results["implementation"].eq("").any():
        raise ValueError("implementation cannot be empty")
    if results[numeric_columns].le(0).any().any():
        raise ValueError("body counts, block sizes, and timings must be positive")
    return results


def summarize(results: pd.DataFrame) -> pd.DataFrame:
    summary = (
        results.groupby(
            ["implementation", "bodies", "block_size"],
            as_index=False,
        )
        .agg(
            runs=("time_ms", "size"),
            median_ms=("time_ms", "median"),
            min_ms=("time_ms", "min"),
            max_ms=("time_ms", "max"),
            stddev_ms=("time_ms", "std"),
            kernel_ms=("kernel_ms", "median"),
        )
        .sort_values(["implementation", "bodies", "block_size"])
    )
    summary["stddev_ms"] = summary["stddev_ms"].fillna(0.0)
    return summary


def select_best(summary: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return the fastest block per case and a robust overall choice per method."""
    best_indices = summary.groupby(["implementation", "bodies"])[
        "median_ms"
    ].idxmin()
    best_per_case = summary.loc[best_indices].sort_values(
        ["implementation", "bodies"]
    )

    normalized = summary.copy()
    normalized["relative_time"] = normalized["median_ms"] / normalized.groupby(
        ["implementation", "bodies"]
    )["median_ms"].transform("min")
    overall = (
        normalized.groupby(["implementation", "block_size"], as_index=False)
        .agg(mean_relative_time=("relative_time", "mean"))
        .sort_values(["implementation", "mean_relative_time"])
    )
    overall_indices = overall.groupby("implementation")[
        "mean_relative_time"
    ].idxmin()
    best_overall = overall.loc[overall_indices].sort_values("implementation")
    return best_per_case, best_overall


def create_plot(summary: pd.DataFrame, output_path: Path) -> None:
    sns.set_theme(style="whitegrid", context="notebook")
    body_counts = sorted(summary["bodies"].unique())
    implementations = sorted(summary["implementation"].unique())
    colors = sns.color_palette("colorblind", n_colors=len(implementations))
    palette = dict(zip(implementations, colors))

    figure, axes = plt.subplots(
        1,
        len(body_counts),
        figsize=(5 * len(body_counts), 5),
        sharey=False,
    )
    if len(body_counts) == 1:
        axes = [axes]

    for axis, bodies in zip(axes, body_counts):
        body_rows = summary[summary["bodies"] == bodies]
        sns.lineplot(
            data=body_rows,
            x="block_size",
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
            rows = body_rows[body_rows["implementation"] == implementation]
            axis.errorbar(
                rows["block_size"],
                rows["median_ms"],
                yerr=[
                    rows["median_ms"] - rows["min_ms"],
                    rows["max_ms"] - rows["median_ms"],
                ],
                fmt="none",
                color=palette[implementation],
                capsize=4,
            )

        axis.set_title(f"{bodies:,} bodies")
        axis.set_xlabel("CUDA block size")
        axis.set_ylabel("Median end-to-end time (ms)")
        axis.set_xticks(sorted(body_rows["block_size"].unique()))
        if axis is not axes[0]:
            axis.get_legend().remove()
        else:
            axis.legend(title="Implementation")

    figure.suptitle("CUDA block-size sweep")
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=200)
    plt.close(figure)


def main() -> int:
    arguments = parse_arguments()
    try:
        results = load_results(arguments.input)
        summary = summarize(results)
        best_per_case, best_overall = select_best(summary)

        for path in (arguments.summary_output, arguments.best_output, arguments.output):
            path.parent.mkdir(parents=True, exist_ok=True)

        summary.to_csv(arguments.summary_output, index=False)
        best_per_case.to_csv(arguments.best_output, index=False)
        create_plot(summary, arguments.output)

        print("Best block size for each body count:")
        print(
            best_per_case[
                ["implementation", "bodies", "block_size", "median_ms"]
            ].to_string(index=False, float_format=lambda value: f"{value:.4f}")
        )
        print("\nBest robust block size across tested body counts:")
        print(
            best_overall.to_string(
                index=False,
                float_format=lambda value: f"{value:.4f}",
            )
        )
        print(f"\nSummary written to: {arguments.summary_output}")
        print(f"Best configurations written to: {arguments.best_output}")
        print(f"Plot written to: {arguments.output}")
    except (OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
