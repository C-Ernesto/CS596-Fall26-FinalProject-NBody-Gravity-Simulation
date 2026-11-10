"""Summarize and plot CUDA N-body benchmark measurements."""

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
DEFAULT_PLOT = PROJECT_ROOT / "visualization" / "output" / "cuda_detailed_scaling.png"
DEFAULT_SUMMARY = PROJECT_ROOT / "results" / "cuda_detailed_scaling_summary.csv"
REQUIRED_COLUMNS = {"implementation", "bodies", "time_ms"}


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize and plot N-body benchmark results."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"Raw benchmark CSV (default: {DEFAULT_INPUT})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_PLOT,
        help=f"Output plot path (default: {DEFAULT_PLOT})",
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=DEFAULT_SUMMARY,
        help=f"Summary CSV path (default: {DEFAULT_SUMMARY})",
    )
    return parser.parse_args()


def load_results(path: Path) -> pd.DataFrame:
    """Read the raw timing rows and validate the columns used by the plot."""
    if not path.is_file():
        raise ValueError(f"benchmark CSV does not exist: {path}")

    results = pd.read_csv(path)
    missing_columns = REQUIRED_COLUMNS - set(results.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"benchmark CSV is missing columns: {missing}")

    results["implementation"] = results["implementation"].astype(str).str.strip()
    results["bodies"] = pd.to_numeric(results["bodies"], errors="raise")
    results["time_ms"] = pd.to_numeric(results["time_ms"], errors="raise")

    if results.empty:
        raise ValueError(f"benchmark CSV contains no data rows: {path}")
    if results["implementation"].eq("").any():
        raise ValueError("implementation cannot be empty")
    if results["bodies"].le(0).any():
        raise ValueError("bodies must be positive")
    if results["time_ms"].le(0).any():
        raise ValueError("time_ms must be positive")

    return results


def summarize(results: pd.DataFrame) -> pd.DataFrame:
    """Calculate descriptive statistics for each method and body count."""
    summary = (
        results.groupby(["implementation", "bodies"], as_index=False)
        .agg(
            runs=("time_ms", "size"),
            median_ms=("time_ms", "median"),
            min_ms=("time_ms", "min"),
            max_ms=("time_ms", "max"),
            stddev_ms=("time_ms", "std"),
        )
        .sort_values(["implementation", "bodies"])
    )

    # Standard deviation is undefined for a group containing only one run.
    summary["stddev_ms"] = summary["stddev_ms"].fillna(0.0)
    return summary


def warn_about_run_counts(summary: pd.DataFrame) -> None:
    run_counts = sorted(summary["runs"].unique())
    if len(run_counts) > 1:
        values = ", ".join(str(count) for count in run_counts)
        print(
            f"warning: configurations have different repetition counts: {values}",
            file=sys.stderr,
        )


def create_plot(summary: pd.DataFrame, output_path: Path) -> None:
    """Plot median runtime with minimum-to-maximum error bars."""
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
        lower_error = rows["median_ms"] - rows["min_ms"]
        upper_error = rows["max_ms"] - rows["median_ms"]
        axis.errorbar(
            rows["bodies"],
            rows["median_ms"],
            yerr=[lower_error, upper_error],
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

    output_path.parent.mkdir(parents=True, exist_ok=True)
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

        create_plot(summary, arguments.output)
        print(f"Plot written to: {arguments.output}")
    except (OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
