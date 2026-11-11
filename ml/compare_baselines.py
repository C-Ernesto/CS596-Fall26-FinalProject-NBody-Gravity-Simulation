"""Combine simple, local-MLP, and pairwise baseline evaluation results."""

from __future__ import annotations

import json
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


SCENARIOS = {
    "Unseen seeds, N=128": "validation_metrics.json",
    "Unseen count, N=768": "interpolation_768_metrics.json",
    "Held-out disk, N=256": "heldout_disk_256_metrics.json",
}
MODEL_DIRECTORIES = {
    "Local particle MLP": PROJECT_ROOT / "results" / "ml_local_baseline",
    "Pairwise model": PROJECT_ROOT / "results" / "ml_baseline",
}
SIMPLE_DIRECTORY = PROJECT_ROOT / "results" / "simple_baselines"
DEFAULT_OUTPUT = PROJECT_ROOT / "results" / "baseline_comparison.csv"
DEFAULT_FIGURE = PROJECT_ROOT / "visualization" / "output" / "baseline_comparison.png"


def load_json(path: Path) -> dict:
    if not path.is_file():
        raise ValueError(f"missing baseline result: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    try:
        rows: list[dict] = []
        for scenario, filename in SCENARIOS.items():
            simple = load_json(SIMPLE_DIRECTORY / filename)["evaluation"]
            for key, label in [("zero", "Zero"), ("training_mean", "Training mean")]:
                rows.append({"scenario": scenario, "model": label, **simple[key]})
            for label, directory in MODEL_DIRECTORIES.items():
                learned = load_json(directory / filename)["overall"]
                rows.append({"scenario": scenario, "model": label, **learned})

        comparison = pd.DataFrame(rows)
        DEFAULT_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
        comparison.to_csv(DEFAULT_OUTPUT, index=False)

        sns.set_theme(style="whitegrid", context="notebook")
        figure, axis = plt.subplots(figsize=(11, 5.5))
        sns.barplot(
            data=comparison,
            x="scenario",
            y="mse_over_zero_predictor",
            hue="model",
            order=list(SCENARIOS),
            ax=axis,
        )
        axis.set_yscale("log")
        axis.axhline(1.0, color="black", linestyle="--", linewidth=1.1)
        axis.set_xlabel("")
        axis.set_ylabel("MSE / zero-predictor MSE (lower is better)")
        axis.set_title("Force-prediction baseline comparison")
        axis.tick_params(axis="x", rotation=10)
        axis.legend(title="Model")
        figure.tight_layout()
        DEFAULT_FIGURE.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(DEFAULT_FIGURE, dpi=200)
        plt.close(figure)

        print(
            comparison[
                [
                    "scenario", "model", "mse", "mse_over_zero_predictor",
                    "mean_relative_vector_error", "mean_cosine_similarity",
                ]
            ].to_string(index=False, float_format=lambda value: f"{value:.6g}")
        )
        print(f"Comparison written to: {DEFAULT_OUTPUT}")
        print(f"Plot written to: {DEFAULT_FIGURE}")
        return 0
    except (OSError, ValueError, KeyError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
