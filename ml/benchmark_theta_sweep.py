"""Measure the large-N accuracy/work/speed tradeoff across opening angles."""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".matplotlib"))

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch

from ml.benchmark_crossover_matrix import FAMILIES, samples_from_family, summarize_exact
from ml.benchmark_hierarchical import load_model, run_once
from ml.train_baseline import choose_device, parse_body_counts


DEFAULT_MODELS = {
    "original": PROJECT_ROOT / "results" / "ml_hierarchical_geometry" / "best_model.pt",
    "guarded": (
        PROJECT_ROOT
        / "results"
        / "ml_hierarchical_balanced_guarded"
        / "best_model.pt"
    ),
}
DEFAULT_RAW_DIR = PROJECT_ROOT / "data" / "raw" / "crossover_matrix"
DEFAULT_EXACT = PROJECT_ROOT / "results" / "crossover_matrix" / "exact_per_run.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "results" / "theta_sweep"
DEFAULT_FIGURE = PROJECT_ROOT / "visualization" / "output" / "theta_sweep.png"


def parse_theta(value: str) -> list[float]:
    try:
        values = [float(item) for item in value.split(",") if item.strip()]
    except ValueError as error:
        raise argparse.ArgumentTypeError("theta must be comma-separated numbers") from error
    if not values or any(item <= 0.0 for item in values):
        raise argparse.ArgumentTypeError("theta values must be positive")
    return values


def parse_models(value: str) -> dict[str, Path]:
    models: dict[str, Path] = {}
    for item in value.split(","):
        if "=" not in item:
            raise argparse.ArgumentTypeError("models must use name=checkpoint entries")
        name, path = item.split("=", 1)
        name = name.strip()
        if not name or name in models:
            raise argparse.ArgumentTypeError("model names must be unique and nonempty")
        models[name] = Path(path.strip())
    if not models:
        raise argparse.ArgumentTypeError("at least one model is required")
    return models


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--body-counts", type=parse_body_counts, default=[32768])
    parser.add_argument("--theta", type=parse_theta, default=[0.5, 0.7, 1.0])
    parser.add_argument("--models", type=parse_models, default=DEFAULT_MODELS)
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--exact-results", type=Path, default=DEFAULT_EXACT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--figure", type=Path, default=DEFAULT_FIGURE)
    parser.add_argument("--seeds-per-case", type=int, default=1)
    parser.add_argument("--warmups", type=int, default=0)
    parser.add_argument("--repetitions", type=int, default=1)
    parser.add_argument("--leaf-capacity", type=int, default=16)
    parser.add_argument("--device", choices=["auto", "cuda"], default="auto")
    return parser.parse_args()


def load_selected_samples(
    raw_dir: Path, body_counts: list[int], seeds_per_case: int
) -> list[dict]:
    selected: list[dict] = []
    for family in FAMILIES:
        path = raw_dir / f"{family}.csv"
        if not path.is_file():
            raise FileNotFoundError(path)
        samples = samples_from_family(path, body_counts)
        for count in body_counts:
            candidates = sorted(
                (sample for sample in samples if sample["particle_count"] == count),
                key=lambda sample: int(sample["metadata"]["seed"]),
            )
            if len(candidates) < seeds_per_case:
                raise ValueError(
                    f"{family} N={count} has {len(candidates)} seeds; "
                    f"requested {seeds_per_case}"
                )
            selected.extend(candidates[:seeds_per_case])
        del samples
        gc.collect()
    return selected


def plot_summary(summary: pd.DataFrame, destination: Path) -> None:
    sns.set_theme(style="whitegrid")
    figure, axes = plt.subplots(1, 3, figsize=(17, 5))
    common = {
        "data": summary,
        "x": "theta",
        "hue": "initial_condition",
        "style": "model",
        "markers": True,
        "dashes": True,
    }
    sns.lineplot(**common, y="relative_l2", ax=axes[0])
    axes[0].set_yscale("log")
    axes[0].set_title("Force accuracy")
    axes[0].set_ylabel("Relative L2 force error")
    sns.lineplot(**common, y="learned_over_exact", ax=axes[1], legend=False)
    axes[1].axhline(1.0, color="black", linestyle="--", linewidth=1)
    axes[1].set_yscale("log")
    axes[1].set_title("End-to-end crossover")
    axes[1].set_ylabel("Sparse ML / exact CUDA time")
    sns.lineplot(**common, y="interaction_ratio", ax=axes[2], legend=False)
    axes[2].set_yscale("log")
    axes[2].set_title("Retained work")
    axes[2].set_ylabel("Retained interaction ratio")
    for axis in axes:
        axis.set_xlabel("Opening angle theta")
    figure.tight_layout()
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=180, bbox_inches="tight")
    plt.close(figure)


def main() -> int:
    arguments = parse_arguments()
    try:
        if arguments.seeds_per_case <= 0 or arguments.repetitions <= 0:
            raise ValueError("seed and repetition counts must be positive")
        if arguments.warmups < 0:
            raise ValueError("warmups cannot be negative")
        for name, checkpoint in arguments.models.items():
            if not checkpoint.is_file():
                raise FileNotFoundError(f"{name}: {checkpoint}")
        if not arguments.exact_results.is_file():
            raise FileNotFoundError(arguments.exact_results)

        samples = load_selected_samples(
            arguments.raw_dir, arguments.body_counts, arguments.seeds_per_case
        )
        device = choose_device(arguments.device)
        if device.type != "cuda":
            raise RuntimeError("theta sweep requires CUDA")

        records: list[dict[str, float | int | str]] = []
        total_cases = len(arguments.models) * len(arguments.theta) * len(samples)
        case_index = 0
        for model_name, checkpoint_path in arguments.models.items():
            model, checkpoint = load_model(
                checkpoint_path,
                device,
                arguments.theta[0],
                arguments.leaf_capacity,
                "cuda",
            )
            with torch.inference_mode():
                for theta in arguments.theta:
                    model.theta = theta
                    for sample in samples:
                        case_index += 1
                        for _ in range(arguments.warmups):
                            run_once(model, sample, device)
                        target = sample["accelerations"]
                        for repetition in range(1, arguments.repetitions + 1):
                            row, prediction = run_once(model, sample, device)
                            difference = prediction - target
                            squared_error = float(difference.square().sum().item())
                            target_squared = float(target.square().sum().item())
                            row.update(
                                {
                                    "model": model_name,
                                    "checkpoint_epoch": checkpoint.get("epoch"),
                                    "theta": theta,
                                    "seed": int(sample["metadata"]["seed"]),
                                    "repetition": repetition,
                                    "relative_l2": (
                                        squared_error / max(target_squared, 1.0e-30)
                                    )
                                    ** 0.5,
                                }
                            )
                            records.append(row)
                        print(
                            f"[{case_index}/{total_cases}] {model_name} theta={theta:g} "
                            f"{sample['metadata']['initial_condition']} "
                            f"N={sample['particle_count']} seed={sample['metadata']['seed']}",
                            flush=True,
                        )

        rows = pd.DataFrame(records)
        exact = summarize_exact(pd.read_csv(arguments.exact_results))
        paired = rows.merge(
            exact[["initial_condition", "particle_count", "seed", "exact_total_ms"]],
            on=["initial_condition", "particle_count", "seed"],
            validate="many_to_one",
        )
        paired["learned_over_exact"] = paired["end_to_end_ms"] / paired["exact_total_ms"]
        metrics = [
            "relative_l2",
            "learned_over_exact",
            "end_to_end_ms",
            "exact_total_ms",
            "interaction_ratio",
            "body_interactions",
            "aggregate_interactions",
            "total_interactions",
        ]
        keys = ["model", "theta", "initial_condition", "particle_count"]
        summary = paired.groupby(keys, as_index=False)[metrics].median()
        arguments.output_dir.mkdir(parents=True, exist_ok=True)
        paired.to_csv(arguments.output_dir / "per_run.csv", index=False)
        summary.to_csv(arguments.output_dir / "summary.csv", index=False)
        plot_summary(summary, arguments.figure)
        configuration = {
            "models": {name: str(path) for name, path in arguments.models.items()},
            "theta": arguments.theta,
            "body_counts": arguments.body_counts,
            "seeds_per_case": arguments.seeds_per_case,
            "warmups": arguments.warmups,
            "repetitions": arguments.repetitions,
            "exact_results": str(arguments.exact_results),
            "device": str(device),
            "gpu": torch.cuda.get_device_name(device),
        }
        (arguments.output_dir / "config.json").write_text(
            json.dumps(configuration, indent=2), encoding="utf-8"
        )
        print("\nTheta sweep summary:")
        print(summary.to_string(index=False, float_format=lambda value: f"{value:.6g}"))
        return 0
    except (FileNotFoundError, KeyError, OSError, RuntimeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
