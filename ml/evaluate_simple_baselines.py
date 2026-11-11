"""Evaluate zero and training-mean acceleration predictors."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import torch

from ml.evaluate_baseline import empty_totals, evaluation_metrics
from ml.torch_dataset import make_dataloader
from ml.train_baseline import choose_device, error_sums, parse_body_counts


DEFAULT_DATASET = PROJECT_ROOT / "data" / "processed" / "candidate_v1"
DEFAULT_OUTPUT = PROJECT_ROOT / "results" / "simple_baselines" / "metrics.csv"


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate simple acceleration baselines.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--split", choices=["train", "validation", "test"], default="validation")
    parser.add_argument("--evaluation-group")
    parser.add_argument("--body-counts", type=parse_body_counts, default=[128])
    parser.add_argument("--fit-body-counts", type=parse_body_counts, default=[128])
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    return parser.parse_args()


def training_mean(dataset: Path, counts: list[int], device: torch.device) -> torch.Tensor:
    loader = make_dataloader(
        dataset,
        split="train",
        particle_counts=counts,
        max_snapshots=8,
        max_particles=4096,
        shuffle=False,
    )
    total = torch.zeros(3, dtype=torch.float64, device=device)
    bodies = 0
    for batch in loader:
        targets = batch["accelerations"].to(device=device, dtype=torch.float64)
        total += targets.sum(dim=0)
        bodies += len(targets)
    return (total / bodies).to(dtype=torch.float32)


def add_prediction(
    totals: dict[str, float], predictions: torch.Tensor, targets: torch.Tensor
) -> None:
    values = error_sums(predictions, targets)
    difference = predictions - targets
    # Huber beta=1, calculated elementwise to match the learned baselines.
    absolute = difference.abs()
    huber = torch.where(absolute < 1.0, 0.5 * difference.square(), absolute - 0.5)
    totals["loss"] += float(huber.sum().item())
    totals["target_squared"] += float(targets.square().sum().item())
    for key, value in values.items():
        totals[key] += value
    totals["snapshots"] += 1


def main() -> int:
    arguments = parse_arguments()
    try:
        device = choose_device(arguments.device)
        mean = training_mean(arguments.dataset, arguments.fit_body_counts, device)
        loader = make_dataloader(
            arguments.dataset,
            split=arguments.split,
            evaluation_group=arguments.evaluation_group,
            particle_counts=arguments.body_counts,
            max_snapshots=4,
            max_particles=4096,
            shuffle=False,
        )
        grouped: dict[tuple[str, str, int], dict[str, float]] = defaultdict(empty_totals)
        overall: dict[str, dict[str, float]] = {
            "zero": empty_totals(),
            "training_mean": empty_totals(),
        }
        with torch.inference_mode():
            for batch in loader:
                targets = batch["accelerations"].to(device, non_blocking=True)
                for system_index, metadata in enumerate(batch["metadata"]):
                    start = int(batch["ptr"][system_index].item())
                    end = int(batch["ptr"][system_index + 1].item())
                    system_targets = targets[start:end]
                    predictions = {
                        "zero": torch.zeros_like(system_targets),
                        "training_mean": mean.expand_as(system_targets),
                    }
                    for name, values in predictions.items():
                        key = (
                            name,
                            str(metadata["initial_condition"]),
                            int(metadata["particle_count"]),
                        )
                        add_prediction(grouped[key], values, system_targets)
                        add_prediction(overall[name], values, system_targets)

        rows = []
        for (name, family, count), totals in sorted(grouped.items()):
            rows.append(
                {
                    "baseline": name,
                    "split": arguments.split,
                    "evaluation_group": arguments.evaluation_group or "all",
                    "initial_condition": family,
                    "particle_count": count,
                    "snapshots": int(totals["snapshots"]),
                    **evaluation_metrics(totals),
                }
            )
        result = pd.DataFrame(rows)
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(arguments.output, index=False)
        summary = {
            "training_mean": mean.cpu().tolist(),
            "fit_body_counts": arguments.fit_body_counts,
            "evaluation": {
                name: evaluation_metrics(totals)
                for name, totals in overall.items()
            },
        }
        arguments.output.with_suffix(".json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )
        print(f"Training mean acceleration: {mean.cpu().tolist()}")
        print(result.to_string(index=False, float_format=lambda value: f"{value:.6g}"))
        print("\nOverall:")
        print(json.dumps(summary["evaluation"], indent=2))
        return 0
    except (OSError, ValueError, RuntimeError, KeyError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
