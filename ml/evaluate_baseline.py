"""Evaluate a trained pairwise baseline by family and particle count."""

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
from torch import nn

from ml.baseline_model import LocalParticleMLP, PairwiseForceBaseline
from ml.torch_dataset import make_dataloader
from ml.train_baseline import (
    choose_device,
    error_sums,
    finalize_metrics,
    parse_body_counts,
    predict,
)


DEFAULT_DATASET = PROJECT_ROOT / "data" / "processed" / "candidate_v1"
DEFAULT_CHECKPOINT = PROJECT_ROOT / "results" / "ml_baseline" / "best_model.pt"
DEFAULT_OUTPUT = PROJECT_ROOT / "results" / "ml_baseline" / "validation_metrics.csv"


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the pairwise force baseline.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--split", choices=["train", "validation", "test"], default="validation")
    parser.add_argument("--evaluation-group")
    parser.add_argument("--body-counts", type=parse_body_counts, default=[128])
    parser.add_argument("--max-snapshots", type=int, default=4)
    parser.add_argument("--max-particles", type=int, default=4096)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    return parser.parse_args()


def empty_totals() -> dict[str, float]:
    return {
        "loss": 0.0,
        "squared_error": 0.0,
        "absolute_error": 0.0,
        "relative_vector_error": 0.0,
        "cosine": 0.0,
        "target_squared": 0.0,
        "bodies": 0,
        "components": 0,
        "snapshots": 0,
    }


def add_system(
    totals: dict[str, float],
    predictions: torch.Tensor,
    targets: torch.Tensor,
    loss_function: nn.Module,
) -> None:
    values = error_sums(predictions, targets)
    totals["loss"] += float(loss_function(predictions, targets).item()) * values["components"]
    totals["target_squared"] += float(targets.square().sum().item())
    for key, value in values.items():
        totals[key] += value
    totals["snapshots"] += 1


def evaluation_metrics(totals: dict[str, float]) -> dict[str, float]:
    metrics = finalize_metrics(totals)
    metrics["loss"] = totals["loss"] / totals["components"]
    metrics["zero_predictor_mse"] = totals["target_squared"] / totals["components"]
    metrics["mse_over_zero_predictor"] = (
        metrics["mse"] / metrics["zero_predictor_mse"]
    )
    return metrics


def main() -> int:
    arguments = parse_arguments()
    try:
        device = choose_device(arguments.device)
        checkpoint = torch.load(
            arguments.checkpoint,
            map_location=device,
            weights_only=False,
        )
        configuration = checkpoint["configuration"]
        model_name = configuration.get("model", "pairwise")
        if model_name == "pairwise":
            model = PairwiseForceBaseline(
                hidden_dim=int(configuration["hidden_dim"]),
                hidden_layers=int(configuration["hidden_layers"]),
                target_chunk_size=int(configuration["target_chunk_size"]),
            ).to(device)
        else:
            model = LocalParticleMLP(
                hidden_dim=int(configuration["hidden_dim"]),
                hidden_layers=int(configuration["hidden_layers"]),
            ).to(device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        loss_function = nn.SmoothL1Loss(
            beta=float(configuration["huber_beta"])
        )
        loader = make_dataloader(
            arguments.dataset,
            split=arguments.split,
            evaluation_group=arguments.evaluation_group,
            particle_counts=arguments.body_counts,
            max_snapshots=arguments.max_snapshots,
            max_particles=arguments.max_particles,
            shuffle=False,
        )

        grouped: dict[tuple[str, int], dict[str, float]] = defaultdict(empty_totals)
        overall = empty_totals()
        with torch.inference_mode():
            for batch in loader:
                positions = batch["positions"].to(device, non_blocking=True)
                masses = batch["masses"].to(device, non_blocking=True)
                epsilon = batch["epsilon"].to(device, non_blocking=True)
                ptr = batch["ptr"].to(device, non_blocking=True)
                targets = batch["accelerations"].to(device, non_blocking=True)
                model_batch = {
                    "positions": positions,
                    "masses": masses,
                    "epsilon": epsilon,
                    "ptr": ptr,
                    "node_features": batch["node_features"].to(
                        device, non_blocking=True
                    ),
                }
                predictions = predict(model, model_name, model_batch)
                for system_index, metadata in enumerate(batch["metadata"]):
                    start = int(batch["ptr"][system_index].item())
                    end = int(batch["ptr"][system_index + 1].item())
                    system_predictions = predictions[start:end]
                    system_targets = targets[start:end]
                    key = (
                        str(metadata["initial_condition"]),
                        int(metadata["particle_count"]),
                    )
                    add_system(
                        grouped[key], system_predictions, system_targets, loss_function
                    )
                    add_system(overall, system_predictions, system_targets, loss_function)

        rows = []
        for (family, particle_count), totals in sorted(grouped.items()):
            metrics = evaluation_metrics(totals)
            rows.append(
                {
                    "split": arguments.split,
                    "evaluation_group": arguments.evaluation_group or "all",
                    "initial_condition": family,
                    "particle_count": particle_count,
                    "snapshots": int(totals["snapshots"]),
                    **metrics,
                }
            )
        result = pd.DataFrame(rows)
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(arguments.output, index=False)
        overall_metrics = evaluation_metrics(overall)
        summary = {
            "checkpoint": str(arguments.checkpoint),
            "checkpoint_epoch": checkpoint["epoch"],
            "split": arguments.split,
            "evaluation_group": arguments.evaluation_group,
            "body_counts": arguments.body_counts,
            "snapshots": int(overall["snapshots"]),
            "overall": overall_metrics,
        }
        summary_path = arguments.output.with_suffix(".json")
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(result.to_string(index=False, float_format=lambda value: f"{value:.6g}"))
        print("\nOverall:")
        print(json.dumps(overall_metrics, indent=2))
        print(f"Results written to: {arguments.output}")
        return 0
    except (OSError, ValueError, RuntimeError, KeyError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
