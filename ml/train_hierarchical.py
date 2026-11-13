"""Fine-tune the hierarchical model's learned far-field interaction."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".matplotlib"))

import pandas as pd
import torch
from torch import nn

from ml.evaluate_hierarchical import load_radial_model
from ml.hierarchical_model import HierarchicalForceModel
from ml.torch_dataset import make_dataloader
from ml.train_baseline import (
    choose_device,
    error_sums,
    finalize_metrics,
    move_batch,
    parse_body_counts,
    save_history_plot,
    set_random_seed,
)


DEFAULT_DATASET = PROJECT_ROOT / "data" / "processed" / "candidate_v1"
DEFAULT_INITIALIZER = PROJECT_ROOT / "results" / "ml_baseline" / "best_model.pt"
DEFAULT_OUTPUT = PROJECT_ROOT / "results" / "ml_hierarchical_geometry"
DEFAULT_FIGURE = (
    PROJECT_ROOT / "visualization" / "output" / "ml_hierarchical_geometry_history.png"
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune learned aggregate-cell forces.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--initializer", type=Path, default=DEFAULT_INITIALIZER)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--figure", type=Path, default=DEFAULT_FIGURE)
    parser.add_argument("--body-counts", type=parse_body_counts, default=[128])
    parser.add_argument("--theta", type=float, default=1.0)
    parser.add_argument("--leaf-capacity", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=1.0e-4)
    parser.add_argument("--minimum-learning-rate", type=float, default=1.0e-6)
    parser.add_argument("--weight-decay", type=float, default=1.0e-6)
    parser.add_argument("--huber-beta", type=float, default=1.0)
    parser.add_argument("--max-snapshots", type=int, default=4)
    parser.add_argument("--max-particles", type=int, default=512)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    return parser.parse_args()


def run_epoch(
    model: HierarchicalForceModel,
    loader,
    loss_function: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    totals = {
        "loss": 0.0,
        "squared_error": 0.0,
        "absolute_error": 0.0,
        "relative_vector_error": 0.0,
        "cosine": 0.0,
        "bodies": 0,
        "components": 0,
    }
    for batch in loader:
        batch = move_batch(batch, device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            predictions = model(
                batch["positions"], batch["masses"], batch["epsilon"], batch["ptr"]
            )
            loss = loss_function(predictions, batch["accelerations"])
            if training:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
                optimizer.step()
        values = error_sums(predictions.detach(), batch["accelerations"])
        totals["loss"] += float(loss.item()) * values["components"]
        for key, value in values.items():
            totals[key] += value
    metrics = finalize_metrics(totals)
    metrics["loss"] = totals["loss"] / totals["components"]
    return metrics


def main() -> int:
    arguments = parse_arguments()
    try:
        if arguments.epochs <= 0 or arguments.theta <= 0.0:
            raise ValueError("epochs and theta must be positive")
        if not 0.0 < arguments.minimum_learning_rate <= arguments.learning_rate:
            raise ValueError("learning rates must satisfy 0 < minimum <= initial")
        set_random_seed(arguments.seed)
        device = choose_device(arguments.device)
        pairwise, initializer = load_radial_model(arguments.initializer, device)
        base_configuration = initializer["configuration"]
        model = HierarchicalForceModel(
            hidden_dim=int(base_configuration["hidden_dim"]),
            hidden_layers=int(base_configuration["hidden_layers"]),
            theta=arguments.theta,
            leaf_capacity=arguments.leaf_capacity,
        ).to(device)
        model.load_pairwise_radial_network(pairwise)

        loader_arguments = {
            "root": arguments.dataset,
            "particle_counts": arguments.body_counts,
            "max_snapshots": arguments.max_snapshots,
            "max_particles": arguments.max_particles,
            "seed": arguments.seed,
        }
        train_loader = make_dataloader(split="train", shuffle=True, **loader_arguments)
        validation_loader = make_dataloader(
            split="validation", shuffle=False, **loader_arguments
        )
        loss_function = nn.SmoothL1Loss(beta=arguments.huber_beta)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=arguments.learning_rate, weight_decay=arguments.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=arguments.epochs, eta_min=arguments.minimum_learning_rate
        )

        arguments.output_dir.mkdir(parents=True, exist_ok=True)
        configuration = vars(arguments).copy()
        for key in ("dataset", "initializer", "output_dir", "figure"):
            configuration[key] = str(configuration[key])
        configuration.update(
            {
                "model": "hierarchical",
                "aggregate_features": [
                    "log_softened_distance_squared",
                    "log_epsilon",
                    "log_cell_size_over_distance",
                    "log1p_cell_occupancy",
                    "log_rms_radius_over_distance",
                ],
                "hidden_dim": int(base_configuration["hidden_dim"]),
                "hidden_layers": int(base_configuration["hidden_layers"]),
                "device_used": str(device),
                "torch_version": torch.__version__,
                "cuda_runtime": torch.version.cuda,
            }
        )
        (arguments.output_dir / "config.json").write_text(
            json.dumps(configuration, indent=2), encoding="utf-8"
        )

        print(
            f"Fine-tuning on {device}: train snapshots={len(train_loader.dataset)}, "
            f"validation snapshots={len(validation_loader.dataset)}, theta={arguments.theta}"
        )
        initial_metrics = run_epoch(model, validation_loader, loss_function, device, None)
        print(
            f"initializer | validation loss {initial_metrics['loss']:.6g} | "
            f"MSE {initial_metrics['mse']:.6g}"
        )
        history_rows: list[dict[str, float | int]] = []
        best_validation = initial_metrics["loss"]
        best_epoch = 0
        started = time.perf_counter()
        for epoch in range(1, arguments.epochs + 1):
            train_loader.batch_sampler.set_epoch(epoch)
            train_metrics = run_epoch(model, train_loader, loss_function, device, optimizer)
            validation_metrics = run_epoch(model, validation_loader, loss_function, device, None)
            row: dict[str, float | int] = {
                "epoch": epoch,
                "learning_rate": optimizer.param_groups[0]["lr"],
            }
            row.update({f"train_{key}": value for key, value in train_metrics.items()})
            row.update(
                {f"validation_{key}": value for key, value in validation_metrics.items()}
            )
            history_rows.append(row)
            print(
                f"epoch {epoch:03d} | train loss {train_metrics['loss']:.6g} | "
                f"validation loss {validation_metrics['loss']:.6g} | "
                f"validation MSE {validation_metrics['mse']:.6g}"
            )
            if validation_metrics["loss"] < best_validation:
                best_validation = validation_metrics["loss"]
                best_epoch = epoch
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "configuration": configuration,
                        "epoch": epoch,
                        "validation_metrics": validation_metrics,
                        "initializer_validation_metrics": initial_metrics,
                    },
                    arguments.output_dir / "best_model.pt",
                )
            scheduler.step()

        elapsed = time.perf_counter() - started
        history = pd.DataFrame(history_rows)
        history.to_csv(arguments.output_dir / "history.csv", index=False)
        save_history_plot(history, arguments.figure, "Hierarchical fine-tuning loss")
        improved = best_epoch > 0
        summary = {
            "initializer_validation_metrics": initial_metrics,
            "best_validation_loss": best_validation,
            "best_epoch": best_epoch,
            "improved_over_initializer": improved,
            "relative_loss_improvement": (
                (initial_metrics["loss"] - best_validation) / initial_metrics["loss"]
            ),
            "elapsed_seconds": elapsed,
        }
        (arguments.output_dir / "summary.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )
        if not improved:
            print("No epoch improved validation loss; no fine-tuned checkpoint was kept.")
        else:
            print(f"Best checkpoint: {arguments.output_dir / 'best_model.pt'}")
        print(f"Training completed in {elapsed:.2f} seconds")
        return 0
    except (OSError, ValueError, RuntimeError, KeyError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
