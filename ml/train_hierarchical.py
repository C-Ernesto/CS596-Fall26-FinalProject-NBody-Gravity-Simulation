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
from torch.nn import functional as F

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
    parser.add_argument(
        "--interaction-backend",
        choices=["python", "cpp", "cuda", "auto"],
        default="auto",
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=1.0e-4)
    parser.add_argument("--minimum-learning-rate", type=float, default=1.0e-6)
    parser.add_argument("--weight-decay", type=float, default=1.0e-6)
    parser.add_argument("--huber-beta", type=float, default=1.0)
    parser.add_argument(
        "--loss-mode",
        choices=["absolute_huber", "snapshot_relative_huber"],
        default="absolute_huber",
    )
    parser.add_argument(
        "--checkpoint-selection",
        choices=["loss", "mean_family_relative_l2", "worst_family_relative_l2"],
        default="loss",
    )
    parser.add_argument(
        "--maximum-family-regression",
        type=float,
        default=None,
        help=(
            "Reject a checkpoint when any trained family's validation relative L2 "
            "exceeds its initializer value by more than this fraction."
        ),
    )
    parser.add_argument("--max-snapshots", type=int, default=4)
    parser.add_argument("--max-particles", type=int, default=512)
    parser.add_argument("--training-snapshots-per-case", type=int, default=0)
    parser.add_argument("--validation-snapshots-per-case", type=int, default=0)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    return parser.parse_args()


def snapshot_losses(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    ptr: torch.Tensor,
    beta: float,
    mode: str,
) -> torch.Tensor:
    """Return one equally weighted loss value per complete simulation snapshot."""
    losses = []
    for start, end in zip(ptr[:-1].tolist(), ptr[1:].tolist()):
        prediction = predictions[start:end]
        target = targets[start:end]
        if mode == "snapshot_relative_huber":
            scale = target.square().mean().sqrt().detach().clamp_min(1.0e-6)
            prediction = prediction / scale
            target = target / scale
        losses.append(F.smooth_l1_loss(prediction, target, beta=beta))
    return torch.stack(losses)


def empty_totals() -> dict[str, float]:
    return {
        "loss": 0.0,
        "squared_error": 0.0,
        "target_squared": 0.0,
        "absolute_error": 0.0,
        "relative_vector_error": 0.0,
        "cosine": 0.0,
        "bodies": 0,
        "components": 0,
        "snapshots": 0,
    }


def finish_totals(totals: dict[str, float]) -> dict[str, float]:
    metrics = finalize_metrics(totals)
    metrics["loss"] = totals["loss"] / totals["snapshots"]
    metrics["relative_l2"] = (
        totals["squared_error"] / max(totals["target_squared"], 1.0e-30)
    ) ** 0.5
    return metrics


def selection_score(metrics: dict[str, float], selection: str) -> float:
    if selection == "loss":
        return metrics["loss"]
    return metrics[selection]


def maximum_family_regression(
    metrics: dict[str, dict[str, float]],
    initializer_metrics: dict[str, dict[str, float]],
) -> float:
    """Return the largest relative-L2 regression from the initializer by family."""
    return max(
        values["relative_l2"]
        / max(initializer_metrics[family]["relative_l2"], 1.0e-30)
        - 1.0
        for family, values in metrics.items()
    )


def run_epoch(
    model: HierarchicalForceModel,
    loader,
    huber_beta: float,
    loss_mode: str,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
) -> tuple[dict[str, float], dict[str, dict[str, float]]]:
    training = optimizer is not None
    model.train(training)
    totals = empty_totals()
    family_totals: dict[str, dict[str, float]] = {}
    for batch in loader:
        batch = move_batch(batch, device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            predictions = model(
                batch["positions"], batch["masses"], batch["epsilon"], batch["ptr"]
            )
            per_snapshot_loss = snapshot_losses(
                predictions,
                batch["accelerations"],
                batch["ptr"],
                huber_beta,
                loss_mode,
            )
            loss = per_snapshot_loss.mean()
            if training:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
                optimizer.step()
        detached = predictions.detach()
        for item, (start, end) in enumerate(
            zip(batch["ptr"][:-1].tolist(), batch["ptr"][1:].tolist())
        ):
            target = batch["accelerations"][start:end]
            values = error_sums(detached[start:end], target)
            values["target_squared"] = float(target.square().sum().item())
            family = str(batch["metadata"][item]["initial_condition"])
            family_values = family_totals.setdefault(family, empty_totals())
            for current in (totals, family_values):
                current["loss"] += float(per_snapshot_loss[item].detach().item())
                current["snapshots"] += 1
                for key, value in values.items():
                    current[key] += value
    family_metrics = {
        family: finish_totals(values) for family, values in sorted(family_totals.items())
    }
    metrics = finish_totals(totals)
    relative_l2_values = [values["relative_l2"] for values in family_metrics.values()]
    metrics["mean_family_relative_l2"] = sum(relative_l2_values) / len(
        relative_l2_values
    )
    metrics["worst_family_relative_l2"] = max(relative_l2_values)
    return metrics, family_metrics


def main() -> int:
    arguments = parse_arguments()
    try:
        if arguments.epochs <= 0 or arguments.theta <= 0.0:
            raise ValueError("epochs and theta must be positive")
        if not 0.0 < arguments.minimum_learning_rate <= arguments.learning_rate:
            raise ValueError("learning rates must satisfy 0 < minimum <= initial")
        if (
            arguments.maximum_family_regression is not None
            and arguments.maximum_family_regression < 0.0
        ):
            raise ValueError("maximum family regression cannot be negative")
        set_random_seed(arguments.seed)
        device = choose_device(arguments.device)
        initializer = torch.load(
            arguments.initializer, map_location=device, weights_only=False
        )
        base_configuration = initializer["configuration"]
        model = HierarchicalForceModel(
            hidden_dim=int(base_configuration["hidden_dim"]),
            hidden_layers=int(base_configuration["hidden_layers"]),
            theta=arguments.theta,
            leaf_capacity=arguments.leaf_capacity,
            interaction_backend=arguments.interaction_backend,
        ).to(device)
        initializer_kind = base_configuration.get("model", "pairwise")
        if initializer_kind == "pairwise":
            pairwise, _ = load_radial_model(arguments.initializer, device)
            model.load_pairwise_radial_network(pairwise)
        elif initializer_kind == "hierarchical":
            model.load_state_dict(initializer["model_state_dict"])
        else:
            raise ValueError(f"unsupported initializer model: {initializer_kind}")

        loader_arguments = {
            "root": arguments.dataset,
            "particle_counts": arguments.body_counts,
            "max_snapshots": arguments.max_snapshots,
            "max_particles": arguments.max_particles,
            "seed": arguments.seed,
        }
        train_loader = make_dataloader(
            split="train",
            shuffle=True,
            snapshots_per_case=arguments.training_snapshots_per_case,
            selection_seed=arguments.seed,
            **loader_arguments,
        )
        validation_loader = make_dataloader(
            split="validation",
            shuffle=False,
            snapshots_per_case=arguments.validation_snapshots_per_case,
            selection_seed=arguments.seed,
            **loader_arguments,
        )
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
        initial_metrics, initial_family_metrics = run_epoch(
            model,
            validation_loader,
            arguments.huber_beta,
            arguments.loss_mode,
            device,
            None,
        )
        print(
            f"initializer | validation loss {initial_metrics['loss']:.6g} | "
            f"MSE {initial_metrics['mse']:.6g} | mean-family relative L2 "
            f"{initial_metrics['mean_family_relative_l2']:.6g}"
        )
        for family, metrics in initial_family_metrics.items():
            print(f"  {family}: relative L2 {metrics['relative_l2']:.6g}")
        history_rows: list[dict[str, float | int]] = []
        best_validation = selection_score(
            initial_metrics, arguments.checkpoint_selection
        )
        best_epoch = 0
        best_metrics = initial_metrics
        best_family_metrics = initial_family_metrics
        checkpoint_path = arguments.output_dir / "best_model.pt"
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "configuration": configuration,
                "epoch": 0,
                "validation_metrics": initial_metrics,
                "validation_metrics_by_family": initial_family_metrics,
                "initializer_validation_metrics": initial_metrics,
                "initializer_validation_metrics_by_family": initial_family_metrics,
            },
            checkpoint_path,
        )
        started = time.perf_counter()
        for epoch in range(1, arguments.epochs + 1):
            train_loader.batch_sampler.set_epoch(epoch)
            train_metrics, train_family_metrics = run_epoch(
                model,
                train_loader,
                arguments.huber_beta,
                arguments.loss_mode,
                device,
                optimizer,
            )
            validation_metrics, validation_family_metrics = run_epoch(
                model,
                validation_loader,
                arguments.huber_beta,
                arguments.loss_mode,
                device,
                None,
            )
            row: dict[str, float | int] = {
                "epoch": epoch,
                "learning_rate": optimizer.param_groups[0]["lr"],
            }
            row.update({f"train_{key}": value for key, value in train_metrics.items()})
            row.update(
                {f"validation_{key}": value for key, value in validation_metrics.items()}
            )
            for family, metrics in train_family_metrics.items():
                row.update(
                    {
                        f"train_{family}_{key}": value
                        for key, value in metrics.items()
                    }
                )
            for family, metrics in validation_family_metrics.items():
                row.update(
                    {
                        f"validation_{family}_{key}": value
                        for key, value in metrics.items()
                    }
                )
            history_rows.append(row)
            current_validation = selection_score(
                validation_metrics, arguments.checkpoint_selection
            )
            family_regression = maximum_family_regression(
                validation_family_metrics, initial_family_metrics
            )
            eligible = (
                arguments.maximum_family_regression is None
                or family_regression <= arguments.maximum_family_regression
            )
            row["maximum_family_regression"] = family_regression
            row["checkpoint_eligible"] = eligible
            print(
                f"epoch {epoch:03d} | train loss {train_metrics['loss']:.6g} | "
                f"validation loss {validation_metrics['loss']:.6g} | "
                f"mean-family relative L2 "
                f"{validation_metrics['mean_family_relative_l2']:.6g} | "
                f"max-family regression {family_regression:+.1%} | "
                f"{'eligible' if eligible else 'rejected'}"
            )
            if eligible and current_validation < best_validation:
                best_validation = current_validation
                best_epoch = epoch
                best_metrics = validation_metrics
                best_family_metrics = validation_family_metrics
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "configuration": configuration,
                        "epoch": epoch,
                        "validation_metrics": validation_metrics,
                        "validation_metrics_by_family": validation_family_metrics,
                        "initializer_validation_metrics": initial_metrics,
                        "initializer_validation_metrics_by_family": (
                            initial_family_metrics
                        ),
                    },
                    checkpoint_path,
                )
            scheduler.step()

        elapsed = time.perf_counter() - started
        history = pd.DataFrame(history_rows)
        history.to_csv(arguments.output_dir / "history.csv", index=False)
        save_history_plot(history, arguments.figure, "Hierarchical fine-tuning loss")
        improved = best_epoch > 0
        summary = {
            "initializer_validation_metrics": initial_metrics,
            "initializer_validation_metrics_by_family": initial_family_metrics,
            "checkpoint_selection": arguments.checkpoint_selection,
            "initializer_selection_score": selection_score(
                initial_metrics, arguments.checkpoint_selection
            ),
            "best_selection_score": best_validation,
            "best_validation_metrics": best_metrics,
            "best_validation_metrics_by_family": best_family_metrics,
            "best_epoch": best_epoch,
            "improved_over_initializer": improved,
            "relative_selection_improvement": (
                (
                    selection_score(initial_metrics, arguments.checkpoint_selection)
                    - best_validation
                )
                / selection_score(initial_metrics, arguments.checkpoint_selection)
            ),
            "elapsed_seconds": elapsed,
        }
        (arguments.output_dir / "summary.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )
        if not improved:
            print("No epoch improved validation loss; using the epoch-0 checkpoint.")
        print(f"Best checkpoint: {checkpoint_path}")
        print(f"Training completed in {elapsed:.2f} seconds")
        return 0
    except (OSError, ValueError, RuntimeError, KeyError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
