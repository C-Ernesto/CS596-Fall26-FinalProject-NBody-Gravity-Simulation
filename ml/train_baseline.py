"""Train the small-N fully connected pairwise force baseline."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
MATPLOTLIB_CONFIG = PROJECT_ROOT / ".matplotlib"
MATPLOTLIB_CONFIG.mkdir(exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MATPLOTLIB_CONFIG))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn

from ml.baseline_model import LocalParticleMLP, PairwiseForceBaseline
from ml.torch_dataset import make_dataloader


DEFAULT_DATASET = PROJECT_ROOT / "data" / "processed" / "candidate_v1"
DEFAULT_OUTPUT = PROJECT_ROOT / "results" / "ml_baseline"
DEFAULT_FIGURE = PROJECT_ROOT / "visualization" / "output" / "ml_baseline_history.png"


def parse_body_counts(value: str) -> list[int]:
    counts = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not counts or any(count <= 0 for count in counts):
        raise argparse.ArgumentTypeError("body counts must be positive comma-separated integers")
    return counts


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the pairwise force baseline.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--figure", type=Path)
    parser.add_argument("--model", choices=["pairwise", "local_mlp"], default="pairwise")
    parser.add_argument("--body-counts", type=parse_body_counts, default=[128])
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--minimum-learning-rate", type=float, default=1.0e-5)
    parser.add_argument("--weight-decay", type=float, default=1.0e-6)
    parser.add_argument("--huber-beta", type=float, default=1.0)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--hidden-layers", type=int, default=3)
    parser.add_argument("--target-chunk-size", type=int, default=128)
    parser.add_argument("--max-snapshots", type=int, default=4)
    parser.add_argument("--max-particles", type=int, default=512)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    return parser.parse_args()


def choose_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but PyTorch cannot access a CUDA device")
    return torch.device(requested)


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def move_batch(batch: dict, device: torch.device) -> dict:
    return {
        key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def predict(model: nn.Module, model_name: str, batch: dict) -> torch.Tensor:
    if model_name == "pairwise":
        return model(
            batch["positions"],
            batch["masses"],
            batch["epsilon"],
            batch["ptr"],
        )
    if model_name == "local_mlp":
        return model(batch["node_features"])
    raise ValueError(f"unknown model: {model_name}")


def error_sums(predictions: torch.Tensor, targets: torch.Tensor) -> dict[str, float]:
    difference = predictions - targets
    vector_error = torch.linalg.vector_norm(difference, dim=1)
    target_norm = torch.linalg.vector_norm(targets, dim=1)
    cosine = nn.functional.cosine_similarity(predictions, targets, dim=1, eps=1.0e-8)
    return {
        "squared_error": float(difference.square().sum().item()),
        "absolute_error": float(difference.abs().sum().item()),
        "relative_vector_error": float(
            (vector_error / target_norm.clamp_min(1.0e-6)).sum().item()
        ),
        "cosine": float(cosine.sum().item()),
        "bodies": len(targets),
        "components": targets.numel(),
    }


def finalize_metrics(sums: dict[str, float]) -> dict[str, float]:
    return {
        "mse": sums["squared_error"] / sums["components"],
        "mae": sums["absolute_error"] / sums["components"],
        "mean_relative_vector_error": sums["relative_vector_error"] / sums["bodies"],
        "mean_cosine_similarity": sums["cosine"] / sums["bodies"],
    }


def run_epoch(
    model: nn.Module,
    loader,
    loss_function: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    model_name: str = "pairwise",
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
            predictions = predict(model, model_name, batch)
            loss = loss_function(predictions, batch["accelerations"])
            if training:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
                optimizer.step()
        batch_sums = error_sums(predictions.detach(), batch["accelerations"])
        totals["loss"] += float(loss.item()) * batch_sums["components"]
        for key, value in batch_sums.items():
            totals[key] += value
    metrics = finalize_metrics(totals)
    metrics["loss"] = totals["loss"] / totals["components"]
    return metrics


def save_history_plot(
    history: pd.DataFrame,
    path: Path,
    loss_title: str = "Baseline loss",
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].plot(history["epoch"], history["train_loss"], label="Train")
    axes[0].plot(history["epoch"], history["validation_loss"], label="Validation")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Huber loss")
    axes[0].set_title(loss_title)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[1].plot(history["epoch"], history["validation_mse"], label="MSE")
    axes[1].plot(history["epoch"], history["validation_mae"], label="MAE")
    axes[1].set_yscale("log")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Normalized error")
    axes[1].set_title("Validation error")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    figure.tight_layout()
    figure.savefig(path, dpi=200)
    plt.close(figure)


def main() -> int:
    arguments = parse_arguments()
    try:
        if arguments.epochs <= 0 or arguments.learning_rate <= 0:
            raise ValueError("epochs and learning rate must be positive")
        if not 0 <= arguments.minimum_learning_rate <= arguments.learning_rate:
            raise ValueError("minimum learning rate must be between zero and learning rate")
        set_random_seed(arguments.seed)
        device = choose_device(arguments.device)
        if arguments.output_dir is None:
            arguments.output_dir = (
                DEFAULT_OUTPUT
                if arguments.model == "pairwise"
                else PROJECT_ROOT / "results" / "ml_local_baseline"
            )
        if arguments.figure is None:
            arguments.figure = (
                DEFAULT_FIGURE
                if arguments.model == "pairwise"
                else PROJECT_ROOT
                / "visualization"
                / "output"
                / "ml_local_baseline_history.png"
            )
        output_dir = arguments.output_dir.resolve()
        output_dir.mkdir(parents=True, exist_ok=True)

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
        if arguments.model == "pairwise":
            model = PairwiseForceBaseline(
                hidden_dim=arguments.hidden_dim,
                hidden_layers=arguments.hidden_layers,
                target_chunk_size=arguments.target_chunk_size,
            ).to(device)
        else:
            model = LocalParticleMLP(
                hidden_dim=arguments.hidden_dim,
                hidden_layers=arguments.hidden_layers,
            ).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=arguments.learning_rate,
            weight_decay=arguments.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=arguments.epochs,
            eta_min=arguments.minimum_learning_rate,
        )
        loss_function = nn.SmoothL1Loss(beta=arguments.huber_beta)

        configuration = vars(arguments).copy()
        configuration["dataset"] = str(arguments.dataset)
        configuration["output_dir"] = str(output_dir)
        configuration["figure"] = str(arguments.figure)
        configuration["device_used"] = str(device)
        configuration["torch_version"] = torch.__version__
        configuration["cuda_runtime"] = torch.version.cuda
        (output_dir / "config.json").write_text(
            json.dumps(configuration, indent=2), encoding="utf-8"
        )

        history_rows: list[dict] = []
        best_validation = float("inf")
        started = time.perf_counter()
        print(
            f"Training on {device}: train snapshots={len(train_loader.dataset)}, "
            f"validation snapshots={len(validation_loader.dataset)}"
        )
        for epoch in range(1, arguments.epochs + 1):
            train_loader.batch_sampler.set_epoch(epoch)
            train_metrics = run_epoch(
                model, train_loader, loss_function, device, optimizer, arguments.model
            )
            validation_metrics = run_epoch(
                model, validation_loader, loss_function, device, None, arguments.model
            )
            row = {"epoch": epoch}
            row["learning_rate"] = optimizer.param_groups[0]["lr"]
            row.update({f"train_{key}": value for key, value in train_metrics.items()})
            row.update(
                {f"validation_{key}": value for key, value in validation_metrics.items()}
            )
            history_rows.append(row)
            print(
                f"epoch {epoch:03d} | train loss {train_metrics['loss']:.6g} | "
                f"validation loss {validation_metrics['loss']:.6g} | "
                f"validation MSE {validation_metrics['mse']:.6g} | "
                f"cosine {validation_metrics['mean_cosine_similarity']:.6g}"
            )
            if validation_metrics["loss"] < best_validation:
                best_validation = validation_metrics["loss"]
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "configuration": configuration,
                        "epoch": epoch,
                        "validation_metrics": validation_metrics,
                    },
                    output_dir / "best_model.pt",
                )
            scheduler.step()

        elapsed = time.perf_counter() - started
        history = pd.DataFrame(history_rows)
        history.to_csv(output_dir / "history.csv", index=False)
        save_history_plot(history, arguments.figure.resolve())
        best_index = history["validation_loss"].idxmin()
        best_row = history.loc[best_index]
        summary = {
            "best_validation_loss": best_validation,
            "best_epoch": int(best_row["epoch"]),
            "best_metrics": {
                key.removeprefix("validation_"): float(value)
                for key, value in best_row.items()
                if key.startswith("validation_")
            },
            "elapsed_seconds": elapsed,
            "final_metrics": history.iloc[-1].to_dict(),
        }
        (output_dir / "summary.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )
        print(f"Training completed in {elapsed:.2f} seconds")
        print(f"Best checkpoint: {output_dir / 'best_model.pt'}")
        print(f"History: {output_dir / 'history.csv'}")
        return 0
    except (OSError, ValueError, RuntimeError, KeyError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
