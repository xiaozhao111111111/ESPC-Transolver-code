"""Train and evaluate the first CSP-GNet model on canonical-surface data."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset

from .data_generation.canonical_dataset import CanonicalSurfaceDataset
from .model import CSPGNet


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def metrics(prediction: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    pred_complex = torch.complex(prediction[..., 0], prediction[..., 1])
    target_complex = torch.complex(target[..., 0], target[..., 1])
    diff = pred_complex - target_complex
    denominator = target_complex.abs().square().sum().clamp_min(1.0e-12)
    rel_l2 = torch.sqrt(diff.abs().square().sum() / denominator)
    amplitude_denominator = target_complex.abs().square().sum().clamp_min(1.0e-12)
    amp_rel_l2 = torch.sqrt((pred_complex.abs() - target_complex.abs()).square().sum() / amplitude_denominator)
    mask = target_complex.abs() > 1.0e-3
    phase_error = torch.angle(pred_complex[mask] * target_complex[mask].conj()).abs()
    angle_mae_deg = phase_error.mean() * (180.0 / torch.pi) if mask.any() else torch.tensor(0.0, device=prediction.device)
    return {
        "rel_l2": float(rel_l2.detach().cpu()),
        "amp_rel_l2": float(amp_rel_l2.detach().cpu()),
        "angle_mae_deg": float(angle_mae_deg.detach().cpu()),
    }


def run_epoch(model, loader, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    sums = {"loss": 0.0, "rel_l2": 0.0, "amp_rel_l2": 0.0, "angle_mae_deg": 0.0}
    count = 0
    for batch in loader:
        physics = batch["physics"].to(device)
        incident = batch["incident"].to(device)
        centers = batch["centers"].to(device)
        k = batch["k"].to(device)
        target = batch["target"].to(device)
        with torch.set_grad_enabled(training):
            prediction = model(physics, incident, centers, k)
            loss = F.mse_loss(prediction, target)
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
        batch_metrics = metrics(prediction, target)
        batch_size = physics.shape[0]
        sums["loss"] += float(loss.detach().cpu()) * batch_size
        for name in batch_metrics:
            sums[name] += batch_metrics[name] * batch_size
        count += batch_size
    return {name: value / max(count, 1) for name, value in sums.items()}


def main(args: argparse.Namespace) -> None:
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    torch.set_float32_matmul_precision("high")
    full_train_set = CanonicalSurfaceDataset(args.train)
    if args.val_fraction > 0.0:
        if not 0.0 < args.val_fraction < 1.0:
            raise ValueError("val_fraction must be between 0 and 1")
        generator = torch.Generator().manual_seed(args.seed)
        indices = torch.randperm(len(full_train_set), generator=generator).tolist()
        split = int(len(indices) * (1.0 - args.val_fraction))
        train_set = Subset(full_train_set, indices[:split])
        test_set = Subset(full_train_set, indices[split:])
    else:
        train_set = full_train_set
        test_set = CanonicalSurfaceDataset(args.test)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=0)

    sample = train_set[0]
    interface = sample["interface_points"]
    model = CSPGNet(
        interface_points=interface,
        surface_layers=args.surface_layers,
        object_layers=args.object_layers,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1.0e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr_min)

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    history = []
    best = float("inf")
    print(f"device={device} train={len(train_set)} test={len(test_set)} params={sum(p.numel() for p in model.parameters())}")
    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(model, train_loader, device, optimizer)
        test_metrics = run_epoch(model, test_loader, device)
        scheduler.step()
        row = {"epoch": epoch, "lr": scheduler.get_last_lr()[0], "train": train_metrics, "test": test_metrics}
        history.append(row)
        if test_metrics["rel_l2"] < best:
            best = test_metrics["rel_l2"]
            torch.save({"model": model.state_dict(), "args": vars(args), "test": test_metrics}, output / "best.pt")
        if epoch == 1 or epoch % args.log_every == 0 or epoch == args.epochs:
            print(json.dumps(row, ensure_ascii=False))
    (output / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    (output / "final_metrics.json").write_text(json.dumps(history[-1], indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True)
    parser.add_argument("--test", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--val-fraction",
        type=float,
        default=0.0,
        help="If set, split the current training directory into train/validation subsets.",
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1.0e-3)
    parser.add_argument("--lr-min", type=float, default=1.0e-5)
    parser.add_argument("--surface-layers", type=int, default=3)
    parser.add_argument("--object-layers", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
