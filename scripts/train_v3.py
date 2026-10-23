"""Train v3 canonical-response PIBNet without opening sealed test splits.

Formal mode is fail-closed: it requires a frozen, passed protocol manifest.
``--development`` permits short train/validation smoke tests only; its output
must never be reported as a preregistered paper result.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

_project = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_project / "src"))
sys.path.insert(0, str(_project / "src" / "pibnet"))
_shared_site = Path(r"C:\Users\27159\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\Lib\site-packages")
if _shared_site.is_dir():
    sys.path.append(str(_shared_site))

from cspgnet.official_pibnet_train import make_model, seed_everything
from cspgnet.official_pibnet_edge_backend import PIBNetEdgeIndex, collate_edge_index_graphs
from cspgnet.v3_inputs import V3CanonicalDataset
from cspgnet.v3_learning import GeoStrongAdapted, loss_and_scene_metrics

import torch
from torch.utils.data import DataLoader


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_json(path: Path, value):
    pending = path.with_name(path.name + ".pending")
    pending.write_text(json.dumps(value, indent=2), encoding="utf-8")
    os.replace(pending, path)


def save_torch(path: Path, value):
    pending = path.with_name(path.name + ".pending")
    torch.save(value, pending)
    os.replace(pending, path)


def summarize(values):
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "p95": float(np.quantile(array, 0.95)),
    }


def run_epoch(model, loader, phi_pinv, statistics, lambda_q, output_head, optimizer=None):
    training = optimizer is not None
    model.train(training)
    collected = {}
    losses = []
    for graph in loader:
        graph = graph.to(next(model.parameters()).device)
        with torch.set_grad_enabled(training):
            prediction = model(graph)
            loss, metrics = loss_and_scene_metrics(
                prediction, graph, phi_pinv, statistics, lambda_q, output_head,
            )
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
        losses.extend([float(loss.detach())] * len(graph.sample_node_counts))
        for name, tensor in metrics.items():
            collected.setdefault(name, []).extend(tensor.detach().cpu().tolist())
    return {"loss": float(np.mean(losses)), "scenes": len(losses),
            **{name: summarize(values) for name, values in collected.items()}}


def args_parse():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--val", type=Path, required=True)
    parser.add_argument("--statistics", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--representation", choices=("z", "zero-z", "geo-matched", "geo-strong-adapted"), required=True)
    parser.add_argument("--output-head", choices=("scattered", "total"), default="scattered")
    parser.add_argument("--lambda-q", type=float, choices=(0.0, 0.1, 1.0), default=0.1)
    parser.add_argument("--seed", type=int, choices=range(5), default=0)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--loader-workers", type=int, choices=(0, 2), default=0)
    parser.add_argument("--pointcloud-cache", type=Path)
    parser.add_argument("--subset-file", type=Path)
    parser.add_argument("--subset-size", type=int, choices=(1000, 5000))
    parser.add_argument("--protocol", type=Path)
    parser.add_argument("--development", action="store_true")
    parser.add_argument("--limit-train", type=int, default=0)
    parser.add_argument("--limit-val", type=int, default=0)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main():
    args = args_parse()
    if args.representation != "z" and (args.lambda_q != 0.1 or args.output_head != "scattered"):
        raise ValueError("geometry/zero controls must use the common lambda_q=0.1 scattered protocol")
    if bool(args.subset_file) != bool(args.subset_size):
        raise ValueError("subset file and size must be supplied together")
    if args.subset_file and args.representation not in ("z", "geo-matched"):
        raise ValueError("E11 is preregistered only for Z and Geo-matched")
    if not args.development:
        if (args.representation not in ("z", "geo-matched", "geo-strong-adapted") or
                args.seed != 0 or args.lambda_q != 0.1 or
                args.output_head != "scattered" or args.subset_file):
            raise ValueError("formal main training allows three main models, one run with seed 0 only")
        if args.limit_train or args.limit_val or args.epochs != 100 or args.batch_size != 16:
            raise ValueError("formal protocol fixes full data, 100 epochs, batch 16")
        if args.protocol is None:
            raise RuntimeError("formal training requires a frozen protocol manifest")
        protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
        if protocol.get("status") != "frozen_and_passed" or not protocol.get("independent_bem_calibration_passed"):
            raise RuntimeError("local and independent BEM gates have not both passed")
        if protocol.get("budgeted_main_gpu_hours", float("inf")) > 120.0:
            raise RuntimeError("single-run main-phase GPU estimate exceeds the preagreed budget")
        if (protocol.get("main_phase_models") != ["z", "geo-matched", "geo-strong-adapted"] or
                protocol.get("main_phase_seeds") != [0]):
            raise RuntimeError("the frozen main-phase arm or single-run seed list differs")
        if args.loader_workers != protocol.get("loader_workers"):
            raise RuntimeError("data-loading worker count differs from the profiled frozen protocol")
        if (protocol.get("train_manifest_sha256") != digest(args.train / "manifest.json") or
                protocol.get("val_manifest_sha256") != digest(args.val / "manifest.json")):
            raise RuntimeError("train/validation manifests changed after protocol freeze")
        for name in (
            "scripts/train_v3.py",
            "src/cspgnet/v3_inputs.py",
            "src/cspgnet/v3_learning.py",
            "src/cspgnet/v3_pointcloud.py",
            "src/cspgnet/official_pibnet_adapter.py",
            "src/cspgnet/official_pibnet_edge_backend.py",
        ):
            if protocol.get("source_sha256", {}).get(name) != digest(_project / name):
                raise RuntimeError(f"training implementation changed after protocol freeze: {name}")
    statistics = json.loads(args.statistics.read_text(encoding="utf-8"))
    if statistics.get("source_train_manifest_sha256") != digest(args.train / "manifest.json"):
        raise RuntimeError("training statistics do not belong to this training data version")
    seed_everything(args.seed)
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    torch.set_float32_matmul_precision("high")
    train = V3CanonicalDataset(
        args.train, statistics, args.representation, args.output_head,
        args.pointcloud_cache, require_complete=not args.development,
    )
    val = V3CanonicalDataset(
        args.val, statistics, args.representation, args.output_head,
        args.pointcloud_cache, require_complete=not args.development,
    )
    if train.manifest["split"] != "train" or val.manifest["split"] != "val":
        raise ValueError("train/val paths are not the intended splits")
    if not args.development and (len(train) != 10000 or len(val) != 1000 or statistics["fit_samples"] != 10000):
        raise RuntimeError("formal train/val/statistics sizes do not match protocol")
    if args.subset_file:
        subset = json.loads(args.subset_file.read_text(encoding="utf-8"))
        if (subset.get("schema") != "canonical-response-v3-nested-subsets"
                or subset.get("train_manifest_sha256") != digest(args.train / "manifest.json")
                or not subset.get("nested")):
            raise RuntimeError("subset is not nested or belongs to another training version")
        indices = subset[f"train_{args.subset_size}"]
        if len(indices) != args.subset_size or len(set(indices)) != len(indices):
            raise RuntimeError("subset size or uniqueness check failed")
        train.files = [args.train / f"sample_{index:05d}.npz" for index in indices]
    if args.limit_train:
        train.files = train.files[:args.limit_train]
    if args.limit_val:
        val.files = val.files[:args.limit_val]
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(train, batch_size=args.batch_size, shuffle=True,
                              generator=generator, num_workers=args.loader_workers,
                              persistent_workers=bool(args.loader_workers),
                              collate_fn=collate_edge_index_graphs)
    val_loader = DataLoader(val, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.loader_workers,
                            persistent_workers=bool(args.loader_workers),
                            collate_fn=collate_edge_index_graphs)
    core = make_model(device, node_dim=140)
    model = (GeoStrongAdapted(core) if args.representation == "geo-strong-adapted"
             else PIBNetEdgeIndex(core)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-4, weight_decay=1.0e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1.0e-7)
    phi_pinv = train.phi_pinv.to(device)
    if not torch.allclose(train.phi_pinv, val.phi_pinv, atol=1.0e-5, rtol=1.0e-5):
        raise ValueError("train/validation projection basis differs")
    args.output.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output / "run_manifest.json"
    checkpoint_path = args.output / "last.pt"
    if not args.resume and (manifest_path.exists() or checkpoint_path.exists()):
        raise FileExistsError("run output already contains a checkpoint; use --resume for an identical run")
    if args.resume and (not manifest_path.exists() or not checkpoint_path.exists()):
        raise FileNotFoundError("resume requires an existing manifest and last.pt")
    try:
        git_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        git_commit = None
    manifest = {
        "schema": "canonical-response-v3-training",
        "development_only": bool(args.development),
        "args": {key: (False if key == "resume" else (str(value) if isinstance(value, Path) else value))
                 for key, value in vars(args).items()},
        "git_commit": git_commit,
        "statistics_sha256": digest(args.statistics),
        "train_manifest_sha256": digest(args.train / "manifest.json"),
        "val_manifest_sha256": digest(args.val / "manifest.json"),
        "protocol_sha256": digest(args.protocol) if args.protocol else None,
        "subset_sha256": digest(args.subset_file) if args.subset_file else None,
        "device": str(device),
        "torch": torch.__version__,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "selection": "validation mean per-scene scattered relative complex L2",
        "source_sha256": {
            str(path.relative_to(_project)): digest(path)
            for path in (
                Path(__file__),
                _project / "src/cspgnet/v3_inputs.py",
                _project / "src/cspgnet/v3_learning.py",
                _project / "src/cspgnet/v3_pointcloud.py",
                _project / "src/cspgnet/official_pibnet_adapter.py",
                _project / "src/cspgnet/official_pibnet_edge_backend.py",
            )
        },
    }
    if args.resume:
        original = json.loads(manifest_path.read_text(encoding="utf-8"))
        if original != manifest:
            raise RuntimeError("resume rejected: run arguments, code, data, statistics or device changed")
        state = torch.load(checkpoint_path, map_location=device, weights_only=False)
        if state["manifest"] != original or state["statistics"] != statistics:
            raise RuntimeError("resume checkpoint provenance changed")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        history = state.get("history")
        if not isinstance(history, list) or len(history) != state["epoch"]:
            raise RuntimeError("resume checkpoint lacks a complete atomic epoch history")
        if state["epoch"] > args.epochs:
            raise RuntimeError("checkpoint is beyond the requested training schedule")
        rng = state["rng"]
        random.setstate(rng["python"])
        np.random.set_state(rng["numpy"])
        torch.set_rng_state(rng["torch"].cpu())
        if device.type == "cuda":
            torch.cuda.set_rng_state_all([value.cpu() for value in rng["cuda"]])
        generator.set_state(rng["loader"].cpu())
        best = float(state["best_validation_scattered"])
        last_is_best = abs(history[-1]["val"]["scattered_rel_l2"]["mean"] - best) <= 1.0e-12
        best_path = args.output / "best.pt"
        if best_path.exists():
            best_saved = torch.load(best_path, map_location="cpu", weights_only=False)
            if best_saved["manifest"] != original:
                raise RuntimeError("best checkpoint belongs to another run")
            if best_saved["best_validation_scattered"] > best + 1.0e-12:
                if not last_is_best:
                    raise RuntimeError("best checkpoint is missing and last epoch is not the best")
                save_torch(best_path, state)
        else:
            if not last_is_best:
                raise RuntimeError("best checkpoint is missing and last epoch is not the best")
            save_torch(best_path, state)
        save_json(args.output / "history.json", history)
        first_epoch = state["epoch"] + 1
        started = time.perf_counter() - history[-1]["elapsed_seconds"]
        print(f"resuming identical run at epoch {first_epoch}/{args.epochs}", flush=True)
    else:
        save_json(manifest_path, manifest)
        best = float("inf")
        history = []
        first_epoch = 1
        started = time.perf_counter()
    for epoch in range(first_epoch, args.epochs + 1):
        train.set_epoch(epoch)
        train_metrics = run_epoch(model, train_loader, phi_pinv, statistics, args.lambda_q,
                                  args.output_head, optimizer)
        val.set_epoch(0)
        val_metrics = run_epoch(model, val_loader, phi_pinv, statistics, args.lambda_q,
                                args.output_head)
        scheduler.step()
        score = val_metrics["scattered_rel_l2"]["mean"]
        row = {"epoch": epoch, "train": train_metrics, "val": val_metrics,
               "learning_rate": scheduler.get_last_lr()[0], "elapsed_seconds": time.perf_counter() - started}
        history.append(row)
        state = {
            "epoch": epoch, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(), "manifest": manifest, "statistics": statistics,
            "best_validation_scattered": min(best, score),
            "history": history,
            "rng": {"python": random.getstate(), "numpy": np.random.get_state(),
                    "torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all() if device.type == "cuda" else [],
                    "loader": generator.get_state()},
        }
        save_torch(args.output / "last.pt", state)
        if score < best:
            best = score
            save_torch(args.output / "best.pt", state)
        save_json(args.output / "history.json", history)
        print(json.dumps(row, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
