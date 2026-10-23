"""Single-run, development-only Transolver-family comparison with Scheme A."""

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
import torch

os.environ.setdefault("DGLBACKEND", "pytorch")
os.environ.setdefault("DGL_SKIP_GRAPHBOLT", "1")
_tv_compat = torch.library.Library("torchvision", "DEF")
_tv_compat.define("nms(Tensor boxes, Tensor scores, float iou_threshold) -> Tensor")

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "work" / "python_deps"), str(ROOT / "src"),
                str(ROOT / "src" / "pibnet"), str(ROOT / "scripts")]
SITE = Path(r"C:\Users\27159\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\Lib\site-packages")
if SITE.is_dir():
    sys.path.append(str(SITE))

from torch.utils.data import DataLoader
from cspgnet.official_pibnet_edge_backend import collate_edge_index_graphs
from cspgnet.official_pibnet_train import seed_everything
from cspgnet.v3_learning import loss_and_scene_metrics
from cspgnet.v3_transolver_comparison import GroupedTransolver, STransformerDataset
from train_v3 import run_epoch, save_json, save_torch


TRAIN = ROOT / "datasets/canonical_response_v3/full/data/train"
VAL = ROOT / "datasets/canonical_response_v3/full/data/val"
STATS = ROOT / "outputs/v3_training_statistics_current_20260926.json"
OLD_TRIAL = ROOT / "outputs/v3_feature_trials_20260927"
OUT = ROOT / "outputs/v3_transolver_comparison_20260928"
ARMS = ("transolver", "transolver-plus")
BASE_A_FIRST40 = 0.23716858719289302
EPOCHS = 40
BATCH_SIZE = 16
SOURCE_PATHS = (
    "scripts/train_v3_transolver_comparison.py",
    "src/cspgnet/v3_transolver_comparison.py",
    "scripts/train_v3.py",
    "src/cspgnet/v3_feature_trials.py",
    "src/cspgnet/v3_learning.py",
    "src/cspgnet/official_pibnet_edge_backend.py",
    "third_party/Transolver/PDE-Solving-StandardBenchmark/model/Transolver_Irregular_Mesh.py",
    "third_party/Transolver/PDE-Solving-StandardBenchmark/model/Physics_Attention.py",
    "third_party/Transolver/PDE-Solving-StandardBenchmark/model/Embedding.py",
    "third_party/Transolver_plus/models/Transolver_plus.py",
)


def sha(path: Path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def committed_head(path: Path):
    return subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()


def specification():
    old = json.loads((OLD_TRIAL / "trial_spec.json").read_text(encoding="utf-8"))
    if old["source_sha256"]["src/cspgnet/v3_feature_trials.py"] != sha(ROOT / "src/cspgnet/v3_feature_trials.py"):
        raise RuntimeError("Scheme-A S converter changed after the prior trial")
    if sha(TRAIN / "manifest.json") != old["train_manifest_sha256"] or sha(VAL / "manifest.json") != old["val_manifest_sha256"]:
        raise RuntimeError("train/validation data changed since Scheme A")
    if sha(STATS) != old["statistics_sha256"]:
        raise RuntimeError("training statistics changed since Scheme A")
    if (sha(OLD_TRIAL / "cache/train/cache_manifest.json") != old["train_cache_manifest_sha256"] or
            sha(OLD_TRIAL / "cache/val/cache_manifest.json") != old["val_cache_manifest_sha256"]):
        raise RuntimeError("derived S caches changed since Scheme A")
    return {
        "schema": "v3-transolver-same-z-development-v1",
        "status": "development_only; sealed tests remain unopened for model selection",
        "arms": list(ARMS), "seed": 0, "epochs": EPOCHS, "batch_size": BATCH_SIZE,
        "train_scenes": 10000, "validation_scenes": 1000,
        "input": "same train-only standardized complex64-Z-derived S as Scheme A; 3 duplicate normal channels replaced by centered relative object positions already conveyed by PIBNet edges",
        "output": "nodewise complex scattered field, real/imag channels",
        "backbone": "pinned official Transolver irregular-mesh or Transolver++ Model, hidden=256 layers=5 heads=8 slices=32; scenes grouped by node count without padding; ++ single-GPU all_reduce identity",
        "training": "AdamW lr=1e-4 weight_decay=1e-5, cosine T_max=100 eta_min=1e-7, gradient_clip=1, lambda_q=0.1; batch 16",
        "validation": "seeded fixed Gumbel draws for repeatable Transolver++ evaluation; select lowest mean per-scene scattered complex relative L2",
        "baseline_a_best_first40": BASE_A_FIRST40,
        "continue_threshold_5pct_relative": BASE_A_FIRST40 * 0.95,
        "train_manifest_sha256": sha(TRAIN / "manifest.json"),
        "val_manifest_sha256": sha(VAL / "manifest.json"),
        "statistics_sha256": sha(STATS),
        "scheme_a_spec_sha256": sha(OLD_TRIAL / "trial_spec.json"),
        "train_S_cache_sha256": sha(OLD_TRIAL / "cache/train/cache_manifest.json"),
        "val_S_cache_sha256": sha(OLD_TRIAL / "cache/val/cache_manifest.json"),
        "official_git_commits": {
            "Transolver": committed_head(ROOT / "third_party/Transolver"),
            "Transolver_plus": committed_head(ROOT / "third_party/Transolver_plus"),
        },
        "source_sha256": {name: sha(ROOT / name) for name in SOURCE_PATHS},
    }


def frozen_spec():
    spec = specification()
    path = OUT / "trial_spec.json"
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != spec:
            raise RuntimeError("source, data, parameters or baseline changed after trial freeze")
    else:
        OUT.mkdir(parents=True, exist_ok=True)
        save_json(path, spec)
    return spec


def make_loaders(stats):
    cache = OLD_TRIAL / "cache"
    train_meta = json.loads((cache / "train/cache_manifest.json").read_text(encoding="utf-8"))
    train = STransformerDataset(TRAIN, stats, cache / "train", train_meta["degree_complex_rms"])
    val = STransformerDataset(VAL, stats, cache / "val", train_meta["degree_complex_rms"])
    if len(train) != 10000 or len(val) != 1000:
        raise RuntimeError("comparison requires original full train/validation splits")
    if not torch.allclose(train.phi_pinv, val.phi_pinv, atol=1e-5, rtol=1e-5):
        raise RuntimeError("projection basis differs between train and validation")
    generator = torch.Generator().manual_seed(0)
    train_loader = DataLoader(train, batch_size=BATCH_SIZE, shuffle=True, generator=generator,
                              num_workers=0, collate_fn=collate_edge_index_graphs)
    val_loader = DataLoader(val, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=0, collate_fn=collate_edge_index_graphs)
    return train, val, train_loader, val_loader, generator


def fixed_validation(model, loader, phi_pinv, stats, device):
    devices = [torch.cuda.current_device()] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(20260928)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(20260928)
        return run_epoch(model, loader, phi_pinv, stats, 0.1, "scattered")


def profile(model, loader, phi_pinv, stats, device):
    batch = next(iter(loader)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    timings = []
    for _ in range(6):
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        pred = model(batch)
        loss, _ = loss_and_scene_metrics(pred, batch, phi_pinv, stats, 0.1)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if device.type == "cuda":
            torch.cuda.synchronize()
        timings.append(time.perf_counter() - start)
    return {"profile_step_seconds_first": timings[0],
            "profile_step_seconds_steady_median": float(np.median(timings[1:])),
            "batch_scenes": len(batch.sample_node_counts),
            "max_scene_nodes": int(batch.sample_node_counts.max()),
            "peak_cuda_mib": (torch.cuda.max_memory_allocated() / 2**20 if device.type == "cuda" else None),
            "loss": float(loss.detach())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", required=True, choices=ARMS)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    if args.profile and args.resume:
        raise ValueError("--profile and --resume are mutually exclusive")
    spec = specification() if args.profile else frozen_spec()
    stats = json.loads(STATS.read_text(encoding="utf-8"))
    seed_everything(0)
    torch.set_float32_matmul_precision("high")
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    train, val, train_loader, val_loader, generator = make_loaders(stats)
    model = GroupedTransolver(args.arm).to(device)
    phi_pinv = train.phi_pinv.to(device)
    parameters = sum(p.numel() for p in model.parameters())
    if args.profile:
        result = {"arm": args.arm, "parameters": parameters,
                  **profile(model, train_loader, phi_pinv, stats, device)}
        print(json.dumps(result), flush=True)
        return
    output = OUT / f"{args.arm}_seed0_40ep"
    output.mkdir(parents=True, exist_ok=True)
    manifest = {"schema": "v3-transolver-comparison-run-v1", "development_only": True,
                "arm": args.arm, "seed": 0, "device": str(device), "torch": torch.__version__,
                "parameters": parameters, "trial_spec_sha256": sha(OUT / "trial_spec.json")}
    manifest_path = output / "run_manifest.json"
    last_path = output / "last.pt"
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-7)
    if args.resume:
        if json.loads(manifest_path.read_text(encoding="utf-8")) != manifest:
            raise RuntimeError("resume manifest mismatch")
        state = torch.load(last_path, map_location=device, weights_only=False)
        if state["manifest"] != manifest:
            raise RuntimeError("resume checkpoint mismatch")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        history, best = state["history"], state["best_score"]
        random.setstate(state["rng"]["python"])
        np.random.set_state(state["rng"]["numpy"])
        torch.set_rng_state(state["rng"]["torch"].cpu())
        if device.type == "cuda":
            torch.cuda.set_rng_state_all([x.cpu() for x in state["rng"]["cuda"]])
        generator.set_state(state["rng"]["loader"].cpu())
        first_epoch = state["epoch"] + 1
        started = time.perf_counter() - history[-1]["elapsed_seconds"]
    else:
        if manifest_path.exists() or last_path.exists():
            raise FileExistsError("run already exists; use --resume")
        save_json(manifest_path, manifest)
        history, best, first_epoch = [], float("inf"), 1
        started = time.perf_counter()
    for epoch in range(first_epoch, EPOCHS + 1):
        train.set_epoch(epoch)
        train_metrics = run_epoch(model, train_loader, phi_pinv, stats, 0.1, "scattered", optimizer)
        val.set_epoch(0)
        val_metrics = fixed_validation(model, val_loader, phi_pinv, stats, device)
        scheduler.step()
        score = val_metrics["scattered_rel_l2"]["mean"]
        row = {"epoch": epoch, "train": train_metrics, "val": val_metrics,
               "learning_rate": scheduler.get_last_lr()[0],
               "elapsed_seconds": time.perf_counter() - started}
        history.append(row)
        state = {"epoch": epoch, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                 "scheduler": scheduler.state_dict(), "manifest": manifest,
                 "history": history, "best_score": min(best, score),
                 "rng": {"python": random.getstate(), "numpy": np.random.get_state(),
                         "torch": torch.get_rng_state(),
                         "cuda": torch.cuda.get_rng_state_all() if device.type == "cuda" else [],
                         "loader": generator.get_state()}}
        save_torch(last_path, state)
        if score < best:
            best = score
            save_torch(output / "best.pt", state)
        save_json(output / "history.json", history)
        print(json.dumps(row, ensure_ascii=False), flush=True)
    print(json.dumps({"arm": args.arm, "best_val_scattered_rel_l2": best,
                      "threshold_5pct_relative": spec["continue_threshold_5pct_relative"]}), flush=True)


if __name__ == "__main__":
    main()
