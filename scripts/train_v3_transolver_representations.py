"""Train matched one-seed Transolver input-representation comparisons."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("DGLBACKEND", "pytorch")
os.environ.setdefault("DGL_SKIP_GRAPHBOLT", "1")
sys.path[:0] = [str(ROOT / "work" / "python_deps"), str(ROOT / "src"),
                str(ROOT / "src" / "pibnet"), str(ROOT / "scripts")]
SITE = Path(r"C:\Users\27159\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\Lib\site-packages")
if SITE.is_dir():
    sys.path.append(str(SITE))

import train_v3_transolver_comparison as base

REPAIRED = ROOT / "work" / "python_deps_repaired"
sys.path.insert(0, str(REPAIRED))
import einops

from cspgnet.official_pibnet_edge_backend import collate_edge_index_graphs
from cspgnet.official_pibnet_train import seed_everything
from cspgnet.v3_learning import loss_and_scene_metrics
from cspgnet.v3_transolver_representations import make_dataset, make_model


TRAIN = ROOT / "datasets/canonical_response_v3/full/data/train"
VAL = ROOT / "datasets/canonical_response_v3/full/data/val"
STATS = ROOT / "outputs/v3_training_statistics_current_20260926.json"
S_CACHE = ROOT / "outputs/v3_feature_trials_20260927/cache/train"
S_CACHE_META = S_CACHE / "cache_manifest.json"
POINTCLOUD_CACHE = ROOT / "outputs/v3_main_single_run_20260927/pointcloud_cache"
S_RUN = ROOT / "outputs/v3_transolver_100ep_v2_20260929/transolver_seed0_100ep"
OUT = ROOT / "outputs/v3_transolver_peanut_repr_20260929"
ARMS = ("s-response", "raw-z", "geo-matched", "geo-strong")
EPOCHS = 100
BATCH_SIZE = 16
LR = 1.0e-4
WEIGHT_DECAY = 1.0e-5
LAMBDA_Q = 0.1


def sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_name(path.name + ".pending")
    pending.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(pending, path)


def save_torch(path: Path, value):
    pending = path.with_name(path.name + ".pending")
    torch.save(value, pending)
    os.replace(pending, path)


def safe_git_dirs():
    count = int(os.environ.get("GIT_CONFIG_COUNT", "0"))
    for index, repo in enumerate(("Transolver", "Transolver_plus"), count):
        os.environ[f"GIT_CONFIG_KEY_{index}"] = "safe.directory"
        os.environ[f"GIT_CONFIG_VALUE_{index}"] = str(ROOT / "third_party" / repo)
    os.environ["GIT_CONFIG_COUNT"] = str(count + 2)


def frozen_spec() -> dict:
    original = base.frozen_spec()
    if einops.__version__ != "0.8.1" or not Path(einops.__file__).resolve().is_relative_to(REPAIRED.resolve()):
        raise RuntimeError("the Transolver comparison requires the project's pinned einops 0.8.1")
    if len(list(TRAIN.glob("sample_*.npz"))) != 10000 or len(list(VAL.glob("sample_*.npz"))) != 1000:
        raise RuntimeError("expected the frozen 10000/1000 train/validation sets")
    s_spec = json.loads((ROOT / "outputs/v3_transolver_100ep_v2_20260929/trial_spec.json").read_text())
    s_manifest = json.loads((S_RUN / "run_manifest.json").read_text())
    s_history = json.loads((S_RUN / "history.json").read_text())
    s_checkpoint = torch.load(S_RUN / "best.pt", map_location="cpu", weights_only=False)
    if len(s_history) != EPOCHS or s_checkpoint.get("epoch") != EPOCHS:
        raise RuntimeError("the S(Z) comparison checkpoint is not a completed 100-epoch run")
    if s_manifest.get("arm") != "transolver" or s_manifest.get("seed") != 0:
        raise RuntimeError("S(Z) checkpoint is not the pinned Transolver seed-0 run")
    if (s_spec.get("train_manifest_sha256") != original["train_manifest_sha256"]
            or s_spec.get("val_manifest_sha256") != original["val_manifest_sha256"]
            or s_spec.get("statistics_sha256") != original["statistics_sha256"]):
        raise RuntimeError("S(Z) checkpoint data/normalization differs from the comparison data")
    if s_spec.get("training") != original.get("training"):
        raise RuntimeError("S(Z) checkpoint training protocol differs from the other arms")
    source_paths = [
        Path(__file__), ROOT / "src/cspgnet/v3_transolver_representations.py",
        ROOT / "scripts/train_v3_transolver_comparison.py",
        ROOT / "src/cspgnet/v3_transolver_comparison.py",
        ROOT / "src/cspgnet/v3_inputs.py", ROOT / "src/cspgnet/v3_pointcloud.py",
        ROOT / "src/cspgnet/v3_feature_trials.py", ROOT / "src/cspgnet/v3_learning.py",
        ROOT / "third_party/Transolver/PDE-Solving-StandardBenchmark/model/Transolver_Irregular_Mesh.py",
        ROOT / "third_party/Transolver/PDE-Solving-StandardBenchmark/model/Physics_Attention.py",
        ROOT / "third_party/Transolver/PDE-Solving-StandardBenchmark/model/Embedding.py",
    ]
    return {
        "schema": "v3-peanut-transolver-representation-comparison-v1",
        "status": "exploratory_single_seed; existing Peanut split previously viewed",
        "seed": 0,
        "arms": list(ARMS),
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "optimizer": "AdamW",
        "learning_rate": LR,
        "weight_decay": WEIGHT_DECAY,
        "scheduler": "CosineAnnealingLR T_max=100 eta_min=1e-7",
        "gradient_clip_norm": 1.0,
        "lambda_q": LAMBDA_Q,
        "output": "scattered complex field, identical official Transolver trunk (hidden=256, 5 layers, 8 heads, 32 slices)",
        "selection": "lowest validation mean per-scene scattered complex relative L2; no Peanut test-based checkpoint selection",
        "input_contract": "each representation maps to 128 real channels per canonical node; columns 133:136 are centered relative object centers for all arms",
        "representations": {
            "s-response": "train-only degree-normalized S=Phi H T_out derived from stored complex64 Z",
            "raw-z": "train-only degree-normalized Re/Im of stored complex64 Z",
            "geo-matched": "train-only standardized 128-channel radial occupancy",
            "geo-strong": "2048 area-sampled surface points, nearest-16 local encoder and max/mean global point encoder to 128 channels",
        },
        "train_manifest_sha256": sha(TRAIN / "manifest.json"),
        "val_manifest_sha256": sha(VAL / "manifest.json"),
        "statistics_sha256": sha(STATS),
        "s_cache_manifest_sha256": sha(S_CACHE_META),
        "s_response_checkpoint_sha256": sha(S_RUN / "best.pt"),
        "s_response_checkpoint_epoch": s_checkpoint["epoch"],
        "pointcloud_cache_root": str(POINTCLOUD_CACHE),
        "pointcloud_cache_existing_files": len(list(POINTCLOUD_CACHE.glob("*.npz"))),
        "original_scheme_a_spec_sha256": sha(ROOT / "outputs/v3_transolver_comparison_20260928/trial_spec.json"),
        "source_sha256": {str(path.relative_to(ROOT)): sha(path) for path in source_paths},
        "official_transolver_git_commit": original["official_git_commits"]["Transolver"],
        "einops": {"version": einops.__version__, "sha256": sha(REPAIRED / "einops" / "__init__.py")},
    }


def get_spec(freeze: bool) -> dict:
    safe_git_dirs()
    spec = frozen_spec()
    path = OUT / "trial_spec.json"
    if freeze:
        if path.exists():
            if json.loads(path.read_text(encoding="utf-8")) != spec:
                raise RuntimeError("frozen comparison spec changed; refusing to continue")
        else:
            save_json(path, spec)
    return spec


def make_loaders(arm: str, statistics: dict):
    train = make_dataset(arm, TRAIN, statistics, phase="train", s_cache=S_CACHE,
                         train_s_scales=json.loads(S_CACHE_META.read_text())["degree_complex_rms"],
                         pointcloud_cache=POINTCLOUD_CACHE)
    val = make_dataset(arm, VAL, statistics, phase="train", s_cache=S_CACHE.parent / "val",
                       train_s_scales=json.loads(S_CACHE_META.read_text())["degree_complex_rms"],
                       pointcloud_cache=POINTCLOUD_CACHE)
    if len(train) != 10000 or len(val) != 1000:
        raise RuntimeError("incomplete train/validation data")
    if not torch.allclose(train.phi_pinv, val.phi_pinv, atol=1.0e-5, rtol=1.0e-5):
        raise RuntimeError("train/validation projection bases differ")
    generator = torch.Generator().manual_seed(0)
    train_loader = DataLoader(train, batch_size=BATCH_SIZE, shuffle=True, generator=generator,
                              num_workers=0, collate_fn=collate_edge_index_graphs)
    val_loader = DataLoader(val, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=0, collate_fn=collate_edge_index_graphs)
    return train, val, train_loader, val_loader, generator


def profile(model, loader, phi_pinv, statistics, device):
    batch = next(iter(loader)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    times = []
    for _ in range(4):
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        prediction = model(batch)
        loss, _ = loss_and_scene_metrics(prediction, batch, phi_pinv, statistics, LAMBDA_Q)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if device.type == "cuda":
            torch.cuda.synchronize()
        times.append(time.perf_counter() - start)
    return {"step_seconds": times, "steady_median_seconds": float(np.median(times[1:])),
            "peak_cuda_mib": float(torch.cuda.max_memory_allocated() / 2**20) if device.type == "cuda" else None,
            "parameters": sum(parameter.numel() for parameter in model.parameters()),
            "batch_scenes": len(batch.sample_node_counts)}


def train_one(arm: str, *, resume: bool, profile_only: bool):
    if arm not in ARMS:
        raise ValueError(arm)
    spec = get_spec(freeze=not profile_only)
    statistics = json.loads(STATS.read_text(encoding="utf-8"))
    seed_everything(0)
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train, val, train_loader, val_loader, generator = make_loaders(arm, statistics)
    model = make_model(arm).to(device)
    phi_pinv = train.phi_pinv.to(device)
    if profile_only:
        print(json.dumps({"arm": arm, **profile(model, train_loader, phi_pinv, statistics, device)}), flush=True)
        return
    run = OUT / f"{arm}_seed0_100ep"
    run.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema": "v3-transolver-representation-run-v1", "arm": arm,
        "seed": 0, "device": str(device), "torch": torch.__version__,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "trial_spec_sha256": sha(OUT / "trial_spec.json"),
    }
    manifest_path, last_path = run / "run_manifest.json", run / "last.pt"
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1.0e-7)
    if resume:
        if not manifest_path.exists() or not last_path.exists():
            raise FileNotFoundError("--resume needs an existing run manifest and last.pt")
        if json.loads(manifest_path.read_text()) != manifest:
            raise RuntimeError("resume manifest mismatch")
        state = torch.load(last_path, map_location=device, weights_only=False)
        if state.get("manifest") != manifest:
            raise RuntimeError("resume checkpoint manifest mismatch")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        history, best = state["history"], float(state["best_score"])
        random.setstate(state["rng"]["python"])
        np.random.set_state(state["rng"]["numpy"])
        torch.set_rng_state(state["rng"]["torch"].cpu())
        if device.type == "cuda":
            torch.cuda.set_rng_state_all([value.cpu() for value in state["rng"]["cuda"]])
        generator.set_state(state["rng"]["loader"].cpu())
        first_epoch = int(state["epoch"]) + 1
        started = time.perf_counter() - history[-1]["elapsed_seconds"]
    else:
        if manifest_path.exists() or last_path.exists():
            raise FileExistsError("run already exists; use --resume only for the exact same run")
        save_json(manifest_path, manifest)
        history, best, first_epoch, started = [], float("inf"), 1, time.perf_counter()

    for epoch in range(first_epoch, EPOCHS + 1):
        train.set_epoch(epoch)
        train_metrics = base.run_epoch(model, train_loader, phi_pinv, statistics, LAMBDA_Q,
                                       "scattered", optimizer)
        val.set_epoch(0)
        val_metrics = base.fixed_validation(model, val_loader, phi_pinv, statistics, device)
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
            save_torch(run / "best.pt", state)
        save_json(run / "history.json", history)
        print(json.dumps(row, ensure_ascii=False), flush=True)
    print(json.dumps({"status": "complete", "arm": arm, "best_val": best,
                      "output": str(run)}, ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()
    if args.resume and args.profile:
        raise ValueError("--resume and --profile cannot be combined")
    train_one(args.arm, resume=args.resume, profile_only=args.profile)


if __name__ == "__main__":
    main()
