"""Restartable canonical-surface training for the six Transolver/PIBNet methods."""
from __future__ import annotations

import sys
from pathlib import Path
sys.path[:0] = [str(Path(__file__).resolve().parents[3] / p) for p in ("src", "src/shared", "src/PIBNet/backbone", "scripts")]

import hashlib
import json
import math
import os
import random
import time
import types

DEPENDENCIES = Path(os.environ.get('ESPC_DEPENDENCY_ROOT', str(Path(__file__).resolve().parents[3] / 'runtime')))
sys.path[:0] = [str(DEPENDENCIES / p) for p in ('scientific_deps', 'numeric_deps', 'point_deps')]
os.environ.setdefault('LOCAL_CACHE', str(DEPENDENCIES / 'scientific_deps/cache'))

import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / "runtime" / "numeric_deps"), str(ROOT / "src"),
                str(ROOT / 'src/PIBNet/backbone'), str(ROOT / "scripts")]

# The edge-index backend needs DGL annotations but does not execute DGL ops.
# PIBNet uses the edge-index backend, so its official
# module import only needs the DGLGraph annotation type; no DGL operation is
# called in forward, data loading, or training.
def install_dgl_type_shim() -> None:
    try:
        import dgl  # noqa: F401
        from dgl import DGLGraph  # noqa: F401
        return
    except Exception:
        for name in list(sys.modules):
            if name == "dgl" or name.startswith("dgl."):
                sys.modules.pop(name, None)
    dgl_stub = types.ModuleType("dgl")
    dgl_stub.__path__ = []
    dgl_stub.DGLGraph = type("DGLGraph", (), {})
    dgl_stub.DGLHeteroGraph = dgl_stub.DGLGraph
    dgl_stub.function = types.ModuleType("dgl.function")
    sys.modules["dgl"] = dgl_stub
    sys.modules["dgl.function"] = dgl_stub.function


install_dgl_type_shim()


from shared.paths import CACHE
from shared.point_reference_runtime import official_imports
official_imports()
TIMM_INITIALIZER_PATH = ROOT / 'src/shared/vendor/timm/weight_init.py'

from PIBNet.S_Z.official_pibnet_train import make_model as make_pibnet_core
from PIBNet.S_Z.official_pibnet_edge_backend import PIBNetEdgeIndex
from ESPC_Transolver.S_Z.model import CompositionalTransolver
from shared.learning import loss_and_scene_metrics
from Transolver.shared.canonical_model import GroupedTransolver


METHODS = ("espc-transolver", "transolver", "transolver-plus", "pibnet", "transolver-geo-matched", "transolver-geo-strong")
EPOCHS = 200
BATCH_SIZE = 16
LR = 1.0e-4
MIN_LR = 1.0e-7
WEIGHT_DECAY = 1.0e-5
LAMBDA_Q = 0.1
NODES = 128
_VERIFIED_CACHE_ROOTS: set[str] = set()


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, value) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_name(path.name + ".pending")
    pending.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(pending, path)


def save_torch(path: Path, value) -> None:
    pending = path.with_name(path.name + ".pending")
    torch.save(value, pending)
    os.replace(pending, path)


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


from shared.mixed20k_data import PreparedScenes, collate_scenes
from Transolver_GEO_strong.Point_cloud.model import GeoStrongTransolver
from Transolver_GEO_matched.Occupancy.model import CompositionalTransolver as GeoMatchedTransolver


def make_model(method: str):
    if method == "espc-transolver":
        return CompositionalTransolver()
    if method == "transolver-geo-matched":
        return GeoMatchedTransolver()
    if method == "transolver-geo-strong":
        return GeoStrongTransolver()
    if method == "transolver":
        return GroupedTransolver("transolver")
    if method == "transolver-plus":
        return GroupedTransolver("transolver-plus")
    if method == "pibnet":
        core = make_pibnet_core(torch.device("cpu"), node_dim=140)
        return PIBNetEdgeIndex(core)
    raise ValueError(method)


def get_split(cache: Path, name: str, method: str, stats: dict, shared: dict, points, normals):
    return PreparedScenes(cache, name, method, stats, shared, points, normals)


def make_loader(ds, *, shuffle: bool, seed: int, workers: int):
    gen = torch.Generator().manual_seed(seed)
    kwargs = {"dataset": ds, "batch_size": BATCH_SIZE, "shuffle": shuffle,
              "generator": gen, "num_workers": workers, "pin_memory": True,
              "collate_fn": collate_scenes, "drop_last": False}
    if workers:
        kwargs["prefetch_factor"] = 2
        kwargs["worker_init_fn"] = _loader_worker_init
    return DataLoader(**kwargs), gen


def _loader_worker_init(_worker_id):
    torch.set_num_threads(1)


def eval_split(model, loader, device, phi, stats, retain_scene_metrics=False):
    model.eval()
    values = {key: [] for key in ("scattered_rel_l2", "scattered_incident_normalized", "total_rel_l2",
                                  "q64_derived_rel_l2", "scattered_amplitude_rel_l2", "scattered_masked_phase_mae_deg")}
    weighted_loss = 0.0
    n = 0
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            loss, metrics = loss_and_scene_metrics(model(batch), batch, phi, stats, LAMBDA_Q)
            count = len(batch.sample_node_counts)
            weighted_loss += float(loss) * count
            n += count
            for key, value in metrics.items():
                values[key].extend(value.detach().cpu().tolist())
    result = {"scenes": n, "loss": weighted_loss / max(n, 1)}
    for key, items in values.items():
        a = np.asarray(items, dtype=np.float64)
        result[key] = {"mean": float(np.mean(a)), "median": float(np.median(a)),
                       "p95": float(np.quantile(a, 0.95))}
    if retain_scene_metrics:
        result["per_scene_metrics"] = values
    return result


def dataset_bundle(cache: Path, method: str):
    manifest = json.loads((cache / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("status") != "complete" or manifest.get("train_scene_count") != 20000:
        raise RuntimeError("prepared 20k cache is incomplete or has an unexpected scene count")
    cache_key = str(cache.resolve())
    if cache_key not in _VERIFIED_CACHE_ROOTS:
        for relative, expected in manifest["files_sha256"].items():
            actual = sha(cache / relative)
            if actual != expected:
                raise RuntimeError(f"remote cache checksum mismatch: {relative}")
        _VERIFIED_CACHE_ROOTS.add(cache_key)
    stats = json.loads((cache / "statistics.json").read_text(encoding="utf-8"))
    with np.load(cache / "shared.npz", allow_pickle=False) as shared_npz:
        shared = {k: shared_npz[k] for k in shared_npz.files}
    points = np.load(cache / "geometry_points.npy", mmap_mode="r", allow_pickle=False)
    normals = np.load(cache / "geometry_normals.npy", mmap_mode="r", allow_pickle=False)
    train_a = get_split(cache, "train_v3", method, stats, shared, points, normals)
    train_b = get_split(cache, "train_v6s11", method, stats, shared, points, normals)
    train = ConcatDataset([train_a, train_b])
    val_a = get_split(cache, "val_v3", method, stats, shared, points, normals)
    val_b = get_split(cache, "val_v6s11", method, stats, shared, points, normals)
    tests = {name: get_split(cache, name, method, stats, shared, points, normals)
             for name in ("id_test", "peanut_ood")}
    for name, ds, expected in (("train_v3", train_a, 10000), ("train_v6s11", train_b, 10000),
                               ("val_v3", val_a, 1000), ("val_v6s11", val_b, 1000),
                               ("id_test", tests["id_test"], 1000), ("peanut_ood", tests["peanut_ood"], 500)):
        if len(ds) != expected:
            raise RuntimeError(f"unexpected count for {name}: {len(ds)}")
    return manifest, stats, shared, train, (val_a, val_b), tests, (points, normals)


def run_one(method: str, args, cache_manifest, stats, shared, train, validations, points, normals, device):
    out = args.run_root
    out.mkdir(parents=True, exist_ok=True)
    seed_all(args.seed)
    model = make_model(method).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=MIN_LR)
    train_ds = train
    train_a, train_b = train.datasets
    train_loader, train_gen = make_loader(train_ds, shuffle=True, seed=args.seed, workers=args.workers)
    val_loaders = []
    for index, ds in enumerate(validations):
        loader, _ = make_loader(ds, shuffle=False, seed=args.seed + 500 + index, workers=args.workers)
        val_loaders.append(loader)
    phi = torch.as_tensor(shared["phi_pinv"], device=device)
    params = sum(p.numel() for p in model.parameters())
    code_hash = sha(Path(__file__))
    run_manifest = {"schema": "scattering-training", "method": method,
                    "seed": args.seed, "epochs": EPOCHS, "batch_size": BATCH_SIZE,
                    "optimizer": "AdamW", "learning_rate": LR, "minimum_learning_rate": MIN_LR,
                    "weight_decay": WEIGHT_DECAY, "gradient_clip": 1.0, "lambda_q": LAMBDA_Q,
                    "checkpoint_selection": "unweighted mean of v3 and v6s11 per-scene validation scattered relative L2 means",
                    "training_scenes": {"canonical-v3": 10000, "v6s11": 10000},
                    "validation_scenes": {"canonical-v3": 1000, "v6s11": 1000},
                    "exploratory_labels": True, "v6s11_independent_bem_validated": False,
                    "cache_manifest_sha256": sha(args.cache / "manifest.json"),
                    "cache_source_manifest_sha256": sha(args.cache / "manifest.json"),
                    "training_statistics_sha256": sha(args.cache / "statistics.json"),
                    "training_script_sha256": code_hash,
                    "timm_0_9_2_initializer_sha256": sha(TIMM_INITIALIZER_PATH),
                    "model_source_sha256": {
                        name: sha(ROOT / name) for name in (
                            'src/ESPC_Transolver/S_Z/model.py',
                            'src/ESPC_Transolver/shared/object_interaction.py',
                            'src/Transolver/shared/canonical_model.py',
                            'src/shared/learning.py',
                            'src/shared/mixed20k_data.py',
                            'src/Transolver_GEO_matched/Occupancy/model.py',
                            'src/Transolver_GEO_strong/Point_cloud/model.py',
                            'src/PIBNet/S_Z/official_pibnet_train.py',
                            'src/PIBNet/S_Z/official_pibnet_edge_backend.py',
                            'src/PIBNet/S_Z/official_pibnet_adapter.py',
                            'src/Transolver/backbone/PDE-Solving-StandardBenchmark/model/Transolver_Irregular_Mesh.py',
                            'src/Transolver/backbone/PDE-Solving-StandardBenchmark/model/Physics_Attention.py',
                            'src/Transolver_plus/backbone/models/Transolver_plus.py')},
                    "torch": torch.__version__,
                    "cuda": torch.version.cuda, "device": torch.cuda.get_device_name(device),
                    "parameters": params, "workers": args.workers}
    manifest_path = out / "run_manifest.json"
    if manifest_path.exists():
        saved = json.loads(manifest_path.read_text(encoding="utf-8"))
        if saved != run_manifest:
            raise RuntimeError(f"run provenance mismatch for {method}")
    else:
        write_json(manifest_path, run_manifest)
    last_path = out / "last.pt"
    history = []
    best = math.inf
    completed = 0
    elapsed_before = 0.0
    if last_path.exists() and args.resume:
        state = torch.load(last_path, map_location=device, weights_only=False)
        if state.get("run_manifest") != run_manifest:
            raise RuntimeError(f"resume checkpoint does not match run manifest for {method}")
        model.load_state_dict(state["model"], strict=True)
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        history, best, completed = state["history"], float(state["best_score"]), int(state["epoch"])
        elapsed_before = float(state.get("elapsed_seconds", 0.0))
        random.setstate(state["python_rng"])
        np.random.set_state(state["numpy_rng"])
        torch.set_rng_state(state["torch_rng"].cpu())
        torch.cuda.set_rng_state_all([x.cpu() for x in state["cuda_rng"]])
        train_loader.generator.set_state(state["loader_rng"].cpu())
    if completed >= EPOCHS:
        write_json(out / "status.json", {"method": method, "status": "complete", "epoch": EPOCHS,
                                         "best_score": best, "resumed_from_completed_checkpoint": True})
        return {"method": method, "status": "already_complete", "epochs": completed}

    start = time.perf_counter()
    for epoch in range(completed, EPOCHS):
        epoch_start = time.perf_counter()
        torch.cuda.reset_peak_memory_stats()
        train_a.set_epoch(epoch)
        train_b.set_epoch(epoch)
        model.train()
        train_loss = 0.0
        seen = 0
        for batch_index, batch in enumerate(train_loader, 1):
            batch = batch.to(device)
            prediction = model(batch)
            loss, _ = loss_and_scene_metrics(prediction, batch, phi, stats, LAMBDA_Q)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"nonfinite loss: {method} epoch {epoch + 1}")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            n = len(batch.sample_node_counts)
            train_loss += float(loss.detach()) * n
            seen += n
            if batch_index == 1 or batch_index % 100 == 0:
                progress = {"method": method, "status": "running", "epoch": epoch + 1,
                            "completed_epochs": epoch, "epochs": EPOCHS,
                            "batch": batch_index, "batches": len(train_loader), "scenes_seen": seen,
                            "epoch_elapsed_seconds": time.perf_counter() - epoch_start,
                            "updated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
                write_json(out / "status.json", progress)
                print(json.dumps({"event": "batch_progress", **progress}), flush=True)
        scheduler.step()
        val_results = [eval_split(model, loader, device, phi, stats) for loader in val_loaders]
        source_scores = [r["scattered_rel_l2"]["mean"] for r in val_results]
        score = float(np.mean(source_scores))
        record = {"epoch": epoch + 1, "train_loss": train_loss / max(seen, 1),
                  "lr": float(optimizer.param_groups[0]["lr"]),
                  "val_v3": val_results[0], "val_v6s11": val_results[1],
                  "selection_score": score, "best": score < best,
                  "epoch_seconds": time.perf_counter() - epoch_start,
                  "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
                  "elapsed_seconds": elapsed_before + time.perf_counter() - start}
        history.append(record)
        if score < best:
            best = score
            save_torch(out / "best.pt", {"model": model.state_dict(), "epoch": epoch + 1,
                                         "selection_score": best, "run_manifest": run_manifest})
        state = {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                 "scheduler": scheduler.state_dict(), "history": history, "best_score": best,
                 "epoch": epoch + 1, "elapsed_seconds": record["elapsed_seconds"],
                 "run_manifest": run_manifest, "python_rng": random.getstate(),
                 "numpy_rng": np.random.get_state(), "torch_rng": torch.get_rng_state(),
                 "cuda_rng": torch.cuda.get_rng_state_all(), "loader_rng": train_loader.generator.get_state()}
        save_torch(last_path, state)
        write_json(out / "history.json", history)
        write_json(out / "status.json", {"method": method, "status": "running", "epoch": epoch + 1,
                                         "epochs": EPOCHS, "best_score": best,
                                         "latest_v3_val_scattered_rel_l2": source_scores[0],
                                         "latest_v6s11_val_scattered_rel_l2": source_scores[1],
                                         "elapsed_seconds": record["elapsed_seconds"]})
        print(json.dumps({"event": "epoch_complete", "method": method, "epoch": epoch + 1,
                          "epochs": EPOCHS, "train_loss": record["train_loss"],
                          "val_v3": source_scores[0], "val_v6s11": source_scores[1],
                          "selection_score": score, "best_score": best,
                          "epoch_seconds": record["epoch_seconds"],
                          "peak_allocated_mib": record["peak_allocated_mib"],
                          "elapsed_seconds": record["elapsed_seconds"]}), flush=True)
    write_json(out / "status.json", {"method": method, "status": "complete", "epoch": EPOCHS,
                                     "best_score": best, "elapsed_seconds": elapsed_before + time.perf_counter() - start})
    return {"method": method, "status": "complete", "epochs": EPOCHS, "best_score": best}


def run(args, method):
    global EPOCHS, BATCH_SIZE
    EPOCHS, BATCH_SIZE = args.epochs, args.batch_size
    args.cache, args.run_root = CACHE, args.output
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required for training')
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision('high')
    device = torch.device('cuda')
    manifest, stats, shared, train, vals, tests, (points, normals) = dataset_bundle(args.cache, method)
    if args.smoke:
        seed_all(args.seed)
        model = make_model(method).to(device)
        for source in train.datasets:
            batch = collate_scenes([source[i] for i in range(BATCH_SIZE)]).to(device)
            model.train()
            model.zero_grad(set_to_none=True)
            prediction = model(batch)
            loss, _ = loss_and_scene_metrics(prediction, batch,
                torch.as_tensor(shared['phi_pinv'], device=device), stats, LAMBDA_Q)
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite smoke loss')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            print(json.dumps({'method': method, 'split': source.split,
                'prediction_shape': list(prediction.shape), 'loss': float(loss.detach())}), flush=True)
        return
    args.run_root.mkdir(parents=True, exist_ok=True)
    if (args.run_root/'last.pt').exists() and not args.resume:
        raise RuntimeError('Checkpoint exists: use --resume or a new --output directory')
    if args.resume and not (args.run_root/'last.pt').is_file():
        raise FileNotFoundError('No complete last.pt to resume')
    run_one(method, args, manifest, stats, shared, train, vals, points, normals, device)
