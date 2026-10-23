"""Restartable seed-0, 200-epoch training/evaluation for M0--M6 on the
verified mixed canonical-v3 + v6s11 prepared cache.

The cache is immutable. All run state is written beneath --run-root. The v6s11
labels remain exploratory in every output manifest and report.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import json
import math
import os
import random
import sys
import time
import types
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "work" / "python_deps_repaired"), str(ROOT / "src"),
                str(ROOT / "src" / "pibnet"), str(ROOT / "scripts")]

# DGL 2.1's GraphBolt extension is not built for the AutoDL Torch 2.8 wheel.
# PIBNet M3 uses the already parity-tested edge-index backend, so its official
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
    dgl_stub.DGLGraph = type("DGLGraph", (), {})
    dgl_stub.DGLHeteroGraph = dgl_stub.DGLGraph
    sys.modules["dgl"] = dgl_stub


install_dgl_type_shim()


def install_pointset_model_leaf() -> None:
    """Load the existing model statements without unused FEM dataset imports.

    The original comparison module also defines a dataset which imports Netgen
    and NGSolve. This prepared-cache trainer needs only its model wrapper. Keep
    the wrapper and official-model loading functions byte-for-byte at AST level.
    """
    name = "cspgnet.v3_transolver_comparison"
    path = ROOT / "src" / "cspgnet" / "v3_transolver_comparison.py"
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    wanted = {"PROJECT", "BASE_ROOT", "PLUS_SOURCE", "NODES_PER_OBJECT",
              "_load_official_module", "_SingleProcessDistributed", "GroupedTransolver"}
    selected = []
    found = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            identifiers = {node.name}
        elif isinstance(node, ast.Assign):
            identifiers = {t.id for t in node.targets if isinstance(t, ast.Name)}
        else:
            continue
        if identifiers & wanted:
            selected.append(node)
            found.update(identifiers & wanted)
    if found != wanted:
        raise ImportError(f"point-set model definitions changed: {wanted - found}")
    module = types.ModuleType(name)
    module.__file__ = str(path)
    module.__dict__.update({"Path": Path, "np": np, "torch": torch,
                            "nn": torch.nn, "sys": sys, "importlib": importlib})
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), "exec"), module.__dict__)
    sys.modules[name] = module


install_pointset_model_leaf()


def install_timm_initializer_leaf() -> Path:
    """Expose the pinned timm 0.9.2 initializer used by both official trunks."""
    path = ROOT / "work" / "timm_0_9_2_weight_init.py"
    if not path.exists():
        path = ROOT / "work" / "subset11_train_deps" / "timm" / "layers" / "weight_init.py"
    spec = importlib.util.spec_from_file_location("_mixed20k_timm_weight_init", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    leaf = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(leaf)
    for name in ("timm", "timm.layers", "timm.models", "timm.models.layers"):
        package = types.ModuleType(name)
        package.__path__ = []
        package.__file__ = str(path)
        package.trunc_normal_ = leaf.trunc_normal_
        sys.modules[name] = package
    sys.modules["timm"].__version__ = "0.9.2"
    return path


TIMM_INITIALIZER_PATH = install_timm_initializer_leaf()

from cspgnet.official_pibnet_train import make_model as make_pibnet_core
from cspgnet.official_pibnet_edge_backend import PIBNetEdgeIndex
from cspgnet.official_pibnet_adapter import (
    EDGE_DIM, EDGE_TYPES, _edge_features_from_cached_geometry,
    _edge_features_from_delta, _get_topology, _paper_sparse_distant_edges,
)
from cspgnet.v3_compositional_transolver import CompositionalTransolver
from cspgnet.v3_learning import loss_and_scene_metrics
from cspgnet.v3_transolver_comparison import GroupedTransolver


METHODS = ("M0", "M1", "M2", "M3", "M4", "M5", "M6")
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


@dataclass
class Scene:
    node_features: torch.Tensor
    labels: torch.Tensor
    incident: torch.Tensor
    q64_gt: torch.Tensor
    z_real: torch.Tensor
    z_imag: torch.Tensor
    sample_index: int
    source_code: int
    edge_features: dict | None = None
    edge_index: dict | None = None
    level_masks: dict | None = None
    geo_local: torch.Tensor | None = None
    geo_global: torch.Tensor | None = None

    def num_nodes(self):
        return int(self.node_features.shape[0])


@dataclass
class Batch:
    node_features: torch.Tensor
    labels: torch.Tensor
    incident: torch.Tensor
    q64_gt: torch.Tensor
    z_real: torch.Tensor
    z_imag: torch.Tensor
    batch_index: torch.Tensor
    sample_index: torch.Tensor
    sample_node_counts: torch.Tensor
    edge_features: dict
    edge_index: dict
    level_masks: dict
    geo_local: torch.Tensor | None = None
    geo_global: torch.Tensor | None = None

    def to(self, device):
        return Batch(**{
            key: (value if key == "sample_node_counts" else
                  value.to(device, non_blocking=True) if torch.is_tensor(value) else
                  {k: v.to(device, non_blocking=True) for k, v in value.items()} if isinstance(value, dict) else value)
            for key, value in self.__dict__.items()
        })

    def pin_memory(self):
        return Batch(**{
            key: (value.pin_memory() if torch.is_tensor(value) else
                  {k: v.pin_memory() for k, v in value.items()} if isinstance(value, dict) else value)
            for key, value in self.__dict__.items()
        })


class PreparedScenes(Dataset):
    def __init__(self, cache: Path, split: str, method: str, stats: dict,
                 shared: dict, geometry_points=None, geometry_normals=None):
        self.root = Path(cache) / split
        self.split = split
        self.method = method
        self.stats = stats
        self.interface = np.asarray(shared["interface_points"], dtype=np.float32)
        self.phi_pinv = np.asarray(shared["phi_pinv"], dtype=np.complex64)
        self.geometry_points = geometry_points
        self.geometry_normals = geometry_normals
        self.arrays = {name: np.load(self.root / f"{name}.npy", mmap_mode="r", allow_pickle=False)
                       for name in ("features", "labels", "incident", "q64", "z_real", "z_imag",
                                    "occupancy", "geometry_index", "rotations", "nearest_indices",
                                    "object_offsets", "source_code", "source_sample_index")}
        self.offsets = self.arrays["object_offsets"]
        self.epoch = 0
        if method == "M5" and (geometry_points is None or geometry_normals is None):
            raise ValueError("M5 requires the verified point-cloud library")

    def __len__(self):
        return len(self.arrays["source_code"])

    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)

    def __getitem__(self, index):
        a = self.arrays
        o0, o1 = map(int, self.offsets[index:index + 2])
        lo, hi = o0 * NODES, o1 * NODES
        features = np.array(a["features"][lo:hi], dtype=np.float32, copy=True)
        centers_relative = features.reshape(o1 - o0, NODES, 140)[:, 0, 133:136].copy()
        if self.method == "M4":
            occ = np.asarray(a["occupancy"][o0:o1], dtype=np.float32)
            features[:, :128] = ((occ - np.asarray(self.stats["occupancy_mean"], np.float32)) /
                                  np.asarray(self.stats["occupancy_std"], np.float32)).reshape(-1, 128)
        elif self.method == "M5":
            features[:, :128] = 0.0
        elif self.method == "M3":
            # Match the official PIBNet feature contract: local coordinates and
            # local normals; object centers enter only through relative edges.
            features[:, 133:136] = np.tile(self.interface / 1.5, (o1 - o0, 1))

        geo_local = geo_global = None
        if self.method == "M5":
            gids = np.asarray(a["geometry_index"][o0:o1], dtype=np.int64)
            rotations = np.asarray(a["rotations"][o0:o1], dtype=np.float64)
            u, _, vt = np.linalg.svd(rotations)
            rotations = (u @ vt).astype(np.float32)
            points = np.einsum("opi,oji->opj", self.geometry_points[gids], rotations)
            normals = np.einsum("opi,oji->opj", self.geometry_normals[gids], rotations)
            nearest = np.asarray(a["nearest_indices"][o0:o1], dtype=np.int64)
            obj = np.arange(o1 - o0)[:, None, None]
            local = np.concatenate(((points[obj, nearest] - self.interface[None, :, None, :]) / 1.5,
                                    normals[obj, nearest]), axis=-1)
            geo_local = torch.from_numpy(np.ascontiguousarray(local.reshape(-1, 16, 6), dtype=np.float32))
            geo_global = torch.from_numpy(np.ascontiguousarray(
                np.concatenate((points / 1.5, normals), axis=-1), dtype=np.float32))

        sample = Scene(
            node_features=torch.from_numpy(features),
            labels=torch.from_numpy(np.array(a["labels"][lo:hi], dtype=np.float32, copy=True)),
            incident=torch.from_numpy(np.array(a["incident"][lo:hi], dtype=np.float32, copy=True)),
            q64_gt=torch.from_numpy(np.array(a["q64"][lo:hi], dtype=np.float32, copy=True)),
            z_real=torch.from_numpy(np.array(a["z_real"][lo:hi], dtype=np.float32, copy=True)),
            z_imag=torch.from_numpy(np.array(a["z_imag"][lo:hi], dtype=np.float32, copy=True)),
            sample_index=int(a["source_sample_index"][index]),
            source_code=int(a["source_code"][index]), geo_local=geo_local, geo_global=geo_global,
        )
        if self.method == "M3":
            sample.edge_features, sample.edge_index, sample.level_masks = make_pibnet_graph(
                sample.node_features, self.interface, centers_relative,
                edge_seed=(int(sample.source_code) * 1_000_003 + sample.sample_index * 10_007 + self.epoch * 97),
            )
        return sample


def make_pibnet_graph(node_features: torch.Tensor, interface_np: np.ndarray,
                      centers_relative: np.ndarray, edge_seed: int):
    objects = node_features.shape[0] // NODES
    interface = torch.from_numpy(np.ascontiguousarray(interface_np)).float()
    interface_faces = None
    topology = _get_topology(interface_np, interface_faces, objects, NODES)
    local_points = interface[None].expand(objects, -1, -1)
    # The center feature is scene-relative, which preserves all pairwise
    # displacements and avoids absolute position as a shortcut.
    centers = torch.from_numpy(np.ascontiguousarray(centers_relative)).float() * 1.5
    global_points = (local_points + centers[:, None, :]).reshape(objects * NODES, 3)
    distant = _paper_sparse_distant_edges(topology["top_nodes"], global_points, edge_seed)
    edge_index = {key: (value[0].clone(), value[1].clone())
                  for key, value in topology["graph_data"].items()}
    edge_index["distant_0"] = distant
    edge_features = {}
    for etype in EDGE_TYPES:
        src, dst = edge_index[etype]
        if etype == "distant_0":
            delta = global_points[dst] - global_points[src]
            edge_features[etype] = _edge_features_from_delta(delta, 2.0)
        else:
            geo = topology["local_edge_geometry"][etype]
            edge_features[etype] = _edge_features_from_cached_geometry(
                geo["positional"], geo["distance"], geo["direction"], 2.0
            )
    return edge_features, edge_index, {i: mask for i, mask in enumerate(topology["levels"])}


def collate_scenes(samples):
    counts = torch.tensor([item.num_nodes() for item in samples], dtype=torch.int64)
    offsets = torch.cat((torch.zeros(1, dtype=torch.int64), counts.cumsum(0)[:-1]))
    edge_types = tuple(samples[0].edge_index or {})
    edge_index = {key: torch.cat([
        torch.stack((sample.edge_index[key][0] + offset, sample.edge_index[key][1] + offset))
        for sample, offset in zip(samples, offsets)
    ], dim=1) for key in edge_types}
    edge_features = {key: torch.cat([sample.edge_features[key] for sample in samples], dim=0)
                     for key in edge_types}
    n_nodes = int(counts.sum())
    return Batch(
        node_features=torch.cat([x.node_features for x in samples]),
        labels=torch.cat([x.labels for x in samples]), incident=torch.cat([x.incident for x in samples]),
        q64_gt=torch.cat([x.q64_gt for x in samples]), z_real=torch.cat([x.z_real for x in samples]),
        z_imag=torch.cat([x.z_imag for x in samples]),
        batch_index=torch.repeat_interleave(torch.arange(len(samples)), counts),
        sample_index=torch.cat([torch.full((x.num_nodes(),), x.sample_index, dtype=torch.int64) for x in samples]),
        sample_node_counts=counts, edge_features=edge_features, edge_index=edge_index,
        level_masks={i: torch.cat([x.level_masks[i] for x in samples]) for i in range(3)} if edge_types else {},
        geo_local=(torch.cat([x.geo_local for x in samples]) if all(x.geo_local is not None for x in samples) else None),
        geo_global=(torch.cat([x.geo_global for x in samples]) if all(x.geo_global is not None for x in samples) else None),
    )


class M5CompositionalGeoStrong(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = CompositionalTransolver()
        self.local_mlp = torch.nn.Sequential(torch.nn.Linear(6, 64), torch.nn.SiLU(), torch.nn.Linear(64, 128))
        self.global_mlp = torch.nn.Sequential(torch.nn.Linear(6, 64), torch.nn.SiLU(), torch.nn.Linear(64, 128))
        self.combiner = torch.nn.Linear(384, 128)

    def forward(self, batch):
        if batch.geo_local is None or batch.geo_global is None:
            raise ValueError("missing M5 point-cloud inputs")
        local = self.local_mlp(batch.geo_local).amax(dim=1)
        enc = self.global_mlp(batch.geo_global)
        global_code = torch.cat((enc.amax(dim=1), enc.mean(dim=1)), dim=-1)
        repeated = torch.repeat_interleave(global_code, NODES, dim=0)
        if repeated.shape[0] != local.shape[0]:
            raise ValueError("M5 geometry/node alignment failed")
        features = batch.node_features.clone()
        features[:, :128] = self.combiner(torch.cat((local, repeated), dim=-1))
        old = batch.node_features
        batch.node_features = features
        try:
            return self.backbone(batch)
        finally:
            batch.node_features = old


def make_model(method: str):
    if method in ("M0", "M4", "M5", "M6"):
        model = M5CompositionalGeoStrong() if method == "M5" else CompositionalTransolver()
        if method == "M6":
            model.coupling_after = ()
        return model
    if method == "M1":
        return GroupedTransolver("transolver")
    if method == "M2":
        return GroupedTransolver("transolver-plus")
    if method == "M3":
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
    out = args.run_root / method
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
    run_manifest = {"schema": "v6s11-20k-all-methods-run-v1", "method": method,
                    "seed": args.seed, "epochs": EPOCHS, "batch_size": BATCH_SIZE,
                    "optimizer": "AdamW", "learning_rate": LR, "minimum_learning_rate": MIN_LR,
                    "weight_decay": WEIGHT_DECAY, "gradient_clip": 1.0, "lambda_q": LAMBDA_Q,
                    "checkpoint_selection": "unweighted mean of v3 and v6s11 per-scene validation scattered relative L2 means",
                    "training_scenes": {"canonical-v3": 10000, "v6s11": 10000},
                    "validation_scenes": {"canonical-v3": 1000, "v6s11": 1000},
                    "exploratory_labels": True, "v6s11_independent_bem_validated": False,
                    "cache_manifest_sha256": sha(args.cache / "manifest.json"),
                    "cache_source_manifest_sha256": cache_manifest["source_sha256"],
                    "training_statistics_sha256": sha(args.cache / "statistics.json"),
                    "training_script_sha256": code_hash,
                    "timm_0_9_2_initializer_sha256": sha(TIMM_INITIALIZER_PATH),
                    "model_source_sha256": {
                        name: sha(ROOT / name) for name in (
                            "src/cspgnet/v3_compositional_transolver.py",
                            "src/cspgnet/v3_object_transolver.py",
                            "src/cspgnet/v3_transolver_comparison.py",
                            "src/cspgnet/v3_learning.py",
                            "src/cspgnet/official_pibnet_train.py",
                            "src/cspgnet/official_pibnet_edge_backend.py",
                            "src/cspgnet/official_pibnet_adapter.py",
                            "third_party/Transolver/PDE-Solving-StandardBenchmark/model/Transolver_Irregular_Mesh.py",
                            "third_party/Transolver/PDE-Solving-StandardBenchmark/model/Physics_Attention.py",
                            "third_party/Transolver_plus/models/Transolver_plus.py")},
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
    if last_path.exists():
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


def review_tests(methods, args, stats, shared, points, normals, device):
    root = args.run_root / "review"
    root.mkdir(parents=True, exist_ok=True)
    # These are previously viewed splits; record them as cross-run review.
    reports = {}
    for method in methods:
        mdir = args.run_root / method
        run_manifest = json.loads((mdir / "run_manifest.json").read_text(encoding="utf-8"))
        best_file = mdir / "best.pt"
        best = torch.load(best_file, map_location=device, weights_only=False)
        if best.get("run_manifest") != run_manifest:
            raise RuntimeError(f"best checkpoint provenance mismatch for {method}")
        model = make_model(method).to(device)
        model.load_state_dict(best["model"], strict=True)
        reports[method] = {}
        for split in ("id_test", "peanut_ood"):
            ds = get_split(args.cache, split, method, stats, shared, points, normals)
            if method == "M3": ds.set_epoch(0)
            loader, _ = make_loader(ds, shuffle=False, seed=args.seed + 9000, workers=args.workers)
            reports[method][split] = eval_split(model, loader, device,
                torch.as_tensor(shared["phi_pinv"], device=device), stats, retain_scene_metrics=True)
            if split == "peanut_ood":
                source_path = ROOT / "work" / "peanut_source_manifest.json"
                expected = json.loads((args.cache / "manifest.json").read_text(encoding="utf-8"))[
                    "splits"]["peanut_ood"]["source_manifest_sha256"]
                if sha(source_path) != expected:
                    raise RuntimeError("Peanut subgroup manifest checksum mismatch")
                source = json.loads(source_path.read_text(encoding="utf-8"))
                groups = {int(row["index"]): row["ood_composition"] for row in source["samples"]}
                group_names = [groups[int(index)] for index in ds.arrays["source_sample_index"]]
                subgroup_results = {}
                for group in sorted(set(group_names)):
                    mask = np.asarray([name == group for name in group_names])
                    subgroup_results[group] = {"scenes": int(mask.sum())}
                    for key, items in reports[method][split]["per_scene_metrics"].items():
                        values = np.asarray(items)[mask]
                        subgroup_results[group][key] = {"mean": float(values.mean()),
                            "median": float(np.median(values)), "p95": float(np.quantile(values, .95))}
                reports[method][split]["subgroups"] = subgroup_results
            print(json.dumps({"event": "review_split_complete", "method": method, "split": split,
                              "scenes": len(ds), "scattered_rel_l2": reports[method][split]["scattered_rel_l2"]}), flush=True)
        del model
    report = {"schema": "v6s11-20k-cross-run-review-v1", "status": "complete",
              "interpretation": "previously viewed ID/Peanut splits; cross-run review, not sealed-test evidence",
              "v6s11_training_labels_exploratory": True, "methods": reports}
    write_json(root / "review_results.json", report)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache", type=Path, required=True)
    p.add_argument("--run-root", type=Path, required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--method", choices=METHODS, default=None, help="one method for smoke/resume, else run all sequentially")
    p.add_argument("--smoke", action="store_true", help="forward/backward smoke only; no epoch or run artifact")
    args = p.parse_args()
    args.cache, args.run_root = args.cache.resolve(), args.run_root.resolve()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the agreed remote training")
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda")
    cache_manifest, stats, shared, train, validations, tests, (points, normals) = dataset_bundle(args.cache, "M0")
    # M3 graph features are constructed lazily per sample; initialize model only after data checks.
    methods = [args.method] if args.method else list(METHODS)
    if args.smoke:
        for method in methods:
            model = None
            for split in ("train_v3", "train_v6s11"):
                seed_all(args.seed)
                ds = get_split(args.cache, split, method, stats, shared, points, normals)
                ds.set_epoch(0)
                items = [ds[i] for i in range(16)]
                batch = collate_scenes(items).to(device)
                if model is None:
                    model = make_model(method).to(device)
                model.train()
                model.zero_grad(set_to_none=True)
                prediction = model(batch)
                loss, _ = loss_and_scene_metrics(prediction, batch,
                    torch.as_tensor(shared["phi_pinv"], device=device), stats, LAMBDA_Q)
                loss.backward()
                grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0))
                torch.cuda.synchronize()
                print(json.dumps({"event": "smoke_passed", "method": method, "split": split,
                                  "batch_scenes": len(batch.sample_node_counts),
                                  "prediction_shape": list(prediction.shape), "loss": float(loss),
                                  "grad_norm": grad_norm, "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20}), flush=True)
                del batch, loss, prediction, ds
            del model
            torch.cuda.empty_cache()
        return

    args.run_root.mkdir(parents=True, exist_ok=True)
    write_json(args.run_root / "supervisor_status.json", {"status": "running", "methods": list(METHODS),
              "completed": [], "active": None, "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
              "v6s11_scientific_release": False})
    completed = []
    for method in methods:
        write_json(args.run_root / "supervisor_status.json", {"status": "running", "methods": list(METHODS),
                  "completed": completed, "active": method, "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                  "v6s11_scientific_release": False})
        manifest, _, _, train, validations, _, (points, normals) = dataset_bundle(args.cache, method)
        result = run_one(method, args, manifest, stats, shared, train, validations, points, normals, device)
        completed.append(result)
        write_json(args.run_root / "supervisor_status.json", {"status": "running", "methods": list(METHODS),
                  "completed": completed, "active": None, "last_completed": method,
                  "v6s11_scientific_release": False})
    if not args.method:
        test_report = review_tests(METHODS, args, stats, shared, points, normals, device)
        write_json(args.run_root / "supervisor_status.json", {"status": "complete", "methods": list(METHODS),
                  "completed": completed, "active": None, "review": "complete",
                  "v6s11_scientific_release": False,
                  "review_report_sha256": sha(args.run_root / "review" / "review_results.json")})
    print(json.dumps({"event": "all_requested_runs_complete", "methods": completed}), flush=True)


if __name__ == "__main__":
    main()
