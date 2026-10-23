"""Train the official PIBNet backbone on formal canonical-surface data.

The processor architecture matches the paper configuration.  The data adapter
changes only the input schema and graph construction required by CSP-GNet.
Losses and metrics are normalized per scene, so variable object counts do not
silently reweight the experiment.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import random
import subprocess
import sys
import types
from collections import Counter
from pathlib import Path

PRIMARY_SITE = r"C:\Users\27159\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\Lib\site-packages"
OFFICIAL_SRC = r"C:\Users\27159\Documents\Codex\2026-09-20\transformer\work\pibnet\src"
OFFICIAL_PIBNET_SRC = r"C:\Users\27159\Documents\Codex\2026-09-20\transformer\work\pibnet\src\pibnet"
OMP_DIR = r"C:\Users\27159\Documents\Codex\2026-09-20\transformer\work\openmp_runtime_20260920\package\runtimes\win-x64\native"

os.environ.setdefault("DGLBACKEND", "pytorch")
os.environ.setdefault("DGL_SKIP_GRAPHBOLT", "1")
if Path(OMP_DIR).exists():
    os.add_dll_directory(OMP_DIR)
sys.path.append(PRIMARY_SITE)
sys.path.insert(0, OFFICIAL_SRC)
sys.path.insert(0, OFFICIAL_PIBNET_SRC)

import dgl
import numpy as np
import torch
from torch.utils.data import DataLoader

try:
    _torchvision_compat = torch.library.Library("torchvision", "DEF")
    _torchvision_compat.define("nms(Tensor boxes, Tensor scores, float iou_threshold) -> Tensor")
except (RuntimeError, TypeError):
    _torchvision_compat = None


def _install_physicsnemo_activation_package_shim() -> None:
    """Expose PhysicsNeMo's exact activation factory without eager layer imports.

    PhysicsNeMo's layers package eagerly imports unrelated optional layers and
    their TorchScript kernels. PIBNet only needs ``get_activation`` here, so
    expose that unchanged factory while retaining the package path for lazy
    imports such as ``fused_silu``.
    """
    if "physicsnemo.models.layers" in sys.modules:
        return
    import physicsnemo
    import physicsnemo.models

    layer_path = Path(physicsnemo.__file__).parent / "models" / "layers"
    activation_path = layer_path / "activations.py"
    spec = importlib.util.spec_from_file_location(
        "pibnet_physicsnemo_activations", activation_path
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load PhysicsNeMo activation factory at {activation_path}")
    activation_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(activation_module)
    layer_package = types.ModuleType("physicsnemo.models.layers")
    layer_package.__file__ = str(layer_path / "__init__.py")
    layer_package.__path__ = [str(layer_path)]
    layer_package.__package__ = "physicsnemo.models.layers"
    layer_package.get_activation = activation_module.get_activation
    sys.modules["physicsnemo.models.layers"] = layer_package
    physicsnemo.models.layers = layer_package


_install_physicsnemo_activation_package_shim()

from gnn.pibnet import PIBNet
from cspgnet.official_pibnet_adapter import (
    EDGE_DIM,
    EDGE_TYPES,
    NODE_DIM,
    NODES_PER_OBJECT,
    OfficialPIBNetCanonicalDataset,
    collate_graphs,
    fit_complex_normalization,
)
from cspgnet.official_pibnet_edge_backend import (
    EdgeIndexGraphBatch,
    PIBNetEdgeIndex,
    collate_edge_index_graphs,
)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_model(device: torch.device, node_dim: int = NODE_DIM) -> PIBNet:
    return PIBNet(
        input_dim_nodes=node_dim,
        input_dim_edges={etype: EDGE_DIM for etype in EDGE_TYPES},
        output_dim=2,
        num_levels=3,
        first_top_processor_size=3,
        last_top_processor_size=3,
        bottom_processor_size=6,
        hidden_dim_scaling=2,
        distant_edge_sample_period=6,
        hidden_dim_processor=64,
        hidden_dim_node_encoder=64,
        hidden_dim_edge_encoder=64,
        hidden_dim_node_decoder=64,
        mlp_activation_fn="silu",
    ).to(device)


def _complex(two_channel: torch.Tensor) -> torch.Tensor:
    return torch.complex(two_channel[..., 0], two_channel[..., 1])


def _scene_reduce(values: torch.Tensor, batch_index: torch.Tensor, sample_count: int) -> torch.Tensor:
    result = torch.zeros(sample_count, dtype=values.dtype, device=values.device)
    result.index_add_(0, batch_index, values)
    return result


def _relative_squared_per_scene(
    prediction: torch.Tensor,
    target: torch.Tensor,
    batch_index: torch.Tensor,
    sample_count: int,
) -> torch.Tensor:
    numerator = _scene_reduce((prediction - target).abs().square(), batch_index, sample_count)
    denominator = _scene_reduce(target.abs().square(), batch_index, sample_count).clamp_min(1.0e-12)
    return numerator / denominator


def projected_flux(
    prediction: torch.Tensor,
    z_real: torch.Tensor,
    z_imag: torch.Tensor,
    phi_pinv: torch.Tensor,
) -> torch.Tensor:
    prediction_complex = _complex(prediction)
    if prediction_complex.numel() % NODES_PER_OBJECT:
        raise ValueError("node count is not divisible by the canonical surface size")
    object_values = prediction_complex.reshape(-1, NODES_PER_OBJECT)
    probes = z_real.shape[-1]
    z_complex = torch.complex(z_real, z_imag).reshape(-1, NODES_PER_OBJECT, probes)
    coefficients = torch.einsum("mn,bn->bm", phi_pinv, object_values)
    return torch.einsum("bnm,bm->bn", z_complex, coefficients).reshape(-1)


def _batch_payload(graph, backend: str):
    if backend == "edge":
        assert isinstance(graph, EdgeIndexGraphBatch)
        return {
            "target": graph.labels,
            "incident": graph.incident,
            "q_target": graph.q64_gt,
            "z_real": graph.z_real,
            "z_imag": graph.z_imag,
            "batch_index": graph.batch_index,
            "sample_count": int(graph.sample_node_counts.numel()),
        }
    node_counts = graph.batch_num_nodes("obs").to(graph.device)
    batch_index = torch.repeat_interleave(
        torch.arange(len(node_counts), device=graph.device), node_counts
    )
    return {
        "target": graph.ndata["labels"],
        "incident": graph.ndata["incident"],
        "q_target": graph.ndata["q64_gt"],
        "z_real": graph.ndata["z_real"],
        "z_imag": graph.ndata["z_imag"],
        "batch_index": batch_index,
        "sample_count": int(len(node_counts)),
    }


def loss_and_sample_metrics(
    prediction: torch.Tensor,
    payload: dict,
    phi_pinv: torch.Tensor,
    lambda_q: float,
    output_scale: float = 1.0,
):
    target = payload["target"]
    incident = payload["incident"]
    batch_index = payload["batch_index"]
    sample_count = payload["sample_count"]
    prediction_complex = _complex(prediction) * output_scale
    target_complex = _complex(target) * output_scale
    incident_complex = _complex(incident)
    q_prediction = projected_flux(
        prediction * output_scale,
        payload["z_real"],
        payload["z_imag"],
        phi_pinv,
    )
    q_target = _complex(payload["q_target"])

    u_squared = _relative_squared_per_scene(
        prediction_complex, target_complex, batch_index, sample_count
    )
    q_squared = _relative_squared_per_scene(q_prediction, q_target, batch_index, sample_count)
    loss = u_squared.mean() + lambda_q * q_squared.mean()

    scattered_prediction = prediction_complex - incident_complex
    scattered_target = target_complex - incident_complex
    scattered_squared = _relative_squared_per_scene(
        scattered_prediction, scattered_target, batch_index, sample_count
    )
    amplitude_squared = _relative_squared_per_scene(
        prediction_complex.abs(), target_complex.abs(), batch_index, sample_count
    )

    target_amplitude = target_complex.abs()
    maximum = torch.zeros(sample_count, dtype=target_amplitude.dtype, device=target_amplitude.device)
    maximum.scatter_reduce_(0, batch_index, target_amplitude, reduce="amax", include_self=False)
    phase_mask = target_amplitude > (0.01 * maximum[batch_index]).clamp_min(1.0e-8)
    phase_error = torch.angle(prediction_complex * target_complex.conj()).abs()
    phase_sum = _scene_reduce(phase_error * phase_mask, batch_index, sample_count)
    phase_count = _scene_reduce(phase_mask.to(phase_error.dtype), batch_index, sample_count)
    phase_mae = phase_sum / phase_count.clamp_min(1.0)
    return loss, {
        "total_rel_l2": u_squared.sqrt(),
        "scattered_rel_l2": scattered_squared.sqrt(),
        "q64_rel_l2": q_squared.sqrt(),
        "amplitude_rel_l2": amplitude_squared.sqrt(),
        "phase_mae_deg": phase_mae * (180.0 / torch.pi),
    }


def _summarize(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "p95": float(np.quantile(array, 0.95)),
    }


def run_epoch(
    model,
    loader,
    device,
    phi_pinv: torch.Tensor,
    lambda_q: float,
    optimizer=None,
    backend="dgl",
    output_scale: float = 1.0,
):
    training = optimizer is not None
    model.train(training)
    loss_sum = 0.0
    sample_count = 0
    metric_values = {
        "total_rel_l2": [],
        "scattered_rel_l2": [],
        "q64_rel_l2": [],
        "amplitude_rel_l2": [],
        "phase_mae_deg": [],
    }
    for graph in loader:
        graph = graph.to(device)
        payload = _batch_payload(graph, backend)
        with torch.set_grad_enabled(training):
            if backend == "edge":
                prediction = model(graph)
            else:
                edge_features = {
                    etype: graph.edges[etype].data["edge_features"] for etype in EDGE_TYPES
                }
                prediction = model(graph.ndata["node_features"], edge_features, graph)
            loss, sample_metrics = loss_and_sample_metrics(
                prediction, payload, phi_pinv, lambda_q, output_scale
            )
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
        batch_samples = payload["sample_count"]
        loss_sum += float(loss.detach().cpu()) * batch_samples
        sample_count += batch_samples
        for name, values in sample_metrics.items():
            metric_values[name].extend(values.detach().cpu().tolist())

    result = {"loss": loss_sum / max(sample_count, 1), "samples": sample_count}
    result.update({name: _summarize(values) for name, values in metric_values.items()})
    return result


def incident_only_metrics(loader, device, phi_pinv, backend, output_scale: float = 1.0):
    collected = {name: [] for name in (
        "total_rel_l2", "scattered_rel_l2", "q64_rel_l2", "amplitude_rel_l2", "phase_mae_deg"
    )}
    for graph in loader:
        graph = graph.to(device)
        payload = _batch_payload(graph, backend)
        _, metrics = loss_and_sample_metrics(
            payload["incident"] / output_scale,
            payload,
            phi_pinv,
            lambda_q=0.0,
            output_scale=output_scale,
        )
        for name, values in metrics.items():
            collected[name].extend(values.detach().cpu().tolist())
    return {name: _summarize(values) for name, values in collected.items()}


def _file_sha256(path: Path) -> str | None:
    if not path.exists():
        return None
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def _atomic_write_json(path: Path, value) -> None:
    pending = path.with_name(path.name + ".pending")
    pending.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(pending, path)


def _atomic_torch_save(path: Path, value) -> None:
    pending = path.with_name(path.name + ".pending")
    torch.save(value, pending)
    os.replace(pending, path)


def _check_resume_provenance(previous: dict, current: dict) -> None:
    previous_args = dict(previous["args"])
    current_args = dict(current["args"])
    previous_args.pop("resume", None)
    current_args.pop("resume", None)
    if previous_args != current_args:
        raise RuntimeError("resume arguments differ from the original run")
    for key in (
        "datasets",
        "node_feature_dim",
        "parameters",
        "torch",
        "cuda",
        "normalization",
        "edge_input_parity_sha256",
        "normalized_edge_parity_sha256",
    ):
        if previous[key] != current[key]:
            raise RuntimeError(f"resume provenance differs in {key}")
    if (previous.get("input_pipeline") == current.get("input_pipeline")
            and previous.get("edge_input_parity_sha256") != current.get("edge_input_parity_sha256")):
        raise RuntimeError("input-pipeline parity evidence changed on resume")


def _git_provenance() -> dict:
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
        dirty = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"], text=True, stderr=subprocess.DEVNULL
            ).strip()
        )
        return {"commit": commit, "dirty": dirty}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}


def _dataset_info(dataset: OfficialPIBNetCanonicalDataset) -> dict:
    manifest_path = dataset.root / "manifest.json"
    return {
        "path": str(dataset.root.resolve()),
        "samples": len(dataset),
        "schema_version": dataset.schema_version,
        "manifest_sha256": _file_sha256(manifest_path),
    }


def _validate_formal_splits(datasets: dict[str, OfficialPIBNetCanonicalDataset]) -> None:
    """Reject incomplete or overlapping validated splits before allocating a model."""
    expected_split = {"train": "train", "val": "val", "test": "test", "count_ood": "count_ood_n8"}
    hashes: dict[str, str] = {}
    library_geometries: dict[tuple, str] = {}
    gate_hashes = set()
    schemas = set()
    for key, dataset in datasets.items():
        if dataset is None:
            continue
        manifest = dataset.manifest
        schema = manifest.get("schema_version")
        schemas.add(schema)
        evidence_key = (
            "validated_tmatrix_evidence" if schema == "cspgnet-neumann-tmatrix-v2"
            else "formal_evidence"
        )
        if manifest.get("status") != "complete" or not manifest.get(evidence_key):
            raise RuntimeError(f"{key} is not a complete, gate-approved dataset")
        if schema == "cspgnet-neumann-tmatrix-v2":
            if manifest.get("label_solver") != "T-matrix Foldy-Lax (not per-scene BEM)":
                raise RuntimeError(f"{key} has an unexpected T-matrix label solver")
            library_path = Path(manifest["library_manifest"])
            if _file_sha256(library_path) != manifest.get("library_sha256"):
                raise RuntimeError(f"{key} shape-library manifest checksum differs")
            library = json.loads(library_path.read_text(encoding="utf-8"))
            if library.get("status") != "complete":
                raise RuntimeError(f"{key} shape library is not complete")
            for entry in library["entries"]:
                if entry.get("status") != "complete":
                    raise RuntimeError(f"{key} shape library contains an unverified operator")
                geometry = (entry["shape"], *entry["size"])
                if geometry in library_geometries:
                    raise RuntimeError(
                        f"base geometry overlaps between {library_geometries[geometry]} and {key}"
                    )
                library_geometries[geometry] = key
        if manifest.get("split") != expected_split[key]:
            raise RuntimeError(
                f"{key} points at split={manifest.get('split')!r}; expected {expected_split[key]!r}"
            )
        gate = manifest.get("numerical_gate") or {}
        gate_hashes.add(gate.get("sha256"))
        sample_records = manifest.get("samples", [])
        if len(sample_records) != int(manifest.get("num_samples", -1)):
            raise RuntimeError(f"{key} manifest does not contain its full declared sample set")
        expected_count = {
            "train": 10000,
            "val": 1000,
            "test": 1000,
            "count_ood": 1000,
        }[key]
        if len(dataset) != expected_count:
            raise RuntimeError(f"{key} has {len(dataset)} samples; expected {expected_count}")
        counts = Counter()
        for sample in sample_records:
            scene = {
                "k": sample["k"],
                "incident_direction": sample["incident_direction"],
                "centers": sample["centers"],
                "objects": sample["objects"],
            }
            digest = hashlib.sha256(
                json.dumps(scene, sort_keys=True, separators=(",", ":")).encode("utf-8")
            ).hexdigest()
            if digest in hashes:
                raise RuntimeError(f"scene parameters overlap between {hashes[digest]} and {key}")
            hashes[digest] = key
            count = len(sample.get("objects", []))
            counts[count] += 1
            if key == "count_ood" and count != 8:
                raise RuntimeError("count_ood split contains a sample other than N=8")
        if key != "count_ood":
            start, end = manifest.get("count_range", [2, 4])
            quotient, remainder = divmod(expected_count, end - start + 1)
            expected_counts = {
                count: quotient + int(count - start < remainder)
                for count in range(start, end + 1)
            }
            if dict(counts) != expected_counts:
                raise RuntimeError(f"{key} object-count distribution is not balanced: {counts}")
    if len(gate_hashes) != 1 or None in gate_hashes:
        raise RuntimeError("formal data splits do not reference the same numerical-gate report")
    if len(schemas) != 1:
        raise RuntimeError("validated data splits use different schemas")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", required=True)
    parser.add_argument("--val", required=True)
    parser.add_argument("--test", required=True)
    parser.add_argument("--count-ood", default="")
    parser.add_argument("--output", required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=1)
    parser.add_argument("--limit-train", type=int, default=0)
    parser.add_argument("--limit-eval", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lambda-q", type=float, choices=(0.0, 0.1, 1.0), default=0.1)
    parser.add_argument("--zero-z-input", "--zero-physics", dest="zero_physics", action="store_true")
    parser.add_argument(
        "--complex-normalization",
        action="store_true",
        help="fit train-only phase-preserving RMS scales for Z and total-field labels",
    )
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--resume", default="")
    parser.add_argument("--backend", choices=("dgl", "edge"), default="edge")
    parser.add_argument("--allow-incomplete", action="store_true")
    return parser.parse_args()


def main(args):
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    torch.set_float32_matmul_precision("high")
    print(
        f"initializing device={device} train={args.train} normalization={args.complex_normalization}",
        flush=True,
    )
    require_complete = not args.allow_incomplete
    train_set = OfficialPIBNetCanonicalDataset(
        args.train, zero_physics=args.zero_physics, require_complete=require_complete,
        edge_direct=args.backend == "edge",
    )
    val_set = OfficialPIBNetCanonicalDataset(
        args.val, zero_physics=args.zero_physics, require_complete=require_complete,
        edge_direct=args.backend == "edge",
    )
    test_set = OfficialPIBNetCanonicalDataset(
        args.test, zero_physics=args.zero_physics, require_complete=require_complete,
        edge_direct=args.backend == "edge",
    )
    count_set = (
        OfficialPIBNetCanonicalDataset(
            args.count_ood, zero_physics=args.zero_physics, require_complete=require_complete,
            edge_direct=args.backend == "edge",
        )
        if args.count_ood
        else None
    )
    if require_complete:
        _validate_formal_splits(
            {"train": train_set, "val": val_set, "test": test_set, "count_ood": count_set}
        )
    if args.limit_train > 0:
        train_set.files = train_set.files[: args.limit_train]
    if args.limit_eval > 0:
        val_set.files = val_set.files[: args.limit_eval]
        test_set.files = test_set.files[: args.limit_eval]
        if count_set is not None:
            count_set.files = count_set.files[: args.limit_eval]

    datasets = [train_set, val_set, test_set] + ([count_set] if count_set is not None else [])
    if args.complex_normalization:
        print(
            f"fitting normalization from {len(train_set.files)} training samples only",
            flush=True,
        )
    normalization = fit_complex_normalization(train_set.files) if args.complex_normalization else None
    for dataset in datasets:
        dataset.normalization = normalization
    output_scale = (
        float(normalization["total_field_rms_shared_real_imag"])
        if normalization is not None
        else 1.0
    )
    node_dim = train_set.node_feature_dim
    for dataset in datasets:
        if dataset.node_feature_dim != node_dim:
            raise ValueError("all datasets must have the same probe count and node input width")
        if not torch.allclose(dataset.phi_pinv, train_set.phi_pinv, atol=1.0e-5, rtol=1.0e-5):
            raise ValueError("all datasets must use the same canonical probe basis")

    collate_fn = collate_edge_index_graphs if args.backend == "edge" else collate_graphs
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
        num_workers=0,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(val_set, batch_size=args.eval_batch_size, shuffle=False, num_workers=0, collate_fn=collate_fn)
    test_loader = DataLoader(test_set, batch_size=args.eval_batch_size, shuffle=False, num_workers=0, collate_fn=collate_fn)
    count_loader = (
        DataLoader(count_set, batch_size=1, shuffle=False, num_workers=0, collate_fn=collate_fn)
        if count_set is not None
        else None
    )

    model = make_model(device, node_dim=node_dim)
    if args.backend == "edge":
        model = PIBNetEdgeIndex(model).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-4, weight_decay=1.0e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1.0e-7
    )
    phi_pinv = train_set.phi_pinv.to(device)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    history = []
    best = float("inf")
    start_epoch = 1

    run_manifest = {
        "args": vars(args),
        "git": _git_provenance(),
        "device": str(device),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "datasets": {
            "train": _dataset_info(train_set),
            "val": _dataset_info(val_set),
            "test": _dataset_info(test_set),
            "count_ood": _dataset_info(count_set) if count_set is not None else None,
        },
        "node_feature_dim": node_dim,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "normalization": normalization,
        "physicsnemo_activation_factory": "installed activations.py get_activation (eager package import bypassed)",
    }
    if args.backend == "edge":
        parity_path = Path(__file__).resolve().parents[2] / "outputs" / "tmatrix_direct_edge_parity.json"
        parity = json.loads(parity_path.read_text(encoding="utf-8"))
        if not parity.get("passed") or parity.get("max_abs_difference") != 0.0:
            raise RuntimeError("direct edge-input tensor parity has not passed")
        run_manifest["input_pipeline"] = "direct_edge_index_v1"
        run_manifest["edge_input_parity_sha256"] = _file_sha256(parity_path)
        if args.complex_normalization:
            normalized_parity_path = (
                Path(__file__).resolve().parents[2]
                / "outputs"
                / "tmatrix_normalized_edge_parity.json"
            )
            if not normalized_parity_path.exists():
                raise RuntimeError("normalized DGL/edge parity evidence is missing")
            normalized_parity = json.loads(
                normalized_parity_path.read_text(encoding="utf-8")
            )
            if (
                normalized_parity.get("schema") != "normalized_edge_parity_v1"
                or not normalized_parity.get("passed")
                or normalized_parity.get("max_abs_difference") != 0.0
            ):
                raise RuntimeError("normalized DGL/edge input parity has not passed")
            run_manifest["normalized_edge_parity_sha256"] = _file_sha256(
                normalized_parity_path
            )
        else:
            run_manifest["normalized_edge_parity_sha256"] = None
    else:
        run_manifest["input_pipeline"] = "dgl_reference"
        run_manifest["edge_input_parity_sha256"] = None
        run_manifest["normalized_edge_parity_sha256"] = None
    manifest_path = output / "run_manifest.json"
    if normalization is not None:
        _atomic_write_json(output / "normalization.json", normalization)
    if args.resume:
        if not manifest_path.exists():
            raise RuntimeError("resume requested but original run manifest is absent")
        previous_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        _check_resume_provenance(previous_manifest, run_manifest)
        if previous_manifest.get("input_pipeline") != run_manifest["input_pipeline"]:
            _atomic_write_json(output / "input_pipeline_transition.json", {
                "from": previous_manifest.get("input_pipeline", "dgl_intermediate_edge_index"),
                "to": run_manifest["input_pipeline"],
                "direct_vs_dgl_tensor_parity_sha256": run_manifest["edge_input_parity_sha256"],
                "note": "The previous checkpoint's input tensors and the new direct tensors match exactly.",
            })
    else:
        _atomic_write_json(manifest_path, run_manifest)

    def model_state():
        return model.core.state_dict() if args.backend == "edge" else model.state_dict()

    def load_model_state(state):
        if args.backend == "edge":
            model.core.load_state_dict(state)
        else:
            model.load_state_dict(state)

    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        _check_resume_provenance(checkpoint["run_manifest"], run_manifest)
        load_model_state(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best = float(checkpoint.get("best_val_total_rel_l2", best))
        history_path = output / "history.json"
        if history_path.exists():
            history = json.loads(history_path.read_text(encoding="utf-8"))
        if "rng" in checkpoint:
            rng = checkpoint["rng"]
            random.setstate(rng["python"])
            np.random.set_state(rng["numpy"])
            torch.set_rng_state(rng["torch_cpu"].cpu())
            if device.type == "cuda":
                torch.cuda.set_rng_state_all([state.cpu() for state in rng["torch_cuda"]])
            generator.set_state(rng["train_loader"].cpu())
        else:
            print("warning: legacy checkpoint lacks exact RNG state", flush=True)

    print(
        f"device={device} backend={args.backend} train={len(train_set)} val={len(val_set)} "
        f"test={len(test_set)} lambda_q={args.lambda_q} zero_physics={args.zero_physics} "
        f"params={sum(parameter.numel() for parameter in model.parameters())}",
        flush=True,
    )
    for epoch in range(start_epoch, args.epochs + 1):
        train_set.set_epoch(epoch)
        train_metrics = run_epoch(
            model, train_loader, device, phi_pinv, args.lambda_q, optimizer, args.backend,
            output_scale,
        )
        val_set.set_epoch(0)
        val_metrics = run_epoch(
            model, val_loader, device, phi_pinv, args.lambda_q,
            backend=args.backend, output_scale=output_scale,
        )
        scheduler.step()
        row = {
            "epoch": epoch,
            "lr": scheduler.get_last_lr()[0],
            "train": train_metrics,
            "val": val_metrics,
        }
        history.append(row)
        val_score = val_metrics["total_rel_l2"]["mean"]
        checkpoint = {
            "epoch": epoch,
            "model": model_state(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "args": vars(args),
            "run_manifest": run_manifest,
            "train": train_metrics,
            "val": val_metrics,
            "best_val_total_rel_l2": min(best, val_score),
            "rng": {
                "python": random.getstate(),
                "numpy": np.random.get_state(),
                "torch_cpu": torch.get_rng_state(),
                "torch_cuda": torch.cuda.get_rng_state_all() if device.type == "cuda" else [],
                "train_loader": generator.get_state(),
            },
        }
        _atomic_torch_save(output / "last.pt", checkpoint)
        if val_score < best:
            best = val_score
            _atomic_torch_save(output / "best.pt", checkpoint)
        _atomic_write_json(output / "history.json", history)
        print(json.dumps(row, ensure_ascii=False), flush=True)

    checkpoint = torch.load(output / "best.pt", map_location=device, weights_only=False)
    load_model_state(checkpoint["model"])
    test_metrics = run_epoch(
        model, test_loader, device, phi_pinv, args.lambda_q,
        backend=args.backend, output_scale=output_scale,
    )
    count_metrics = (
        run_epoch(
            model, count_loader, device, phi_pinv, args.lambda_q,
            backend=args.backend, output_scale=output_scale,
        )
        if count_loader is not None
        else None
    )
    baselines = {
        "test_incident_only": incident_only_metrics(
            test_loader, device, phi_pinv, args.backend, output_scale
        ),
        "count_ood_incident_only": (
            incident_only_metrics(count_loader, device, phi_pinv, args.backend, output_scale)
            if count_loader is not None
            else None
        ),
    }
    final = {
        "best_epoch": checkpoint["epoch"],
        "best_val": checkpoint["val"],
        "test": test_metrics,
        "count_ood_n8": count_metrics,
        "baselines": baselines,
    }
    _atomic_write_json(output / "final_metrics.json", final)
    print(json.dumps(final, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main(parse_args())
