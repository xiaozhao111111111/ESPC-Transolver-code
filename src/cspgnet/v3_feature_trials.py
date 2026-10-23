"""Isolated development-only tests of two representations of stored v3 Z."""

from __future__ import annotations

from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch
from torch import nn

from cspgnet.data_generation import formal_neumann_3d as formal
from cspgnet.data_generation import tmatrix_neumann_3d as tmatrix
from cspgnet.data_generation.reduced_z64_solver import modal_dtn_from_stored_z
from cspgnet.official_pibnet_edge_backend import PIBNetEdgeIndex
from cspgnet.v3_inputs import V3CanonicalDataset


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def scattering_response(sample: dict) -> tuple[np.ndarray, dict]:
    """Use saved complex64 Z only: S=Phi diag(h_l^(1)(kR)) T_out."""
    b64, projection = modal_dtn_from_stored_z(sample)
    k = float(sample["k"])
    radius = projection["radius"]
    if not np.isclose(k, 2.0) or not np.isclose(radius, 1.5):
        raise ValueError("feature trial requires the frozen k=2, R=1.5 problem")
    phi = formal.spherical_harmonic_matrix(
        np.asarray(sample["interface_points"], dtype=np.float64), formal.spherical_modes(7)
    )
    h, _ = tmatrix.radial_values(k, radius, 7, outgoing=True)
    waves = phi * h[None, :]
    independent_waves = tmatrix.spherical_wave_values(
        np.asarray(sample["interface_points"], dtype=np.float64), k, 7, outgoing=True,
    )
    output = np.empty((len(b64), 128, 64), dtype=np.complex64)
    maximum_residual = 0.0
    maximum_identity_error = 0.0
    vector = np.random.default_rng(20260927).standard_normal(64) + 1j * np.random.default_rng(20260928).standard_normal(64)
    for index, modal in enumerate(b64):
        transition, diagnostics = tmatrix.transition_matrix(
            modal, k=k, canonical_radius=radius, lmax=7,
        )
        maximum_residual = max(maximum_residual, diagnostics["conversion_residual"])
        response = waves @ transition
        independent = independent_waves @ (transition @ vector)
        error = np.linalg.norm(response @ vector - independent) / max(np.linalg.norm(independent), 1e-15)
        maximum_identity_error = max(maximum_identity_error, float(error))
        # Stored interface coordinates have float32-scale radial jitter; their
        # pointwise wave values therefore differ slightly from the exact-R H.
        if not np.isfinite(response).all() or error > 1e-6 or diagnostics["conversion_residual"] > 1e-10:
            raise ArithmeticError(
                f"S conversion failed: identity={error:.3e}, "
                f"conversion={diagnostics['conversion_residual']:.3e}, "
                f"finite={bool(np.isfinite(response).all())}"
            )
        output[index] = response.astype(np.complex64)
    return output, {"max_conversion_residual": maximum_residual, "max_identity_error": maximum_identity_error}


def build_response_cache(root: Path, cache_dir: Path) -> dict:
    """Cache derived S without editing FEM samples; reject mismatched sources."""
    root, cache_dir = Path(root), Path(cache_dir)
    files = sorted(root.glob("sample_*.npz"))
    manifest_hash = sha(root / "manifest.json")
    array_path, meta_path = cache_dir / "S.npy", cache_dir / "cache_manifest.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if (meta.get("schema") != "v3-derived-S-cache-v1" or
            meta.get("source_manifest_sha256") != manifest_hash or
            meta.get("files") != len(files) or
            not array_path.exists()):
            raise RuntimeError("existing S cache does not match this data")
        return meta
    if array_path.exists():
        raise RuntimeError("incomplete S cache exists; inspect before rebuilding")
    if not files:
        raise FileNotFoundError(root)
    cache_dir.mkdir(parents=True, exist_ok=True)
    counts = []
    for path in files:
        with np.load(path) as sample:
            counts.append(int(sample["Z"].shape[0]))
    offsets = np.concatenate(([0], np.cumsum(counts))).astype(np.int64)
    pending = cache_dir / "S.pending.npy"
    if pending.exists():
        raise RuntimeError("incomplete pending S cache exists; inspect before rebuilding")
    output = np.lib.format.open_memmap(pending, mode="w+", dtype=np.complex64,
                                       shape=(int(offsets[-1]), 128, 64))
    powers = np.zeros(8, dtype=np.float64)
    counts_by_degree = np.zeros(8, dtype=np.int64)
    residual = identity_error = 0.0
    for index, path in enumerate(files):
        with np.load(path) as opened:
            sample = {key: opened[key] for key in opened.files}
        s, diagnostics = scattering_response(sample)
        output[offsets[index]:offsets[index+1]] = s
        residual = max(residual, diagnostics["max_conversion_residual"])
        identity_error = max(identity_error, diagnostics["max_identity_error"])
        if index % 500 == 0:
            output.flush()
            print(json.dumps({"cache": str(cache_dir), "done": index + 1, "total": len(files)}), flush=True)
        for degree in range(8):
            block = s[..., degree * degree:(degree + 1) ** 2].astype(np.complex128)
            powers[degree] += float(np.sum(np.abs(block) ** 2))
            counts_by_degree[degree] += block.size
    output.flush()
    del output
    os.replace(pending, array_path)
    meta = {
        "schema": "v3-derived-S-cache-v1", "source_manifest_sha256": manifest_hash,
        "files": len(files), "objects": int(offsets[-1]), "offsets": offsets.tolist(),
        "degree_complex_rms": np.sqrt(np.maximum(powers / counts_by_degree, 1e-24)).tolist(),
        "max_conversion_residual": residual, "max_identity_error": identity_error,
        "dtype": "complex64", "shape": [int(offsets[-1]), 128, 64],
    }
    pending_meta = meta_path.with_suffix(".pending.json")
    pending_meta.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    os.replace(pending_meta, meta_path)
    return meta


class SResponseDataset(V3CanonicalDataset):
    """Exactly the v3 graph/target, replacing only 128 physical input channels."""

    def __init__(self, root: Path, statistics: dict, cache_dir: Path, train_scales):
        super().__init__(root, statistics, "z", "scattered")
        self.meta = json.loads((Path(cache_dir) / "cache_manifest.json").read_text(encoding="utf-8"))
        if self.meta["source_manifest_sha256"] != sha(Path(root) / "manifest.json"):
            raise RuntimeError("S cache source does not match dataset")
        self.responses = np.load(Path(cache_dir) / "S.npy", mmap_mode="r")
        self.offsets = self.meta["offsets"]
        self.scales = np.repeat(np.asarray(train_scales, dtype=np.float64), np.arange(1, 16, 2))
        if self.responses.shape != tuple(self.meta["shape"]):
            raise RuntimeError("S cache dimensions do not match manifest")

    def __getitem__(self, index):
        graph = super().__getitem__(index)
        s = np.asarray(self.responses[self.offsets[index]:self.offsets[index + 1]], dtype=np.complex128)
        s /= self.scales[None, None, :]
        first = np.concatenate((s.real, s.imag), axis=-1).astype(np.float32)
        graph.node_features[:, :128] = torch.from_numpy(first.reshape(-1, 128))
        return graph


class ZContextEncoder(nn.Module):
    """Node-local and object-global learned codes; unchanged PIBNet backbone."""

    def __init__(self, core):
        super().__init__()
        self.local_mlp = nn.Sequential(nn.Linear(128, 32), nn.SiLU(), nn.Linear(32, 128))
        self.global_mlp = nn.Sequential(nn.Linear(128, 32), nn.SiLU(), nn.Linear(32, 128))
        self.combiner = nn.Linear(384, 128)
        self.backbone = PIBNetEdgeIndex(core)

    def forward(self, graph):
        original = graph.node_features[:, :128]
        if len(original) % 128:
            raise ValueError("node sequence does not consist of 128-node canonical spheres")
        local = self.local_mlp(original)
        global_nodes = self.global_mlp(original).reshape(-1, 128, 128)
        global_code = torch.cat((global_nodes.amax(dim=1), global_nodes.mean(dim=1)), dim=-1)
        global_per_node = torch.repeat_interleave(global_code, 128, dim=0)
        code = self.combiner(torch.cat((local, global_per_node), dim=-1))
        features = graph.node_features.clone()
        features[:, :128] = code
        return self.backbone(replace(graph, node_features=features))
