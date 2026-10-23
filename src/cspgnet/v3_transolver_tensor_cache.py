"""Immutable, mmap-backed Transolver tensors, equivalent to the v3 graph input.

The graph's edge lists are unused by Transolver. This derived cache removes their
construction without changing forward inputs, labels, or the operator loss.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from cspgnet.official_pibnet_edge_backend import EdgeIndexGraphBatch
from cspgnet.official_pibnet_adapter import _wavelength_encoding


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(2**20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_name(path.name + ".pending")
    pending.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(pending, path)


def sample_tensors(data, response, scales, statistics):
    if not np.isclose(float(data["k"]), 2.0):
        raise ValueError("only the frozen k=2 problem is allowed")
    z = np.asarray(data["Z"], dtype=np.complex64)
    objects = len(z)
    if z.shape != (objects, 128, 64):
        raise ValueError("invalid fingerprint shape")
    response = np.asarray(response, dtype=np.complex128) / scales[None, None, :]
    features = np.empty((objects, 128, 140), dtype=np.float32)
    features[..., :64], features[..., 64:128] = response.real, response.imag
    incident = np.stack((data["u_inc"].real, data["u_inc"].imag), -1).astype(np.float32)
    features[..., 128:130] = incident
    local = torch.as_tensor(np.asarray(data["interface_points"], dtype=np.float32)) / 1.5
    centers = torch.as_tensor(np.asarray(data["centers"], dtype=np.float32))
    centers = (centers - centers.mean(0, keepdim=True)) / 1.5
    features[..., 130:133] = local.numpy()[None]
    features[..., 133:136] = centers.numpy()[:, None]
    features[..., 136:139] = np.asarray(data["incident_direction"], dtype=np.float32)
    features[..., 139] = float(_wavelength_encoding(torch.tensor([[2 * torch.pi / 2.0]], dtype=torch.float32)))
    scatter = np.asarray(data["u_total_gt"] - data["u_inc"], dtype=np.complex64)
    labels = np.stack((scatter.real, scatter.imag), -1).astype(np.float32) / float(statistics["scatter_complex_rms"])
    q = np.stack((data["q64_gt"].real, data["q64_gt"].imag), -1).astype(np.float32)
    fields = np.concatenate((labels, incident, q), -1).reshape(-1, 6)
    if not all(np.isfinite(value).all() for value in (features, fields, z)):
        raise ArithmeticError("nonfinite source tensors")
    return {"features": features.reshape(-1, 140), "fields": fields, "z": z.reshape(-1, 64)}


def build_cache(root, s_cache, output, statistics, scales):
    root, output = Path(root), Path(output)
    s_cache = None if s_cache is None else Path(s_cache)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    files = sorted(root.glob("sample_*.npz"))
    if manifest.get("status") != "complete" or len(files) != len(manifest["samples"]):
        raise RuntimeError("incomplete source split")
    s_meta = None if s_cache is None else json.loads((s_cache / "cache_manifest.json").read_text(encoding="utf-8"))
    identity = {"schema": "v3-transolver-derived-tensors-v1", "source_manifest_sha256": sha(root / "manifest.json"),
                "s_cache_manifest_sha256": None if s_cache is None else sha(s_cache / "cache_manifest.json"),
                "statistics": statistics, "scales": list(scales), "source_sha256": sha(__file__)}
    if s_meta is not None and s_meta["source_manifest_sha256"] != identity["source_manifest_sha256"]:
        raise RuntimeError("S cache has different source data")
    complete_path = output / "manifest.json"
    if complete_path.exists():
        meta = json.loads(complete_path.read_text(encoding="utf-8"))
        if meta["identity"] != identity or meta["status"] != "complete":
            raise RuntimeError("existing tensor cache provenance differs")
        return meta
    output.mkdir(parents=True, exist_ok=True)
    records = {int(item["index"]): item for item in manifest["samples"]}
    counts = [len(records[int(path.stem.split("_")[-1])]["centers"]) * 128 for path in files]
    offsets = np.r_[0, np.cumsum(counts)].astype(np.int64).tolist()
    progress_path = output / "progress.json"
    progress = json.loads(progress_path.read_text()) if progress_path.exists() else None
    if progress is not None and progress["identity"] != identity:
        raise RuntimeError("partial cache provenance differs")
    start = int(progress["completed"]) if progress else 0
    arrays = {}
    for name, width, dtype in (("features", 140, "float32"), ("fields", 6, "float32"), ("z", 64, "complex64")):
        path = output / f"{name}.npy"
        arrays[name] = np.lib.format.open_memmap(path, mode="r+" if path.exists() else "w+", dtype=dtype, shape=(offsets[-1], width))
        if arrays[name].shape != (offsets[-1], width) or arrays[name].dtype != np.dtype(dtype):
            raise RuntimeError("partial tensor array header differs")
    responses = None if s_cache is None else np.load(s_cache / "S.npy", mmap_mode="r")
    scale_vector = np.repeat(np.asarray(scales, dtype=np.float64), np.arange(1, 16, 2))
    for index in range(start, len(files)):
        with np.load(files[index]) as sample:
            keys = ("Z", "u_inc", "u_total_gt", "q64_gt", "interface_points", "centers", "incident_direction", "k")
            data = {key: sample[key] for key in keys}
            if index == 0:
                np.save(output / "phi_pinv.npy", np.asarray(sample["phi_pinv"], dtype=np.complex64))
        if responses is None:
            from cspgnet.v3_feature_trials import scattering_response
            response, _ = scattering_response(data)
        else:
            response = responses[s_meta["offsets"][index]:s_meta["offsets"][index + 1]]
        tensors = sample_tensors(data, response, scale_vector, statistics)
        for name, array in arrays.items():
            array[offsets[index]:offsets[index + 1]] = tensors[name]
        if (index + 1) % 250 == 0 or index + 1 == len(files):
            for array in arrays.values():
                array.flush()
            write_json(progress_path, {"identity": identity, "completed": index + 1})
            print(json.dumps({"cache": manifest["split"], "completed": index + 1, "scenes": len(files)}), flush=True)
    meta = {"identity": identity, "status": "complete", "scenes": len(files), "offsets": offsets,
            "sample_indices": [int(path.stem.split("_")[-1]) for path in files],
            "files_sha256": {name: sha(output / f"{name}.npy") for name in (*arrays, "phi_pinv")}}
    write_json(complete_path, meta)
    return meta


class TensorSceneDataset(Dataset):
    def __init__(self, root):
        self.root = Path(root)
        self.meta = json.loads((self.root / "manifest.json").read_text(encoding="utf-8"))
        if self.meta["status"] != "complete":
            raise RuntimeError("incomplete tensor cache")
        self.arrays = {name: np.load(self.root / f"{name}.npy", mmap_mode="r") for name in ("features", "fields", "z")}
        self.phi_pinv = torch.from_numpy(np.load(self.root / "phi_pinv.npy"))

    def __len__(self):
        return self.meta["scenes"]

    def __getitem__(self, index):
        lo, hi = self.meta["offsets"][index:index + 2]
        return {name: value[lo:hi] for name, value in self.arrays.items()} | {"index": self.meta["sample_indices"][index]}


def collate_tensors(samples):
    features, fields, z = (torch.from_numpy(np.concatenate([item[name] for item in samples])) for name in ("features", "fields", "z"))
    counts = torch.tensor([len(item["features"]) for item in samples], dtype=torch.int64)
    return EdgeIndexGraphBatch(node_features=features, labels=fields[:, :2], incident=fields[:, 2:4],
        q64_gt=fields[:, 4:6], z_real=z.real.contiguous(), z_imag=z.imag.contiguous(),
        batch_index=torch.repeat_interleave(torch.arange(len(samples)), counts),
        sample_index=torch.repeat_interleave(torch.tensor([item["index"] for item in samples]), counts),
        sample_node_counts=counts, edge_features={}, edge_index={}, level_masks={})
