"""Same-family, radius-matched Z swaps without altering any other input."""

from __future__ import annotations

import hashlib
import json
from bisect import bisect_left, bisect_right

import numpy as np
import torch


def bounding_radius(record: dict) -> float:
    shape = record["shape"]
    values = np.asarray(record["size"], dtype=np.float64)
    if shape == "sphere":
        return float(values[0])
    if shape == "ellipsoid":
        return float(values.max())
    if shape == "box":
        return float(np.linalg.norm(values))
    if shape == "peanut":
        return float(values[2] / 2.0 + max(values[0], values[1]))
    if shape == "torus":
        return float(values[0] + values[1])
    raise ValueError(f"unknown shape {shape}")


class MatchedZSwapDataset(torch.utils.data.Dataset):
    def __init__(self, base, replicate: int, max_radius_gap: float = 0.10):
        if base.representation != "z" or not 0 <= replicate < 10:
            raise ValueError("fingerprint swap requires a Z model and replicate 0..9")
        self.base = base
        self.replicate = replicate
        self.max_radius_gap = max_radius_gap
        self.radius_gaps = []
        pool = []
        for sample_index, record in base.records.items():
            for object_index, obj in enumerate(record["objects"]):
                pool.append((sample_index, object_index, obj["shape"],
                             bounding_radius(obj), obj["library_id"]))
        by_family = {}
        for item in pool:
            by_family.setdefault(item[2], []).append(item)
        for family in by_family:
            by_family[family].sort(key=lambda item: item[3])
        radii = {family: [item[3] for item in entries]
                 for family, entries in by_family.items()}
        self.donor = {}
        for sample_index, object_index, family, radius, library_id in pool:
            family_pool = by_family[family]
            lower = bisect_left(radii[family], radius - max_radius_gap)
            upper = bisect_right(radii[family], radius + max_radius_gap)
            candidates = [item for item in family_pool[lower:upper]
                          if item[4] != library_id]
            if not candidates:
                raise RuntimeError(f"no radius-matched donor for {family} {radius:.3f}")
            nearest_gap = min(abs(radius - item[3]) for item in candidates)
            candidates = [item for item in candidates
                          if abs(radius - item[3]) <= nearest_gap + 0.01]
            token = json.dumps(["canonical-response-v3", "z-swap", replicate,
                                sample_index, object_index], separators=(",", ":"))
            pick = int.from_bytes(hashlib.sha256(token.encode()).digest()[:8], "little")
            chosen = candidates[pick % len(candidates)]
            self.donor[(sample_index, object_index)] = chosen[:2]
            self.radius_gaps.append(abs(radius - chosen[3]))
        degree_rms = np.asarray(base.statistics["harmonic_degree_complex_rms"])
        self.scale = np.repeat(degree_rms, np.arange(1, 16, 2))

    def __len__(self):
        return len(self.base)

    def __getitem__(self, index):
        graph = self.base[index]
        sample_index = int(self.base.files[index].stem.split("_")[-1])
        for object_index in range(len(graph.node_features) // 128):
            donor_sample, donor_object = self.donor[(sample_index, object_index)]
            donor_file = self.base.root / f"sample_{donor_sample:05d}.npz"
            with np.load(donor_file) as opened:
                response = np.asarray(opened["Z"][donor_object], dtype=np.complex64)
            normalized = response / self.scale[None, :]
            channels = np.concatenate((normalized.real, normalized.imag), axis=-1)
            start = object_index * 128
            graph.node_features[start:start + 128, :128] = torch.from_numpy(
                channels.astype(np.float32),
            )
        return graph
