"""Deterministic area-weighted surface sampling for Geo-strong-adapted."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch
from scipy.spatial import cKDTree

from cspgnet.data_generation import formal_neumann_3d as formal
from cspgnet.data_generation.pointcloud_artifacts import load_precomputed_pointcloud


POINTS_PER_OBJECT = 2048
NEAREST_PER_NODE = 16
_CODE_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _seed(library_id: str, shape: str, size: tuple[float, ...]) -> int:
    payload = json.dumps(["v3-pointcloud", library_id, shape, size], separators=(",", ":"))
    return int.from_bytes(hashlib.sha256(payload.encode()).digest()[:8], "little")


def area_sampled_pointcloud(spec: formal.ObjectSpec, seed: int, maxh: float = 0.15):
    vertices, faces = formal._netgen_surface_arrays(spec, (0.0, 0.0, 0.0), maxh)
    a, b, c = (vertices[faces[:, index]] for index in range(3))
    cross = np.cross(b - a, c - a)
    length = np.linalg.norm(cross, axis=1)
    if np.any(length <= 0):
        raise ValueError("zero-area CAD triangle cannot be sampled")
    rng = np.random.default_rng(seed)
    selected = rng.choice(len(faces), size=POINTS_PER_OBJECT, p=length / length.sum())
    barycentric = rng.random((POINTS_PER_OBJECT, 2))
    root = np.sqrt(barycentric[:, 0])
    u = 1.0 - root
    v = root * (1.0 - barycentric[:, 1])
    w = root * barycentric[:, 1]
    points = u[:, None] * a[selected] + v[:, None] * b[selected] + w[:, None] * c[selected]
    normals = cross[selected] / length[selected, None]
    return points.astype(np.float32), normals.astype(np.float32)


class PointCloudCache:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.memory: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    def get(self, record: dict) -> tuple[np.ndarray, np.ndarray]:
        library_id = str(record["library_id"])
        shape = str(record["shape"])
        size = tuple(float(value) for value in record["size"])
        direct = load_precomputed_pointcloud(record, self.root, self.memory)
        if direct is not None:
            return direct
        fingerprint = hashlib.sha256(json.dumps(
            ["area-mesh-h0.15-v1", _CODE_SHA256, library_id, shape, size,
             record.get("smoothing_beta"), record.get("meridian_points", 65)],
            separators=(",", ":"),
        ).encode()).hexdigest()
        if fingerprint in self.memory:
            return self.memory[fingerprint]
        path = self.root / f"{fingerprint}.npz"
        if path.exists():
            with np.load(path) as opened:
                points = opened["points"]
                normals = opened["normals"]
                if str(opened["fingerprint"]) != fingerprint:
                    raise RuntimeError(f"pointcloud provenance mismatch: {path}")
        else:
            identity = tuple(tuple(float(value) for value in row) for row in np.eye(3))
            spec = formal.ObjectSpec(
                shape, size, identity,
                smoothing_beta=record.get("smoothing_beta"),
                meridian_points=int(record.get("meridian_points", 65)),
            )
            points, normals = area_sampled_pointcloud(spec, _seed(library_id, shape, size))
            self.root.mkdir(parents=True, exist_ok=True)
            pending = path.with_name(path.name + ".pending")
            with pending.open("wb") as handle:
                np.savez_compressed(handle, points=points, normals=normals, fingerprint=fingerprint)
            os.replace(pending, path)
        if points.shape != (POINTS_PER_OBJECT, 3) or normals.shape != points.shape:
            raise ValueError(f"invalid pointcloud cache: {path}")
        self.memory[fingerprint] = (points, normals)
        return points, normals


def attach_geostrong_features(graph, data: dict[str, np.ndarray], scene_record: dict,
                              cache: PointCloudCache) -> None:
    objects = scene_record["objects"]
    interface = np.asarray(data["interface_points"], dtype=np.float32)
    if len(objects) != len(data["Z"]):
        raise ValueError("scene geometry and operator object counts disagree")
    all_local = []
    all_global = []
    for obj in objects:
        points, normals = cache.get(obj)
        rotation = np.asarray(obj["rotation"], dtype=np.float32)
        points = points @ rotation.T
        normals = normals @ rotation.T
        neighbor = cKDTree(points).query(interface, k=NEAREST_PER_NODE)[1]
        offsets = (points[neighbor] - interface[:, None, :]) / 1.5
        all_local.append(np.concatenate((offsets, normals[neighbor]), axis=-1))
        all_global.append(np.concatenate((points / 1.5, normals), axis=-1))
    graph.geo_local = torch.from_numpy(np.concatenate(all_local, axis=0).astype(np.float32))
    graph.geo_global = torch.from_numpy(np.asarray(all_global, dtype=np.float32))
