"""Frozen, train-only normalization and matched geometry inputs for v3."""

from __future__ import annotations

import os
import json
import multiprocessing
from pathlib import Path

os.environ.setdefault("DGLBACKEND", "pytorch")
os.environ.setdefault("DGL_SKIP_GRAPHBOLT", "1")
if os.name == "nt":
    _omp = Path(__file__).resolve().parents[3] / "openmp_runtime_20260920/package/runtimes/win-x64/native"
    if _omp.is_dir():
        os.add_dll_directory(str(_omp))

import numpy as np
import torch

from cspgnet.data_generation import formal_neumann_3d as formal
from cspgnet.official_pibnet_adapter import sample_to_edge_graph
from cspgnet.v3_pointcloud import PointCloudCache, attach_geostrong_features
from cspgnet.v3_protocol import derived_seed


SHAPES = ("sphere", "ellipsoid", "box", "peanut", "torus")
RADII = 0.9 * (np.arange(128, dtype=np.float64) + 0.5) / 128.0
RADIAL_VARIANCE_FLOOR = 1.0e-4


def radial_occupancy(data: dict[str, np.ndarray]) -> np.ndarray:
    """Return exactly 128 radial occupancy channels per canonical node."""
    if "geometry_occupancy" in data:
        occupied = np.asarray(data["geometry_occupancy"])
        expected_radii = 0.9 * (np.arange(128, dtype=np.float64) + 0.5) / 128
        if occupied.shape != (len(data["shape_codes"]), 128, 128):
            raise ValueError("precomputed geometry occupancy must be [objects,128 canonical nodes,128 radii]")
        if not (np.issubdtype(occupied.dtype, np.bool_) or np.issubdtype(occupied.dtype, np.integer)):
            raise ValueError("precomputed geometry occupancy must be a boolean/integer mask")
        if not np.isin(occupied, (0, 1)).all():
            raise ValueError("geometry occupancy must contain only 0/1")
        if data.get("geometry_occupancy_schema") != "canonical-rays-r0.9-128-midpoints-v1":
            raise ValueError("unknown precomputed geometry occupancy convention")
        if "geometry_occupancy_radii" not in data or not np.array_equal(
            np.asarray(data["geometry_occupancy_radii"], dtype=np.float64), expected_radii,
        ):
            raise ValueError("precomputed occupancy radial coordinates do not match the frozen physical grid")
        return occupied.astype(np.float32, copy=False)
    points = np.asarray(data["interface_points"], dtype=np.float64)
    directions = points / np.linalg.norm(points, axis=1, keepdims=True)
    coordinates = directions[:, None, :] * RADII[None, :, None]
    codes = np.asarray(data["shape_codes"], dtype=int)
    sizes = np.asarray(data["shape_parameters"], dtype=np.float64)
    rotations = np.asarray(data["rotations"], dtype=np.float64)
    if rotations.shape != (len(codes), 3, 3):
        raise ValueError("missing actual SO(3) rotations for geometry baseline")
    result = np.empty((len(codes), len(points), len(RADII)), dtype=np.float32)
    for index, code in enumerate(codes):
        if not 0 <= code < len(SHAPES):
            raise ValueError(f"unknown shape code {code}")
        local = coordinates @ rotations[index]
        shape = SHAPES[code]
        size = sizes[index]
        if shape == "sphere":
            inside = np.linalg.norm(local, axis=-1) <= size[0]
        elif shape == "ellipsoid":
            inside = np.sum(np.square(local / size[None, None, :3]), axis=-1) <= 1.0
        elif shape == "box":
            inside = np.all(np.abs(local) <= size[None, None, :3], axis=-1)
        elif shape == "peanut":
            inside = formal.peanut_sdf(
                np.linalg.norm(local[..., :2], axis=-1), local[..., 2],
                size[0], size[1], size[2], 30.0,
            ) <= 0.0
        else:
            inside = np.hypot(
                np.linalg.norm(local[..., :2], axis=-1) - size[0], local[..., 2],
            ) <= size[1]
        result[index] = inside.astype(np.float32)
    return result


def fit_training_statistics(files: list[Path]) -> dict:
    if not files:
        raise ValueError("training statistics require nonempty training files")
    z_power = np.zeros(8, dtype=np.float64)
    z_count = np.zeros(8, dtype=np.int64)
    radial_sum = np.zeros(128, dtype=np.float64)
    radial_square = np.zeros(128, dtype=np.float64)
    radial_count = 0
    scatter_power = 0.0
    total_power = 0.0
    field_count = 0
    q_scene_energy = []
    for path in files:
        with np.load(path) as opened:
            data = {key: opened[key] for key in opened.files}
        z = np.asarray(data["Z"], dtype=np.complex128)
        if z.ndim != 3 or z.shape[-2:] != (128, 64):
            raise ValueError(f"invalid Z in {path}")
        for ell in range(8):
            block = z[..., ell * ell:(ell + 1) ** 2]
            z_power[ell] += float(np.sum(np.abs(block) ** 2))
            z_count[ell] += block.size
        occupancy = radial_occupancy(data).astype(np.float64)
        radial_sum += occupancy.sum(axis=(0, 1))
        radial_square += np.square(occupancy).sum(axis=(0, 1))
        radial_count += occupancy.shape[0] * occupancy.shape[1]
        total = np.asarray(data["u_total_gt"], dtype=np.complex128)
        incident = np.asarray(data["u_inc"], dtype=np.complex128)
        scatter = total - incident
        scatter_power += float(np.sum(np.abs(scatter) ** 2))
        total_power += float(np.sum(np.abs(total) ** 2))
        field_count += scatter.size
        q_scene_energy.append(float(np.sum(np.abs(data["q64_gt"]) ** 2)))
    radial_mean = radial_sum / radial_count
    radial_variance = np.maximum(radial_square / radial_count - radial_mean ** 2,
                                 RADIAL_VARIANCE_FLOOR)
    return {
        "schema": "canonical-response-v3-train-only-statistics",
        "fit_split": "train",
        "fit_samples": len(files),
        "harmonic_degree_complex_rms": np.sqrt(np.maximum(z_power / z_count, 1.0e-24)).tolist(),
        "radial_occupancy_mean": radial_mean.tolist(),
        "radial_occupancy_std": np.sqrt(radial_variance).tolist(),
        "radial_variance_floor": RADIAL_VARIANCE_FLOOR,
        "scatter_complex_rms": float(np.sqrt(scatter_power / field_count)),
        "total_complex_rms": float(np.sqrt(total_power / field_count)),
        "q_scene_energy_median": float(np.median(q_scene_energy)),
        "q_loss_stabilizer": float(1.0e-6 * np.median(q_scene_energy)),
        "canonical_radius": 1.5,
        "no_complex_mean_subtraction": True,
    }


def sample_to_v3_edge_graph(
    data: dict[str, np.ndarray], *, representation: str, statistics: dict,
    output_head: str = "scattered", edge_seed: int = 0, sample_index: int = 0,
):
    if representation not in ("z", "zero-z", "geo-matched", "geo-strong-adapted"):
        raise ValueError(f"unknown representation {representation}")
    if output_head not in ("scattered", "total"):
        raise ValueError(f"unknown output head {output_head}")
    if statistics.get("schema") != "canonical-response-v3-train-only-statistics":
        raise ValueError("v3 inputs require frozen train-only statistics")
    # This legacy builder supplies only the fixed graph and physical q labels;
    # zero_physics prevents raw Z from reaching geometry-only input channels.
    graph = sample_to_edge_graph(
        data, edge_seed=edge_seed, sample_index=sample_index,
        zero_physics=True, normalization=None,
    )
    if representation == "z":
        z = np.asarray(data["Z"], dtype=np.complex64)
        degree_scale = np.repeat(np.asarray(statistics["harmonic_degree_complex_rms"]),
                                 np.arange(1, 16, 2))
        z = z / degree_scale[None, None, :]
        first = np.concatenate((z.real, z.imag), axis=-1).astype(np.float32)
    elif representation == "geo-matched":
        occupancy = radial_occupancy(data)
        first = (occupancy - np.asarray(statistics["radial_occupancy_mean"])) / np.asarray(
            statistics["radial_occupancy_std"]
        )
        first = first.astype(np.float32)
    else:
        first = np.zeros((*np.asarray(data["Z"]).shape[:2], 128), dtype=np.float32)
    graph.node_features[:, :128] = torch.from_numpy(first.reshape(-1, 128))
    graph.node_features[:, 130:133] /= float(statistics["canonical_radius"])
    if output_head == "scattered":
        target = np.asarray(data["u_total_gt"] - data["u_inc"], dtype=np.complex64)
        scale = float(statistics["scatter_complex_rms"])
    else:
        target = np.asarray(data["u_total_gt"], dtype=np.complex64)
        scale = float(statistics["total_complex_rms"])
    graph.labels = torch.from_numpy(
        np.stack((target.real, target.imag), axis=-1).astype(np.float32).reshape(-1, 2) / scale,
    )
    return graph


class V3CanonicalDataset(torch.utils.data.Dataset):
    """Version-checked scene dataset with deterministic per-epoch long edges."""

    def __init__(self, root: Path, statistics: dict, representation: str,
                 output_head: str = "scattered", pointcloud_cache: Path | None = None,
                 require_complete: bool = True):
        self.root = Path(root)
        self.files = sorted(self.root.glob("sample_*.npz"))
        if not self.files:
            raise FileNotFoundError(self.root)
        self.manifest = json.loads((self.root / "manifest.json").read_text(encoding="utf-8"))
        if require_complete and self.manifest.get("status") != "complete":
            raise RuntimeError(f"dataset is incomplete: {self.root}")
        if self.manifest.get("schema_version") not in (
            "cspgnet-neumann-tmatrix-v2", "canonical-response-v3",
        ):
            raise ValueError("unsupported data schema for v3 training")
        if require_complete and self.manifest.get("schema_version") == "cspgnet-neumann-tmatrix-v2":
            if not self.manifest.get("validated_tmatrix_evidence"):
                raise RuntimeError("legacy T-matrix split lacks its old numerical gate")
        self.records = {int(record["index"]): record for record in self.manifest["samples"]}
        if len(self.records) != len(self.files):
            raise RuntimeError("manifest and sample files disagree")
        self.statistics = statistics
        self.representation = representation
        self.output_head = output_head
        # Spawned Windows DataLoader workers retain their Dataset instances.
        # A shared epoch makes remote-edge seeds follow the parent trainer even
        # when persistent workers are used to amortize costly imports.
        self._shared_epoch = multiprocessing.Value("i", 0)
        if representation == "geo-strong-adapted":
            if pointcloud_cache is None:
                raise ValueError("Geo-strong requires a fixed pointcloud cache directory")
            self.pointclouds = PointCloudCache(pointcloud_cache)
        else:
            self.pointclouds = None
        with np.load(self.files[0]) as opened:
            self.phi_pinv = torch.from_numpy(np.asarray(opened["phi_pinv"], dtype=np.complex64))

    def __len__(self):
        return len(self.files)

    def set_epoch(self, epoch: int):
        with self._shared_epoch.get_lock():
            self._shared_epoch.value = int(epoch)

    def __getitem__(self, index):
        with np.load(self.files[index]) as opened:
            data = {key: opened[key] for key in opened.files}
        sample_number = int(self.files[index].stem.split("_")[-1])
        epoch = self._shared_epoch.value
        component = "remote_edges" if epoch == 0 else f"remote_edges_epoch_{epoch}"
        edge_seed = derived_seed(self.manifest["split"], component, sample_number)
        graph = sample_to_v3_edge_graph(
            data, representation=self.representation, statistics=self.statistics,
            output_head=self.output_head, edge_seed=edge_seed, sample_index=sample_number,
        )
        if self.pointclouds is not None:
            attach_geostrong_features(graph, data, self.records[sample_number], self.pointclouds)
        return graph
