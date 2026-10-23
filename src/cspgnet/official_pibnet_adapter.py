"""Adapt canonical-surface samples to the official three-level PIBNet API.

The canonical surface makes all local graph relations geometry invariant. We
cache those relations once, while regenerating the paper's sparse long-range
edges deterministically from each scene's translated surface positions.
"""

from __future__ import annotations

import json
from pathlib import Path

import dgl
import numpy as np
import torch
from scipy.spatial import ConvexHull
from torch.utils.data import Dataset

from data_generation.data_gen_utils.binary_tree import TreeOpTransform


NON_PROBE_NODE_DIM = 12
DEFAULT_NUM_PROBES = 64
NODE_DIM = 2 * DEFAULT_NUM_PROBES + NON_PROBE_NODE_DIM
EDGE_DIM = 134
NODES_PER_OBJECT = 128
DISTANT_EDGE_RATIO = 0.1
CANDIDATE_EDGE_RATIO = 2
EDGE_TYPES = (
    "neighbors",
    "distant_0",
    "down_0_1",
    "up_1_0",
    "down_1_2",
    "up_2_1",
)


_TOPOLOGY_CACHE: dict[tuple[int, int, bytes, bytes], dict] = {}


def _positional_encoding(dist: torch.Tensor, channels: int = 128) -> torch.Tensor:
    max_dist = 10.0 * np.sqrt(2.0)
    omega_min = torch.pi / (2.0 * max_dist)
    omega_max = torch.pi / (2.0 * 0.1)
    div_term = torch.exp(
        torch.linspace(0.0, 1.0, channels // 2, device=dist.device, dtype=dist.dtype)
        * np.log(omega_min / omega_max)
    ).unsqueeze(0) * omega_max
    return torch.cat([torch.sin(dist * div_term), -torch.cos(dist * div_term)], dim=-1)


def _wavelength_encoding(wavelength: torch.Tensor) -> torch.Tensor:
    min_inv = 1.0 / 6.0
    max_inv = 1.0 / 0.6
    return (1.0 / wavelength - min_inv) / (max_inv - min_inv) * 2.0 - 1.0


def _unique_edges(source: torch.Tensor, destination: torch.Tensor, num_nodes: int):
    source = source.to(torch.int64)
    destination = destination.to(torch.int64)
    keep = source != destination
    source, destination = source[keep], destination[keep]
    if source.numel() == 0:
        return source, destination
    keys = torch.unique(source * num_nodes + destination, sorted=True)
    return torch.div(keys, num_nodes, rounding_mode="floor"), keys.remainder(num_nodes)


def _surface_edges(
    interface_points: np.ndarray,
    interface_faces: np.ndarray | None,
    objects: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if interface_faces is None:
        interface_faces = ConvexHull(interface_points).simplices
    faces = torch.as_tensor(np.asarray(interface_faces), dtype=torch.int64)
    undirected = torch.cat(
        [faces[:, (0, 1)], faces[:, (1, 2)], faces[:, (2, 0)]], dim=0
    )
    local_source = torch.cat([undirected[:, 0], undirected[:, 1]])
    local_destination = torch.cat([undirected[:, 1], undirected[:, 0]])
    local_source, local_destination = _unique_edges(
        local_source, local_destination, len(interface_points)
    )
    sources, destinations = [], []
    for obj in range(objects):
        offset = obj * len(interface_points)
        sources.append(local_source + offset)
        destinations.append(local_destination + offset)
    return torch.cat(sources), torch.cat(destinations)


def _tree_edges(interface_points: torch.Tensor, objects: int, nodes_per_object: int):
    transform = TreeOpTransform(2)
    down = {1: [], 2: []}
    levels = [
        torch.ones(objects * nodes_per_object, dtype=torch.bool),
        torch.zeros(objects * nodes_per_object, dtype=torch.bool),
        torch.zeros(objects * nodes_per_object, dtype=torch.bool),
    ]
    local_edges = transform(interface_points)
    for obj in range(objects):
        offset = obj * nodes_per_object
        for level in (1, 2):
            edge = local_edges[level].to(torch.int64) + offset
            source, destination = _unique_edges(
                edge[1], edge[0], objects * nodes_per_object
            )
            down[level].append((source, destination))
            levels[level][torch.unique(destination)] = True

    down_edges, up_edges = {}, {}
    for level in (1, 2):
        source = torch.cat([edge[0] for edge in down[level]])
        destination = torch.cat([edge[1] for edge in down[level]])
        source, destination = _unique_edges(
            source, destination, objects * nodes_per_object
        )
        down_edges[level] = (source, destination)
        up_edges[level] = _unique_edges(
            destination, source, objects * nodes_per_object
        )
    return levels, down_edges, up_edges


def _paper_sparse_distant_edges(
    top_nodes: torch.Tensor,
    global_points: torch.Tensor,
    seed: int,
    new_edges_ratio: float = DISTANT_EDGE_RATIO,
    candidate_edge_ratio: int = CANDIDATE_EDGE_RATIO,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Paper rule: sample candidates and retain the nearest proposal."""
    num_top = int(top_nodes.numel())
    generator = torch.Generator().manual_seed(int(seed) & 0x7FFF_FFFF_FFFF_FFFF)
    new_edges_per_source = new_edges_ratio * num_top
    if new_edges_per_source < 1.0:
        count = int(num_top * new_edges_per_source)
        local_source = torch.randperm(num_top, generator=generator)[:count]
    else:
        count = int(np.round(new_edges_per_source))
        local_source = torch.arange(num_top).view(-1, 1).expand(-1, count).reshape(-1)
    if local_source.numel() == 0:
        return torch.empty(0, dtype=torch.int64), torch.empty(0, dtype=torch.int64)

    candidates = torch.randint(
        0,
        num_top,
        (local_source.numel(), candidate_edge_ratio),
        generator=generator,
    )
    top_positions = global_points[top_nodes]
    candidate_delta = top_positions[candidates] - top_positions[local_source, None, :]
    selected = candidate_delta.norm(dim=-1).argmin(dim=-1)
    local_destination = candidates[torch.arange(len(candidates)), selected]
    source = top_nodes[local_source]
    destination = top_nodes[local_destination]
    forward_source, forward_destination = source, destination
    source = torch.cat([forward_source, forward_destination])
    destination = torch.cat([forward_destination, forward_source])
    return _unique_edges(source, destination, len(global_points))


def _edge_features_from_delta(delta: torch.Tensor, k: float):
    dist = delta.norm(dim=-1, keepdim=True).clamp_min(1.0e-8)
    rel_dir = delta / dist
    return _edge_features_from_cached_geometry(_positional_encoding(dist), dist, rel_dir, k)


def _edge_features_from_cached_geometry(
    positional: torch.Tensor,
    dist: torch.Tensor,
    rel_dir: torch.Tensor,
    k: float,
):
    wavelength = torch.full_like(dist, 2.0 * torch.pi / k)
    phase = torch.cat([torch.sin(k * dist), torch.cos(k * dist)], dim=-1)
    return torch.cat(
        [positional, phase, rel_dir, _wavelength_encoding(wavelength)], dim=-1
    )


def _topology_key(
    interface_points: np.ndarray,
    interface_faces: np.ndarray | None,
    objects: int,
    nodes_per_object: int,
):
    face_bytes = b"" if interface_faces is None else np.ascontiguousarray(interface_faces).tobytes()
    return (
        objects,
        nodes_per_object,
        np.ascontiguousarray(interface_points).tobytes(),
        face_bytes,
    )


def _get_topology(
    interface_points: np.ndarray,
    interface_faces: np.ndarray | None,
    objects: int,
    nodes_per_object: int,
):
    key = _topology_key(interface_points, interface_faces, objects, nodes_per_object)
    cached = _TOPOLOGY_CACHE.get(key)
    if cached is not None:
        return cached

    interface = torch.from_numpy(np.ascontiguousarray(interface_points)).float()
    local_points = interface[None, :, :].expand(objects, -1, -1).reshape(-1, 3)
    neighbor = _surface_edges(interface_points, interface_faces, objects)
    levels, down, up = _tree_edges(interface, objects, nodes_per_object)
    graph_data = {
        "neighbors": neighbor,
        "down_0_1": down[1],
        "up_1_0": up[1],
        "down_1_2": down[2],
        "up_2_1": up[2],
    }
    local_edge_geometry = {}
    for etype, (source, destination) in graph_data.items():
        delta = local_points[destination] - local_points[source]
        distance = delta.norm(dim=-1, keepdim=True).clamp_min(1.0e-8)
        local_edge_geometry[etype] = {
            "distance": distance,
            "direction": delta / distance,
            "positional": _positional_encoding(distance),
        }
    cached = {
        "graph_data": graph_data,
        "local_edge_geometry": local_edge_geometry,
        "levels": levels,
        "top_nodes": torch.where(levels[2])[0],
    }
    _TOPOLOGY_CACHE[key] = cached
    return cached


def _complex_array(data: dict[str, np.ndarray], key: str, legacy_prefix: str) -> np.ndarray:
    if key in data:
        return np.asarray(data[key])
    if legacy_prefix in data:
        return np.asarray(data[legacy_prefix])
    real_key = f"{legacy_prefix}_real"
    imag_key = f"{legacy_prefix}_imag"
    if real_key not in data or imag_key not in data:
        raise KeyError(f"missing {key} and legacy pair {real_key}/{imag_key}")
    return np.asarray(data[real_key]) + 1j * np.asarray(data[imag_key])


def fit_complex_normalization(files: list[Path]) -> dict:
    """Fit phase-preserving RMS scales using only the selected training files."""
    if not files:
        raise ValueError("cannot fit normalization without training samples")
    probe_power = np.zeros(DEFAULT_NUM_PROBES, dtype=np.float64)
    probe_count = 0
    target_power = 0.0
    target_count = 0
    surface_radius = None
    for index, path in enumerate(files, start=1):
        with np.load(path) as loaded:
            data = {key: loaded[key] for key in loaded.files}
        z = np.asarray(_complex_array(data, "Z", "Z"), dtype=np.complex128)
        target = np.asarray(
            _complex_array(data, "u_total_gt", "u_surface_gt"), dtype=np.complex128
        )
        if z.ndim != 3 or z.shape[-1] != DEFAULT_NUM_PROBES:
            raise ValueError(f"unexpected Z shape in {path}: {z.shape}")
        if target.shape != z.shape[:2]:
            raise ValueError(f"target shape {target.shape} does not match Z in {path}")
        probe_power += np.square(z.real).sum(axis=(0, 1)) + np.square(z.imag).sum(axis=(0, 1))
        probe_count += int(np.prod(z.shape[:2]))
        target_power += float(np.square(target.real).sum() + np.square(target.imag).sum())
        target_count += int(target.size)
        if surface_radius is None:
            points = np.asarray(data["interface_points"], dtype=np.float64)
            radii = np.linalg.norm(points, axis=-1)
            surface_radius = float(np.median(radii))
            if surface_radius <= 0 or not np.allclose(radii, surface_radius, rtol=1e-5, atol=1e-6):
                raise ValueError(f"interface_points are not on a centered sphere in {path}")
        if index % 1000 == 0 or index == len(files):
            print(f"normalization statistics: {index}/{len(files)} training samples", flush=True)

    probe_rms = np.sqrt(np.maximum(probe_power / probe_count, 1.0e-24))
    target_rms = float(np.sqrt(max(target_power / target_count, 1.0e-24)))
    return {
        "schema": "complex_shared_rms_v1",
        "fit_split": "train",
        "fit_samples": len(files),
        "probe_rms_shared_real_imag": probe_rms.tolist(),
        "total_field_rms_shared_real_imag": target_rms,
        "canonical_surface_radius": surface_radius,
        "centering": "none",
        "edge_features": "unchanged; positional/phase encodings and directions are bounded",
    }


def _prepare_arrays(data: dict[str, np.ndarray]):
    z = _complex_array(data, "Z", "Z")
    target = _complex_array(data, "u_total_gt", "u_surface_gt")
    incident = _complex_array(data, "u_inc", "incident_surface")
    if "phi_pinv" in data:
        phi_pinv = np.asarray(data["phi_pinv"])
    else:
        phi_pinv = np.linalg.pinv(np.asarray(data["probe_values"]))
    if "q64_gt" in data:
        q64 = np.asarray(data["q64_gt"])
    else:
        coefficients = np.einsum("mn,on->om", phi_pinv, target, optimize=True)
        q64 = np.einsum("onm,om->on", z, coefficients, optimize=True)
    return z, target, incident, q64, phi_pinv


def sample_to_graph(
    data: dict[str, np.ndarray],
    *,
    edge_seed: int = 0,
    sample_index: int = 0,
    zero_physics: bool = False,
    normalization: dict | None = None,
) -> dgl.DGLHeteroGraph:
    z_complex, target_complex, incident_complex, q_complex, _ = _prepare_arrays(data)
    z = torch.from_numpy(
        np.concatenate([z_complex.real, z_complex.imag], axis=-1).astype(np.float32)
    )
    objects, nodes_per_object, probes_twice = z.shape
    if nodes_per_object != NODES_PER_OBJECT:
        raise ValueError(f"formal graph requires {NODES_PER_OBJECT} nodes per object")
    interface_array = np.asarray(data["interface_points"])
    interface_faces = np.asarray(data["interface_faces"]) if "interface_faces" in data else None
    interface = torch.from_numpy(interface_array).float()
    centers = torch.from_numpy(np.asarray(data["centers"])).float()
    k = float(data["k"])
    direction = torch.from_numpy(np.asarray(data["incident_direction"])).float()
    incident = torch.from_numpy(
        np.stack([incident_complex.real, incident_complex.imag], axis=-1).astype(np.float32)
    )

    topology = _get_topology(interface_array, interface_faces, objects, nodes_per_object)
    local_points = interface[None, :, :].expand(objects, -1, -1)
    global_points = (local_points + centers[:, None, :]).reshape(objects * nodes_per_object, 3)
    distant = _paper_sparse_distant_edges(topology["top_nodes"], global_points, seed=edge_seed)
    graph_data = {
        ("obs", "neighbors", "obs"): topology["graph_data"]["neighbors"],
        ("obs", "distant_0", "obs"): distant,
        ("obs", "down_0_1", "obs"): topology["graph_data"]["down_0_1"],
        ("obs", "up_1_0", "obs"): topology["graph_data"]["up_1_0"],
        ("obs", "down_1_2", "obs"): topology["graph_data"]["down_1_2"],
        ("obs", "up_2_1", "obs"): topology["graph_data"]["up_2_1"],
    }
    graph = dgl.heterograph(
        graph_data,
        num_nodes_dict={"obs": objects * nodes_per_object},
        idtype=torch.int64,
    )
    for level, mask in enumerate(topology["levels"]):
        graph.ndata[f"lvl_{level}"] = mask

    normals = local_points / local_points.norm(dim=-1, keepdim=True).clamp_min(1.0e-8)
    wavelength = torch.tensor(2.0 * torch.pi / k, dtype=torch.float32)
    wavelength_pe = _wavelength_encoding(wavelength.reshape(1, 1)).expand(
        objects, nodes_per_object, 1
    )
    feature_points = local_points
    target_scale = 1.0
    if normalization is not None:
        probe_rms = torch.as_tensor(
            normalization["probe_rms_shared_real_imag"], dtype=z.dtype
        )
        z_scale = torch.cat([probe_rms, probe_rms]).view(1, 1, -1)
        z = z / z_scale
        feature_points = local_points / float(normalization["canonical_surface_radius"])
        target_scale = float(normalization["total_field_rms_shared_real_imag"])
    z_for_input = torch.zeros_like(z) if zero_physics else z
    node_features = torch.cat(
        [
            z_for_input,
            incident,
            feature_points,
            normals,
            direction[None, None, :].expand(objects, nodes_per_object, -1),
            wavelength_pe,
        ],
        dim=-1,
    )
    expected_node_dim = probes_twice + NON_PROBE_NODE_DIM
    if node_features.shape[-1] != expected_node_dim:
        raise ValueError(
            f"node feature width mismatch: got {node_features.shape[-1]}, expected {expected_node_dim}"
        )
    graph.ndata["node_features"] = node_features.reshape(-1, expected_node_dim)
    graph.ndata["labels"] = torch.from_numpy(
        np.stack([target_complex.real, target_complex.imag], axis=-1).astype(np.float32)
        / target_scale
    ).reshape(-1, 2)
    graph.ndata["incident"] = incident.reshape(-1, 2)
    graph.ndata["q64_gt"] = torch.from_numpy(
        np.stack([q_complex.real, q_complex.imag], axis=-1).astype(np.float32)
    ).reshape(-1, 2)
    graph.ndata["z_real"] = torch.from_numpy(z_complex.real.astype(np.float32)).reshape(
        -1, z_complex.shape[-1]
    )
    graph.ndata["z_imag"] = torch.from_numpy(z_complex.imag.astype(np.float32)).reshape(
        -1, z_complex.shape[-1]
    )
    graph.ndata["position"] = global_points
    graph.ndata["sample_index"] = torch.full(
        (graph.num_nodes(),), int(sample_index), dtype=torch.int64
    )

    for etype in EDGE_TYPES:
        source, destination = graph.edges(etype=etype)
        if etype == "distant_0":
            delta = global_points[destination] - global_points[source]
            features = _edge_features_from_delta(delta, k)
        else:
            geometry = topology["local_edge_geometry"][etype]
            features = _edge_features_from_cached_geometry(
                geometry["positional"], geometry["distance"], geometry["direction"], k
            )
        graph.edges[etype].data["edge_features"] = features
    return graph


def sample_to_edge_graph(
    data: dict[str, np.ndarray],
    *,
    edge_seed: int = 0,
    sample_index: int = 0,
    zero_physics: bool = False,
    normalization: dict | None = None,
):
    """Build the same tensors as ``sample_to_graph`` without a transient DGL graph.

    The fixed local/down/up topology is drawn from the same cache as the DGL
    reference. Only the paper's sparse distant edges depend on the scene and
    epoch. A separate parity test checks every tensor before this path is used
    for long-running training.
    """
    from cspgnet.official_pibnet_edge_backend import EdgeIndexGraphBatch

    z_complex, target_complex, incident_complex, q_complex, _ = _prepare_arrays(data)
    z = torch.from_numpy(
        np.concatenate([z_complex.real, z_complex.imag], axis=-1).astype(np.float32)
    )
    objects, nodes_per_object, probes_twice = z.shape
    if nodes_per_object != NODES_PER_OBJECT:
        raise ValueError(f"formal graph requires {NODES_PER_OBJECT} nodes per object")
    interface_array = np.asarray(data["interface_points"])
    interface_faces = np.asarray(data["interface_faces"]) if "interface_faces" in data else None
    interface = torch.from_numpy(interface_array).float()
    centers = torch.from_numpy(np.asarray(data["centers"])).float()
    k = float(data["k"])
    direction = torch.from_numpy(np.asarray(data["incident_direction"])).float()
    incident = torch.from_numpy(
        np.stack([incident_complex.real, incident_complex.imag], axis=-1).astype(np.float32)
    ).reshape(-1, 2)
    topology = _get_topology(interface_array, interface_faces, objects, nodes_per_object)
    local_points = interface[None, :, :].expand(objects, -1, -1)
    global_points = (local_points + centers[:, None, :]).reshape(objects * nodes_per_object, 3)
    distant = _paper_sparse_distant_edges(topology["top_nodes"], global_points, seed=edge_seed)

    normals = local_points / local_points.norm(dim=-1, keepdim=True).clamp_min(1.0e-8)
    wavelength = torch.tensor(2.0 * torch.pi / k, dtype=torch.float32)
    wavelength_pe = _wavelength_encoding(wavelength.reshape(1, 1)).expand(
        objects, nodes_per_object, 1
    )
    feature_points = local_points
    target_scale = 1.0
    if normalization is not None:
        probe_rms = torch.as_tensor(
            normalization["probe_rms_shared_real_imag"], dtype=z.dtype
        )
        z_scale = torch.cat([probe_rms, probe_rms]).view(1, 1, -1)
        z = z / z_scale
        feature_points = local_points / float(normalization["canonical_surface_radius"])
        target_scale = float(normalization["total_field_rms_shared_real_imag"])
    node_features = torch.cat(
        [
            torch.zeros_like(z) if zero_physics else z,
            incident.reshape(objects, nodes_per_object, 2),
            feature_points,
            normals,
            direction[None, None, :].expand(objects, nodes_per_object, -1),
            wavelength_pe,
        ],
        dim=-1,
    ).reshape(-1, probes_twice + NON_PROBE_NODE_DIM)

    edge_index = {**topology["graph_data"], "distant_0": distant}
    edge_features = {}
    for etype in EDGE_TYPES:
        if etype == "distant_0":
            source, destination = distant
            delta = global_points[destination] - global_points[source]
            edge_features[etype] = _edge_features_from_delta(delta, k)
        else:
            geometry = topology["local_edge_geometry"][etype]
            edge_features[etype] = _edge_features_from_cached_geometry(
                geometry["positional"], geometry["distance"], geometry["direction"], k
            )
    nodes = objects * nodes_per_object
    return EdgeIndexGraphBatch(
        node_features=node_features,
        labels=torch.from_numpy(
            np.stack([target_complex.real, target_complex.imag], axis=-1).astype(np.float32)
            / target_scale
        ).reshape(-1, 2),
        incident=incident,
        q64_gt=torch.from_numpy(
            np.stack([q_complex.real, q_complex.imag], axis=-1).astype(np.float32)
        ).reshape(-1, 2),
        z_real=torch.from_numpy(z_complex.real.astype(np.float32)).reshape(-1, z_complex.shape[-1]),
        z_imag=torch.from_numpy(z_complex.imag.astype(np.float32)).reshape(-1, z_complex.shape[-1]),
        batch_index=torch.zeros(nodes, dtype=torch.int64),
        sample_index=torch.full((nodes,), int(sample_index), dtype=torch.int64),
        sample_node_counts=torch.tensor([nodes], dtype=torch.int64),
        edge_features=edge_features,
        edge_index={etype: torch.stack(edge_index[etype], dim=0) for etype in EDGE_TYPES},
        level_masks={level: topology["levels"][level] for level in range(3)},
    )


class OfficialPIBNetCanonicalDataset(Dataset):
    def __init__(
        self,
        root: str | Path,
        *,
        zero_physics: bool = False,
        require_complete: bool = True,
        edge_direct: bool = False,
    ):
        self.root = Path(root)
        self.files = sorted(self.root.glob("sample_*.npz"))
        if not self.files:
            raise FileNotFoundError(f"No sample_*.npz files found in {root}")
        self.zero_physics = zero_physics
        self.edge_direct = edge_direct
        self.normalization = None
        self.epoch = 0
        self.manifest = {}
        manifest_path = self.root / "manifest.json"
        if manifest_path.exists():
            self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if require_complete and self.manifest.get("schema_version") in (
                "cspgnet-neumann-3d-v1", "cspgnet-neumann-tmatrix-v2"
            ):
                if self.manifest.get("status") != "complete":
                    raise RuntimeError(f"validated dataset is not complete: {root}")
                evidence_key = (
                    "formal_evidence" if self.manifest["schema_version"] == "cspgnet-neumann-3d-v1"
                    else "validated_tmatrix_evidence"
                )
                if not self.manifest.get(evidence_key) or not self.manifest.get("numerical_gate"):
                    raise RuntimeError(
                        f"validated dataset has no passed numerical-gate provenance: {root}"
                    )
        self.base_seed = int(self.manifest.get("seed", 0))
        with np.load(self.files[0]) as loaded:
            data = {key: loaded[key] for key in loaded.files}
        _, _, _, _, phi_pinv = _prepare_arrays(data)
        self.phi_pinv = torch.from_numpy(phi_pinv.astype(np.complex64))

    def __len__(self):
        return len(self.files)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    @property
    def node_feature_dim(self):
        with np.load(self.files[0]) as loaded:
            if "Z" in loaded.files:
                num_probes = int(loaded["Z"].shape[-1])
            else:
                num_probes = int(loaded["Z_real"].shape[-1])
        return 2 * num_probes + NON_PROBE_NODE_DIM

    @property
    def schema_version(self) -> str:
        return str(self.manifest.get("schema_version", "legacy-canonical"))

    def __getitem__(self, index):
        with np.load(self.files[index]) as loaded:
            data = {key: loaded[key] for key in loaded.files}
        edge_seed = self.base_seed + 1_000_003 * int(index) + 97_409 * self.epoch
        builder = sample_to_edge_graph if self.edge_direct else sample_to_graph
        return builder(
            data,
            edge_seed=edge_seed,
            sample_index=index,
            zero_physics=self.zero_physics,
            normalization=self.normalization,
        )


def collate_graphs(samples):
    return dgl.batch(samples)
