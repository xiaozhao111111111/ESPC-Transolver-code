"""Adapt canonical-surface samples to the official three-level PIBNet API.

The canonical surface makes all local graph relations geometry invariant. We
cache those relations once, while regenerating the paper's sparse long-range
edges deterministically from each scene's translated surface positions."""

from __future__ import annotations




import numpy as np

import torch

from scipy.spatial import ConvexHull


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
