from dataclasses import dataclass
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
from PIBNet.S_Z.official_pibnet_adapter import (EDGE_TYPES, _edge_features_from_cached_geometry, _edge_features_from_delta, _get_topology, _paper_sparse_distant_edges)

NODES = 128

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
        if method == "transolver-geo-strong" and (geometry_points is None or geometry_normals is None):
            raise ValueError("transolver-geo-strong requires the verified point-cloud library")

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
        if self.method == "transolver-geo-matched":
            occ = np.asarray(a["occupancy"][o0:o1], dtype=np.float32)
            features[:, :128] = ((occ - np.asarray(self.stats["occupancy_mean"], np.float32)) /
                                  np.asarray(self.stats["occupancy_std"], np.float32)).reshape(-1, 128)
        elif self.method == "transolver-geo-strong":
            features[:, :128] = 0.0
        elif self.method == "pibnet":
            # Match the official PIBNet feature contract: local coordinates and
            # local normals; object centers enter only through relative edges.
            features[:, 133:136] = np.tile(self.interface / 1.5, (o1 - o0, 1))

        geo_local = geo_global = None
        if self.method == "transolver-geo-strong":
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
        if self.method == "pibnet":
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
