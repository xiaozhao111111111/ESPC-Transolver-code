"""Matched input representations for the v3 official Transolver experiments."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from torch import nn

from cspgnet.v3_inputs import V3CanonicalDataset
from cspgnet.v3_pointcloud import PointCloudCache
from cspgnet.v3_feature_trials import scattering_response
from cspgnet.v3_transolver_comparison import GroupedTransolver, STransformerDataset


NODES_PER_OBJECT = 128


class CenteredChannelsDataset(torch.utils.data.Dataset):
    """Apply the same scene-centered object-position channels to every arm."""

    def __init__(self, base):
        self.base = base
        self.files = base.files
        self.records = base.records
        self.manifest = base.manifest
        self.phi_pinv = base.phi_pinv

    def __len__(self):
        return len(self.base)

    def set_epoch(self, epoch: int):
        self.base.set_epoch(epoch)

    def __getitem__(self, index):
        graph = self.base[index]
        sample_number = int(self.files[index].stem.rsplit("_", 1)[1])
        centers = torch.as_tensor(self.records[sample_number]["centers"], dtype=torch.float32)
        if graph.node_features.shape != (len(centers) * NODES_PER_OBJECT, 140):
            raise ValueError("representation features, object centers, and node count disagree")
        relative = (centers - centers.mean(dim=0, keepdim=True)) / 1.5
        graph.node_features[:, 133:136] = torch.repeat_interleave(
            relative, NODES_PER_OBJECT, dim=0
        )
        return graph


class OnTheFlySResponseDataset(torch.utils.data.Dataset):
    """Build S from stored Z for evaluation splits without touching their labels."""

    def __init__(self, root: Path, statistics: dict, train_scales):
        self.base = V3CanonicalDataset(root, statistics, "z", "scattered", require_complete=True)
        self.files = self.base.files
        self.records = self.base.records
        self.manifest = self.base.manifest
        self.phi_pinv = self.base.phi_pinv
        self.scales = np.repeat(np.asarray(train_scales, dtype=np.float64), np.arange(1, 16, 2))

    def __len__(self):
        return len(self.base)

    def set_epoch(self, epoch: int):
        self.base.set_epoch(epoch)

    def __getitem__(self, index):
        graph = self.base[index]
        with np.load(self.files[index]) as opened:
            sample = {key: opened[key] for key in opened.files}
        response, _ = scattering_response(sample)
        response = response.astype(np.complex128) / self.scales[None, None, :]
        channels = np.concatenate((response.real, response.imag), axis=-1).astype(np.float32)
        graph.node_features[:, :128] = torch.from_numpy(channels.reshape(-1, 128))
        return graph


def make_dataset(representation: str, root: Path, statistics: dict, *,
                 phase: str, s_cache: Path, train_s_scales,
                 pointcloud_cache: Path):
    if representation == "s-response":
        if phase == "train":
            base = STransformerDataset(root, statistics, s_cache, train_s_scales)
        else:
            base = OnTheFlySResponseDataset(root, statistics, train_s_scales)
    elif representation in ("raw-z", "geo-matched", "geo-strong"):
        v3_name = {"raw-z": "z", "geo-matched": "geo-matched",
                   "geo-strong": "geo-strong-adapted"}[representation]
        base = V3CanonicalDataset(
            root, statistics, v3_name, "scattered",
            pointcloud_cache=pointcloud_cache if representation == "geo-strong" else None,
            require_complete=True,
        )
    else:
        raise ValueError(f"unknown representation: {representation}")
    return CenteredChannelsDataset(base)


class TransolverGeoStrong(nn.Module):
    """Point-cloud encoder followed by the unchanged official Transolver trunk."""

    def __init__(self):
        super().__init__()
        # Initialize the shared trunk first so seed-0 trunk weights match the
        # other three arms; the point-cloud encoder is the only added capacity.
        self.backbone = GroupedTransolver("transolver")
        self.local_mlp = nn.Sequential(nn.Linear(6, 64), nn.SiLU(), nn.Linear(64, 128))
        self.global_mlp = nn.Sequential(nn.Linear(6, 64), nn.SiLU(), nn.Linear(64, 128))
        self.combiner = nn.Linear(384, 128)

    def forward(self, graph):
        if graph.geo_local is None or graph.geo_global is None:
            raise ValueError("Geo-strong requires nearest-surface and full point-cloud tensors")
        local = self.local_mlp(graph.geo_local).amax(dim=1)
        global_points = self.global_mlp(graph.geo_global)
        global_code = torch.cat((global_points.amax(dim=1), global_points.mean(dim=1)), dim=-1)
        object_counts = graph.sample_node_counts // NODES_PER_OBJECT
        if int(object_counts.sum()) != global_code.shape[0]:
            raise ValueError("point-cloud objects do not align with Transolver scenes")
        global_per_node = torch.repeat_interleave(global_code, NODES_PER_OBJECT, dim=0)
        if global_per_node.shape[0] != local.shape[0]:
            raise ValueError("point-cloud object/node alignment failed")
        encoded = self.combiner(torch.cat((local, global_per_node), dim=-1))
        features = graph.node_features.clone()
        features[:, :128] = encoded
        return self.backbone(replace(graph, node_features=features))


def make_model(representation: str) -> nn.Module:
    if representation == "geo-strong":
        return TransolverGeoStrong()
    if representation in ("s-response", "raw-z", "geo-matched"):
        return GroupedTransolver("transolver")
    raise ValueError(representation)
