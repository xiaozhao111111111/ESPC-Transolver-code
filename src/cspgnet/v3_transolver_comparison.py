"""Single-GPU adapters for the pinned official Transolver-family backbones.

Both arms consume the exact cached S derived from the saved complex64 Z in the
earlier Scheme-A experiment.  The only new spatial input is scene-centered
object position, which carries the relative-center information that PIBNet
receives through its inter-object edges.  It replaces a duplicate normal
channel rather than adding a new geometry descriptor.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

from cspgnet.v3_feature_trials import SResponseDataset


PROJECT = Path(__file__).resolve().parents[2]
BASE_ROOT = PROJECT / "third_party" / "Transolver" / "PDE-Solving-StandardBenchmark"
PLUS_SOURCE = PROJECT / "third_party" / "Transolver_plus" / "models" / "Transolver_plus.py"
NODES_PER_OBJECT = 128


class STransformerDataset(SResponseDataset):
    """Scheme-A input plus information-equivalent relative object centers."""

    def __getitem__(self, index):
        graph = super().__getitem__(index)
        with np.load(self.files[index]) as sample:
            centers = torch.from_numpy(np.asarray(sample["centers"], dtype=np.float32))
        if graph.node_features.shape[0] != len(centers) * NODES_PER_OBJECT:
            raise ValueError("object centers and canonical nodes do not match")
        # Columns 130:133 are unit local coordinates; 133:136 were the same
        # unit outward normals for a canonical sphere, hence redundant.
        if not torch.allclose(graph.node_features[:, 130:133],
                              graph.node_features[:, 133:136], atol=2e-5):
            raise ValueError("expected redundant local-coordinate and normal features")
        relative = (centers - centers.mean(dim=0, keepdim=True)) / 1.5
        graph.node_features[:, 133:136] = torch.repeat_interleave(
            relative, NODES_PER_OBJECT, dim=0
        )
        return graph


def _load_official_module(name: str, file: Path):
    spec = importlib.util.spec_from_file_location(name, file)
    if spec is None or spec.loader is None:
        raise ImportError(file)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class _SingleProcessDistributed:
    """Mathematically exact all-reduce shim for a world size of one."""

    class ReduceOp:
        SUM = "sum"

    @staticmethod
    def all_reduce(tensor, op=None):
        return tensor


class GroupedTransolver(nn.Module):
    """Apply an official point-set backbone without padded attention tokens."""

    def __init__(self, arm: str, hidden: int = 256, layers: int = 5,
                 heads: int = 8, slices: int = 32):
        super().__init__()
        if arm == "transolver":
            root = str(BASE_ROOT)
            if root not in sys.path:
                sys.path.insert(0, root)
            official = _load_official_module(
                "_v3_official_transolver",
                BASE_ROOT / "model" / "Transolver_Irregular_Mesh.py",
            )
            self.backbone = official.Model(
                space_dim=140, fun_dim=0, out_dim=2, n_layers=layers,
                n_hidden=hidden, n_head=heads, slice_num=slices,
                mlp_ratio=1, dropout=0.0, unified_pos=False,
            )
        elif arm == "transolver-plus":
            official = _load_official_module("_v3_official_transolver_plus", PLUS_SOURCE)
            # The authors' implementation calls distributed all_reduce even in
            # its single-GPU path. With world_size=1 this is exactly identity.
            official.dist_nn = _SingleProcessDistributed
            self.backbone = official.Model(
                space_dim=140, fun_dim=0, out_dim=2, n_layers=layers,
                n_hidden=hidden, n_head=heads, slice_num=slices,
                mlp_ratio=1, dropout=0.0, unified_pos=False,
            )
        else:
            raise ValueError(arm)
        self.arm = arm

    def forward(self, graph):
        counts = graph.sample_node_counts.tolist()
        if any(count % NODES_PER_OBJECT or count <= 0 for count in counts):
            raise ValueError("each scene must have complete canonical spheres")
        scenes = torch.split(graph.node_features, counts, dim=0)
        predictions = [None] * len(scenes)
        for count in sorted(set(counts)):
            indices = [index for index, size in enumerate(counts) if size == count]
            batch = torch.stack([scenes[index] for index in indices], dim=0)
            if self.arm == "transolver":
                prediction = self.backbone(batch, None)
            else:
                prediction = self.backbone((batch, None, None))
            for index, item in zip(indices, prediction.unbind(dim=0)):
                predictions[index] = item
        return torch.cat(predictions, dim=0)
