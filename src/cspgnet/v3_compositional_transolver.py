"""Object-local Transolver with relative, additive, shared inter-object updates.

The canonical-node order is fixed within each object. No scene-center coordinate
enters the local encoder or attention. Centers are available only to the shared
pairwise adapter. The five pinned official blocks run independently on each
128-node object; one adapter is reused after blocks 1, 2, and 3.
"""
from __future__ import annotations

import torch
from torch import nn

from cspgnet.v3_object_transolver import ObjectSliceInteraction
from cspgnet.v3_transolver_comparison import GroupedTransolver


class CompositionalTransolver(nn.Module):
    coupling_after = (0, 1, 2)

    def __init__(self):
        super().__init__()
        self.backbone = GroupedTransolver("transolver").backbone
        self.object_branch = ObjectSliceInteraction("object-sum")

    def load_parent(self, state):
        result = self.load_state_dict(state, strict=False)
        if result.unexpected_keys or any(
            not name.startswith("object_branch.") for name in result.missing_keys
        ):
            raise RuntimeError(f"incompatible epoch-100 parent: {result}")
        if not result.missing_keys:
            raise RuntimeError("expected original parent without an object adapter")

    def local_embedding(self, raw):
        """Retain input width/parameters, but physically exclude center channels."""
        if raw.ndim != 3 or raw.shape[-1] != 140 or raw.shape[1] % 128:
            raise ValueError("expected [batch, 128*N, 140] canonical features")
        batch, points, _ = raw.shape
        local = raw.reshape(batch * (points // 128), 128, 140).clone()
        local[..., 133:136] = 0.0
        return self.backbone.preprocess(local) + self.backbone.placeholder[None, None, :]

    def _forward_scene_batch(self, raw):
        batch, points, _ = raw.shape
        objects = points // 128
        hidden = self.local_embedding(raw)
        for index, block in enumerate(self.backbone.blocks):
            # All official slice normalizations are over one object, never N*128.
            hidden = block(hidden)
            if index in self.coupling_after:
                grouped = hidden.reshape(batch, objects * 128, 256)
                grouped = grouped + self.object_branch(grouped, raw)
                hidden = grouped.reshape(batch * objects, 128, 256)
        return hidden.reshape(batch, points, 2)

    def forward(self, graph):
        counts = graph.sample_node_counts.tolist()
        if any(count <= 0 or count % 128 for count in counts):
            raise ValueError("every scene must contain complete canonical objects")
        scenes = torch.split(graph.node_features, counts, dim=0)
        predictions = [None] * len(scenes)
        for count in sorted(set(counts)):
            indices = [index for index, size in enumerate(counts) if size == count]
            prediction = self._forward_scene_batch(torch.stack([scenes[index] for index in indices]))
            for index, value in zip(indices, prediction.unbind(0)):
                predictions[index] = value
        return torch.cat(predictions)
