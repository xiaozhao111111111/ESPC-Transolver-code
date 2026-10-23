"""Object-aware residual adapters for the unchanged S(Z)-Transolver.

Both adapters have identical parameters. Only cross-object normalization differs.
No physical Green operator or additional geometry descriptor is introduced.
"""
from __future__ import annotations

import torch
from torch import nn

from cspgnet.v3_transolver_comparison import GroupedTransolver

ARMS = ("baseline", "object-joint", "object-sum")


class ObjectSliceInteraction(nn.Module):
    def __init__(self, mode: str, hidden=256, heads=8, slices=8):
        super().__init__()
        if mode not in ("object-joint", "object-sum"):
            raise ValueError(mode)
        self.mode, self.heads, self.slices = mode, heads, slices
        self.width = hidden // heads
        self.norm = nn.LayerNorm(hidden)
        self.x_proj = nn.Linear(hidden, hidden)
        self.v_proj = nn.Linear(hidden, hidden)
        self.slice_proj = nn.Linear(self.width, slices)
        self.temperature = nn.Parameter(torch.full((1, heads, 1, 1, 1), 0.5))
        self.q = nn.Linear(self.width, self.width, bias=False)
        self.k = nn.Linear(self.width, self.width, bias=False)
        self.v = nn.Linear(self.width, self.width, bias=False)
        # Slice-pair-specific relative geometry: delta, distance, unit direction,
        # sin(k*r), cos(k*r). A constant pair bias would cancel in pair softmax.
        self.edge_mlp = nn.Sequential(nn.Linear(9, 32), nn.SiLU(), nn.Linear(32, 2))
        self.out = nn.Linear(hidden, hidden, bias=False)
        self.alpha = nn.Parameter(torch.zeros(()))
        self.apply(self._init)
        nn.init.orthogonal_(self.slice_proj.weight)

    @staticmethod
    def _init(module):
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, hidden, raw):
        batch, points, channels = hidden.shape
        objects = points // 128
        if points % 128:
            raise ValueError("partial canonical object")
        if objects == 1:
            return torch.zeros_like(hidden)
        x = self.norm(hidden).reshape(batch, objects, 128, channels)
        xp = self.x_proj(x).reshape(batch, objects, 128, self.heads, self.width)
        vp = self.v_proj(x).reshape(batch, objects, 128, self.heads, self.width)
        xp, vp = xp.permute(0, 3, 1, 2, 4), vp.permute(0, 3, 1, 2, 4)
        weights = (self.slice_proj(xp) / self.temperature.clamp(0.1, 5)).softmax(-1)
        mass = weights.sum(-2).clamp_min(1e-5)
        tokens = torch.einsum("bhnrd,bhnrs->bhnsd", vp, weights) / mass[..., None]
        local = raw.reshape(batch, objects, 128, 140)[..., 130:133]
        centers = raw.reshape(batch, objects, 128, 140)[:, :, 0, 133:136]
        positions = torch.einsum("bnrc,bhnrs->bhnsc", local, weights) / mass[..., None]
        positions = positions + centers[:, None, :, None, :]
        delta = positions[:, :, :, None, :, None, :] - positions[:, :, None, :, None, :, :]
        distance = torch.linalg.vector_norm(delta, dim=-1, keepdim=True).clamp_min(1e-6)
        edge = torch.cat((delta, distance, delta / distance,
                          torch.sin(3.0 * distance), torch.cos(3.0 * distance)), dim=-1)
        edge_code = self.edge_mlp(edge)
        query, key, value = self.q(tokens), self.k(tokens), self.v(tokens)
        logits = torch.einsum("bhisd,bhjtd->bhijst", query, key) * self.width**-0.5
        logits = logits + edge_code[..., 0]
        source_gate = 1.0 + torch.tanh(edge_code[..., 1])
        self_mask = torch.eye(objects, dtype=torch.bool, device=hidden.device)[None, None, :, :, None, None]
        if self.mode == "object-joint":
            logits = logits.masked_fill(self_mask, -torch.inf)
            ordered = logits.permute(0, 1, 2, 4, 3, 5)
            attention = ordered.flatten(-2).softmax(-1).reshape_as(ordered).permute(0, 1, 2, 4, 3, 5)
        else:
            attention = logits.softmax(-1).masked_fill(self_mask, 0.0)
        messages = torch.einsum("bhijst,bhjtd->bhisd", attention * source_gate, value)
        nodes = torch.einsum("bhnsd,bhnrs->bhnrd", messages, weights)
        nodes = nodes.permute(0, 2, 3, 1, 4).reshape(batch, points, channels)
        return self.alpha * self.out(nodes)


class ObjectAwareTransolver(nn.Module):
    def __init__(self, arm: str):
        super().__init__()
        if arm not in ARMS:
            raise ValueError(arm)
        self.arm = arm
        self.backbone = GroupedTransolver("transolver").backbone
        self.object_branch = None if arm == "baseline" else ObjectSliceInteraction(arm)

    def load_parent(self, state):
        result = self.load_state_dict(state, strict=False)
        if result.unexpected_keys or any(not key.startswith("object_branch.") for key in result.missing_keys):
            raise RuntimeError(f"parent weights incompatible: {result}")
        if self.arm == "baseline" and result.missing_keys:
            raise RuntimeError("baseline did not load exactly")

    def _forward_scene_batch(self, raw):
        hidden = self.backbone.preprocess(raw) + self.backbone.placeholder[None, None, :]
        for index, block in enumerate(self.backbone.blocks):
            hidden = block(hidden)
            if index == 2 and self.object_branch is not None:
                hidden = hidden + self.object_branch(hidden, raw)
        return hidden

    def forward(self, graph):
        counts = graph.sample_node_counts.tolist()
        if any(count <= 0 or count % 128 for count in counts):
            raise ValueError("invalid scene object layout")
        scenes = torch.split(graph.node_features, counts, dim=0)
        predictions = [None] * len(scenes)
        for count in sorted(set(counts)):
            indices = [index for index, size in enumerate(counts) if size == count]
            prediction = self._forward_scene_batch(torch.stack([scenes[index] for index in indices]))
            for index, value in zip(indices, prediction.unbind(0)):
                predictions[index] = value
        return torch.cat(predictions)
