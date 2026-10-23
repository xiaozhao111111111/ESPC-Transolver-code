"""Two-level surface/object graph model for the CSP-GNet prototype."""

from __future__ import annotations

import torch
from torch import nn


class SurfaceMessageLayer(nn.Module):
    def __init__(self, hidden_dim: int, neighbors: torch.Tensor):
        super().__init__()
        self.register_buffer("neighbors", neighbors.long(), persistent=False)
        self.edge_mlp = nn.Sequential(
            nn.Linear(3 * hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.node_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, objects, surface_nodes, hidden]
        neigh = x[:, :, self.neighbors, :]
        center = x.unsqueeze(3).expand_as(neigh)
        message = self.edge_mlp(torch.cat([center, neigh, neigh - center], dim=-1))
        aggregate = message.mean(dim=3)
        return self.norm(x + self.node_mlp(torch.cat([x, aggregate], dim=-1)))


class ObjectMessageLayer(nn.Module):
    def __init__(self, hidden_dim: int, edge_dim: int):
        super().__init__()
        self.message_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim + edge_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.update_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor, centers: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        # x: [batch, objects, hidden], centers: [batch, objects, 3]
        delta = centers.unsqueeze(1) - centers.unsqueeze(2)
        distance = delta.norm(dim=-1, keepdim=True).clamp_min(1.0e-6)
        direction = delta / distance
        kr = k[:, None, None, None] * distance
        green = torch.exp(1j * kr.to(torch.complex64)) / (4.0 * torch.pi * distance)
        edge = torch.cat(
            [delta, direction, kr, torch.cos(kr), torch.sin(kr), green.real, green.imag],
            dim=-1,
        )
        receiver = x.unsqueeze(2).expand(-1, -1, x.shape[1], -1)
        sender = x.unsqueeze(1).expand(-1, x.shape[1], -1, -1)
        messages = self.message_mlp(torch.cat([receiver, sender, edge], dim=-1))
        objects = x.shape[1]
        mask = (~torch.eye(objects, device=x.device, dtype=torch.bool))[None, :, :, None]
        aggregate = (messages * mask).sum(dim=2) / max(objects - 1, 1)
        return self.norm(x + self.update_mlp(torch.cat([x, aggregate], dim=-1)))


class CSPGNet(nn.Module):
    """Surface encoder -> object graph -> canonical-surface decoder."""

    def __init__(
        self,
        interface_points: torch.Tensor,
        hidden_surface: int = 64,
        hidden_object: int = 128,
        surface_layers: int = 3,
        object_layers: int = 6,
        surface_neighbors: int = 6,
    ):
        super().__init__()
        self.register_buffer("interface_points", interface_points.float(), persistent=False)
        self.register_buffer("surface_normals", interface_points.float() / interface_points.norm(dim=-1, keepdim=True), persistent=False)
        distances = torch.cdist(interface_points.float(), interface_points.float())
        neighbors = distances.topk(surface_neighbors + 1, largest=False).indices[:, 1:]

        self.surface_input = nn.Sequential(
            nn.Linear(41, hidden_surface),
            nn.SiLU(),
            nn.Linear(hidden_surface, hidden_surface),
        )
        self.surface_layers = nn.ModuleList(
            [SurfaceMessageLayer(hidden_surface, neighbors) for _ in range(surface_layers)]
        )
        self.attention = nn.Linear(hidden_surface, 1)
        self.value = nn.Linear(hidden_surface, hidden_object)

        # delta(3), direction(3), kr/cos/sin(3), Re/Im Green(2)
        self.object_input = nn.Sequential(
            nn.Linear(hidden_object + 3 + 1 + 2, hidden_object),
            nn.SiLU(),
            nn.Linear(hidden_object, hidden_object),
        )
        self.object_layers = nn.ModuleList(
            [ObjectMessageLayer(hidden_object, edge_dim=11) for _ in range(object_layers)]
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_object + hidden_surface + 41, hidden_surface),
            nn.SiLU(),
            nn.Linear(hidden_surface, hidden_surface),
            nn.SiLU(),
            nn.Linear(hidden_surface, 2),
        )

    def forward(
        self,
        physics: torch.Tensor,
        incident: torch.Tensor,
        centers: torch.Tensor,
        k: torch.Tensor,
    ) -> torch.Tensor:
        batch, objects, surface_nodes, _ = physics.shape
        points = self.interface_points[None, None, :, :].expand(batch, objects, -1, -1)
        normals = self.surface_normals[None, None, :, :].expand(batch, objects, -1, -1)
        k_surface = k[:, None, None, None].expand(batch, objects, surface_nodes, 1)
        surface_input = torch.cat([physics, incident, points, normals, k_surface], dim=-1)
        surface_state = self.surface_input(surface_input)
        for layer in self.surface_layers:
            surface_state = layer(surface_state)

        weights = torch.softmax(self.attention(surface_state), dim=2)
        object_token = (weights * self.value(surface_state)).sum(dim=2)
        incident_summary = incident.mean(dim=2)
        object_input = torch.cat([object_token, centers, k[:, None, None].expand(batch, objects, 1), incident_summary], dim=-1)
        object_state = self.object_input(object_input)
        for layer in self.object_layers:
            object_state = layer(object_state, centers, k)

        object_state_surface = object_state[:, :, None, :].expand(-1, -1, surface_nodes, -1)
        decoder_input = torch.cat([object_state_surface, surface_state, surface_input], dim=-1)
        return self.decoder(decoder_input)
