"""GPU edge-index backend for the official PIBNet.

The trainable modules are taken from the official ``PIBNet`` instance.  This
module only replaces DGL's edge gather/scatter operations with PyTorch tensor
indexing and ``index_add``.  The message equations, residual connections,
MLP weights, hierarchy, and output head therefore remain the same."""

from __future__ import annotations

from dataclasses import dataclass

from typing import Dict, Optional

import torch

from torch import Tensor, nn

from gnn.pibnet import AggNodeBlock, EdgeBlock, NodeBlock, NodeDimensionExpander, PIBNet

@dataclass
class EdgeIndexGraphBatch:
    node_features: Tensor
    labels: Tensor
    incident: Tensor
    q64_gt: Tensor
    z_real: Tensor
    z_imag: Tensor
    batch_index: Tensor
    sample_index: Tensor
    sample_node_counts: Tensor
    edge_features: Dict[str, Tensor]
    edge_index: Dict[str, Tensor]
    level_masks: Dict[int, Tensor]
    geo_local: Optional[Tensor] = None
    geo_global: Optional[Tensor] = None

    def to(self, device: torch.device | str):
        return EdgeIndexGraphBatch(
            node_features=self.node_features.to(device),
            labels=self.labels.to(device),
            incident=self.incident.to(device),
            q64_gt=self.q64_gt.to(device),
            z_real=self.z_real.to(device),
            z_imag=self.z_imag.to(device),
            batch_index=self.batch_index.to(device),
            sample_index=self.sample_index.to(device),
            sample_node_counts=self.sample_node_counts.to(device),
            edge_features={key: value.to(device) for key, value in self.edge_features.items()},
            edge_index={key: value.to(device) for key, value in self.edge_index.items()},
            level_masks={key: value.to(device) for key, value in self.level_masks.items()},
            geo_local=None if self.geo_local is None else self.geo_local.to(device),
            geo_global=None if self.geo_global is None else self.geo_global.to(device),
        )

class PIBNetEdgeIndex(nn.Module):
    """Official PIBNet modules with DGL message passing replaced by edge-index ops."""

    def __init__(self, core: PIBNet):
        super().__init__()
        self.core = core

    def _edge_update(
        self,
        layer: EdgeBlock,
        edge_features: Dict[str, Tensor],
        node_features: Dict[str, Tensor],
        graph: EdgeIndexGraphBatch,
        edge_index: Optional[int] = None,
    ):
        etype = layer.etype
        if edge_index is not None:
            etype = f"{etype}_{edge_index}"
        src, dst = graph.edge_index[etype]
        edge_input = edge_features[etype]
        node_input = node_features[str(layer.level)]

        # Official PIBNet uses MeshGraphEdgeMLPConcat because do_concat_trick
        # is false in the paper configuration.  DGL's concat_efeat order is
        # exactly [edge, source node, destination node].
        edge_mlp = layer.edge_mlp
        edge_cat = torch.cat((edge_input, node_input[src], node_input[dst]), dim=-1)
        edge_features[etype] = edge_mlp.model(edge_cat) + edge_input

    def _node_update(
        self,
        layer: NodeBlock,
        edge_features: Dict[str, Tensor],
        node_features: Dict[str, Tensor],
        graph: EdgeIndexGraphBatch,
        edge_index: Optional[int] = None,
    ):
        etypes = layer.etype if isinstance(layer.etype, list) else [layer.etype]
        if edge_index is not None:
            etypes = [f"{etype}_{edge_index}" for etype in etypes]

        destination = node_features[str(layer.level_in)]
        edge_width = edge_features[etypes[0]].shape[-1]
        aggregated = torch.zeros(
            destination.shape[0],
            edge_width,
            device=destination.device,
            dtype=destination.dtype,
        )
        degree = torch.zeros(destination.shape[0], device=destination.device, dtype=destination.dtype)
        for etype in etypes:
            _, dst = graph.edge_index[etype]
            aggregated.index_add_(0, dst, edge_features[etype])
            if layer.aggregation == "mean":
                degree.index_add_(0, dst, torch.ones_like(dst, dtype=destination.dtype))
        if layer.aggregation == "mean":
            aggregated = aggregated / degree.clamp_min(1.0).unsqueeze(-1)

        node_input = torch.cat((aggregated, destination), dim=-1)
        level_out = str(layer.level_out)
        node_features[level_out] = layer.node_mlp(node_input) + node_features[level_out]

    @staticmethod
    def _snapshot(node_features: Dict[str, Tensor], edge_features: Dict[str, Tensor]):
        return {
            "nodes": {key: value.detach().clone() for key, value in node_features.items()},
            "edges": {key: value.detach().clone() for key, value in edge_features.items()},
        }

    def forward(self, graph: EdgeIndexGraphBatch, return_trace: bool = False):
        node_features = {"0": self.core.node_encoder(graph.node_features)}
        edge_features = {}
        for etype in self.core.etypes:
            encoder_key = etype.split("_")[0] if etype.split("_")[0] in self.core.long_range_etypes else etype
            edge_features[etype] = self.core.edge_encoders[encoder_key](graph.edge_features[etype])

        trace = {}
        if return_trace:
            trace["encoded"] = self._snapshot(node_features, edge_features)

        layer_number = 0
        for layer in self.core.processor.in_layers:
            if isinstance(layer, EdgeBlock):
                self._edge_update(layer, edge_features, node_features, graph)
            elif isinstance(layer, NodeBlock):
                self._node_update(layer, edge_features, node_features, graph)
            elif isinstance(layer, NodeDimensionExpander):
                projected = layer.node_mlp(node_features[str(layer.level)])
                node_features[str(layer.level + 1)] = projected
                node_features[f"{layer.level}_proj"] = projected.clone()
            else:
                raise TypeError(f"Unsupported official PIBNet input layer: {type(layer).__name__}")
            if return_trace:
                trace[f"processor_{layer_number}"] = self._snapshot(node_features, edge_features)
            layer_number += 1

        distant_index = 0
        for layer in self.core.processor.bottom_layers:
            edge_sample = distant_index // self.core.processor.distant_edge_sample_period
            if isinstance(layer, EdgeBlock):
                self._edge_update(layer, edge_features, node_features, graph, edge_sample)
            elif isinstance(layer, NodeBlock):
                self._node_update(layer, edge_features, node_features, graph, edge_sample)
                distant_index += 1
            else:
                raise TypeError(f"Unsupported official PIBNet bottom layer: {type(layer).__name__}")
            if return_trace:
                trace[f"processor_{layer_number}"] = self._snapshot(node_features, edge_features)
            layer_number += 1

        for layer in self.core.processor.out_layers:
            if isinstance(layer, AggNodeBlock):
                level = layer.level
                mask = ~(
                    graph.level_masks[level].eq(0) & graph.level_masks[level - 1].eq(1)
                ).unsqueeze(-1)
                node_features[str(level)] = (
                    node_features[str(level)] * mask
                    + node_features[f"{level - 1}_proj"] * ~mask
                )
            elif isinstance(layer, EdgeBlock):
                self._edge_update(layer, edge_features, node_features, graph)
            elif isinstance(layer, NodeBlock):
                self._node_update(layer, edge_features, node_features, graph)
            else:
                raise TypeError(f"Unsupported official PIBNet output layer: {type(layer).__name__}")
            if return_trace:
                trace[f"processor_{layer_number}"] = self._snapshot(node_features, edge_features)
            layer_number += 1

        output = self.core.node_decoder(node_features["0"])
        if return_trace:
            trace["decoded"] = output.detach().clone()
            return output, trace
        return output
