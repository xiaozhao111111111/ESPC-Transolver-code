"""Layer-wise DGL versus edge-index validation for official PIBNet."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

PRIMARY_SITE = r"C:\Users\27159\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\Lib\site-packages"
OFFICIAL_SRC = r"C:\Users\27159\Documents\Codex\2026-09-20\transformer\work\pibnet\src"
OFFICIAL_PIBNET_SRC = r"C:\Users\27159\Documents\Codex\2026-09-20\transformer\work\pibnet\src\pibnet"
OMP_DIR = r"C:\Users\27159\Documents\Codex\2026-09-20\transformer\work\openmp_runtime_20260920\package\runtimes\win-x64\native"

os.environ.setdefault("DGLBACKEND", "pytorch")
os.environ.setdefault("DGL_SKIP_GRAPHBOLT", "1")
if Path(OMP_DIR).exists():
    os.add_dll_directory(OMP_DIR)
sys.path.append(PRIMARY_SITE)
sys.path.insert(0, OFFICIAL_SRC)
sys.path.insert(0, OFFICIAL_PIBNET_SRC)

import dgl
import torch

try:
    _torchvision_compat = torch.library.Library("torchvision", "DEF")
    _torchvision_compat.define("nms(Tensor boxes, Tensor scores, float iou_threshold) -> Tensor")
except (RuntimeError, TypeError):
    _torchvision_compat = None

from cspgnet.official_pibnet_adapter import (
    CANDIDATE_EDGE_RATIO,
    DISTANT_EDGE_RATIO,
    OfficialPIBNetCanonicalDataset,
    collate_graphs,
)
from cspgnet.official_pibnet_edge_backend import PIBNetEdgeIndex, collate_edge_index_graphs
from cspgnet.official_pibnet_train import make_model
from gnn.pibnet import AggNodeBlock, EdgeBlock, NodeBlock, NodeDimensionExpander


def _snapshot(node_features, edge_features):
    return {
        "nodes": {key: value.detach().clone() for key, value in node_features.items()},
        "edges": {key: value.detach().clone() for key, value in edge_features.items()},
    }


@torch.no_grad()
def dgl_trace(core, graph):
    node_features = {"0": core.node_encoder(graph.ndata["node_features"])}
    edge_features = {}
    for etype in core.etypes:
        encoder_key = (
            etype.split("_")[0]
            if etype.split("_")[0] in core.long_range_etypes
            else etype
        )
        edge_features[etype] = core.edge_encoders[encoder_key](
            graph.edges[etype].data["edge_features"]
        )

    trace = {"encoded": _snapshot(node_features, edge_features)}
    layer_number = 0
    for layer in core.processor.in_layers:
        edge_features, node_features = layer(edge_features, node_features, graph)
        trace[f"processor_{layer_number}"] = _snapshot(node_features, edge_features)
        layer_number += 1

    edge_index = 0
    for layer in core.processor.bottom_layers:
        edge_features, node_features = layer(
            edge_features,
            node_features,
            graph,
            edge_index=edge_index // core.processor.distant_edge_sample_period,
        )
        if isinstance(layer, NodeBlock):
            edge_index += 1
        trace[f"processor_{layer_number}"] = _snapshot(node_features, edge_features)
        layer_number += 1

    for layer in core.processor.out_layers:
        edge_features, node_features = layer(edge_features, node_features, graph)
        trace[f"processor_{layer_number}"] = _snapshot(node_features, edge_features)
        layer_number += 1

    trace["decoded"] = core.node_decoder(node_features["0"]).detach().clone()
    return trace


def compare(dgl_value, edge_value, atol=1.0e-6):
    dgl_value = dgl_value.detach().cpu()
    edge_value = edge_value.detach().cpu()
    difference = (dgl_value - edge_value).abs()
    scale = dgl_value.abs().maximum(edge_value.abs()).clamp_min(1.0e-12)
    return {
        "max_abs": float(difference.max()),
        "max_rel": float((difference / scale).max()),
        "allclose": bool(torch.allclose(dgl_value, edge_value, atol=atol, rtol=1.0e-5)),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        default=r"C:\Users\27159\Documents\Codex\2026-09-20\transformer\work\pibnet\work\cspgnet_paper_neumann_train",
    )
    parser.add_argument("--allow-incomplete", action="store_true")
    parser.add_argument(
        "--report",
        default=r"C:\Users\27159\Documents\Codex\2026-09-20\transformer\outputs\pibnet_edge_backend_validation.json",
    )
    args = parser.parse_args()
    torch.manual_seed(1234)
    dataset = OfficialPIBNetCanonicalDataset(
        args.data, require_complete=not args.allow_incomplete
    )
    node_dim = dataset.node_feature_dim
    sample = dataset[0]
    dgl_graph = collate_graphs([sample])
    edge_graph = collate_edge_index_graphs([sample])

    cpu_core = make_model(torch.device("cpu"), node_dim=node_dim).eval()
    edge_cpu = PIBNetEdgeIndex(cpu_core).eval()
    with torch.no_grad():
        dgl_result = dgl_trace(cpu_core, dgl_graph)
        _, edge_result = edge_cpu(edge_graph, return_trace=True)

    structural_failures = []
    structure = {
        "tree_transform_depth": 2,
        "distant_edge_ratio": DISTANT_EDGE_RATIO,
        "candidate_edge_ratio": CANDIDATE_EDGE_RATIO,
        "relations": {},
    }
    level_counts = [int(sample.ndata[f"lvl_{level}"].sum()) for level in range(3)]
    object_count = sample.num_nodes() // 128
    expected_levels = [128 * object_count, 32 * object_count, 8 * object_count]
    structure["level_node_counts"] = level_counts
    structure["expected_level_node_counts"] = expected_levels
    if level_counts != expected_levels:
        structural_failures.append(
            f"hierarchy counts {level_counts} != expected {expected_levels}"
        )
    top_mask = sample.ndata["lvl_2"]
    for etype in sample.etypes:
        source, destination = sample.edges(etype=etype)
        keys = source.to(torch.int64) * sample.num_nodes() + destination.to(torch.int64)
        self_loops = int((source == destination).sum())
        duplicates = int(source.numel() - torch.unique(keys).numel())
        structure["relations"][etype] = {
            "edges": int(source.numel()),
            "self_loops": self_loops,
            "duplicates": duplicates,
        }
        if self_loops or duplicates:
            structural_failures.append(
                f"{etype}: self_loops={self_loops}, duplicates={duplicates}"
            )
        if etype == "distant_0" and source.numel() and not bool(
            torch.all(top_mask[source] & top_mask[destination])
        ):
            structural_failures.append("distant_0 has endpoints outside level 2")

    num_top = expected_levels[2]
    proposals_per_source = DISTANT_EDGE_RATIO * num_top
    if proposals_per_source < 1.0:
        forward_proposals = int(num_top * proposals_per_source)
    else:
        forward_proposals = num_top * int(round(proposals_per_source))
    maximum_distant_edges = 2 * forward_proposals
    structure["maximum_distant_edges_before_dedup"] = maximum_distant_edges
    if structure["relations"]["distant_0"]["edges"] > maximum_distant_edges:
        structural_failures.append("distant_0 exceeds the paper sparse-edge budget")

    report = {
        "structure": structure,
        "structural_failures": structural_failures,
        "cpu": {},
    }
    for key in dgl_result:
        if key == "decoded":
            report["cpu"][key] = compare(dgl_result[key], edge_result[key])
            continue
        report["cpu"][key] = {
            "nodes": {
                name: compare(dgl_result[key]["nodes"][name], edge_result[key]["nodes"][name])
                for name in dgl_result[key]["nodes"]
            },
            "edges": {
                name: compare(dgl_result[key]["edges"][name], edge_result[key]["edges"][name])
                for name in dgl_result[key]["edges"]
            },
        }

    if torch.cuda.is_available():
        cuda = torch.device("cuda")
        cuda_core = make_model(cuda, node_dim=node_dim).eval()
        cuda_core.load_state_dict(cpu_core.state_dict())
        edge_cuda = PIBNetEdgeIndex(cuda_core).eval()
        with torch.no_grad():
            _, edge_cuda_result = edge_cuda(edge_graph.to(cuda), return_trace=True)
        report["cuda_edge_vs_cpu_dgl"] = {}
        for key in dgl_result:
            if key == "decoded":
                report["cuda_edge_vs_cpu_dgl"][key] = compare(
                    dgl_result[key], edge_cuda_result[key], atol=2.0e-5
                )
                continue
            report["cuda_edge_vs_cpu_dgl"][key] = {
                "nodes": {
                    name: compare(
                        dgl_result[key]["nodes"][name],
                        edge_cuda_result[key]["nodes"][name],
                        atol=2.0e-5,
                    )
                    for name in dgl_result[key]["nodes"]
                },
                "edges": {
                    name: compare(
                        dgl_result[key]["edges"][name],
                        edge_cuda_result[key]["edges"][name],
                        atol=2.0e-5,
                    )
                    for name in dgl_result[key]["edges"]
                },
            }

    import json

    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    def leaves(value):
        if isinstance(value, dict):
            if "max_abs" in value and "allclose" in value:
                yield value
            else:
                for child in value.values():
                    yield from leaves(child)

    summary = {}
    for name, value in report.items():
        if name in {"structure", "structural_failures"}:
            continue
        items = list(leaves(value))
        summary[name] = {
            "comparisons": len(items),
            "max_abs": max(item["max_abs"] for item in items),
            "max_rel": max(item["max_rel"] for item in items),
            "allclose": all(item["allclose"] for item in items),
        }
    passed = (
        node_dim == 140
        and not structural_failures
        and summary["cpu"]["allclose"]
        and summary["cpu"]["max_abs"] <= 1.0e-6
        and (
            "cuda_edge_vs_cpu_dgl" not in summary
            or (
                summary["cuda_edge_vs_cpu_dgl"]["allclose"]
                and summary["cuda_edge_vs_cpu_dgl"]["max_abs"] <= 2.0e-5
            )
        )
    )
    report["passed"] = passed
    report["summary"] = summary
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"passed": passed, "node_feature_dim": node_dim, "summary": summary, "report": str(report_path)}, indent=2, ensure_ascii=False))
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
