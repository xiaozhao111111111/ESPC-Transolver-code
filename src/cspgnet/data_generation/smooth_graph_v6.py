"""Deterministic smooth thickened-graph geometries for the v6 pilot.

The pilot uses eight hand-stratified graph topologies (two per genus 0--3).
The realized CAD boundary, not the abstract graph cycle rank, is the source
of truth for genus. These are geometry objects only, never training samples.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math

import numpy as np
from scipy.spatial.transform import Rotation


VERSION = "smooth-thickened-graph-v6-pilot2-tangent-junctions"
GENUS_QUOTAS = {0: 2, 1: 2, 2: 2, 3: 2}

# Node coordinates are dimensionless; every structure receives a deterministic
# rotation, small non-rigid perturbation, anisotropic scale and tube radius.
_TEMPLATES = {
    "g0-path4": (0, [(-.42, 0, 0), (-.14, .16, .03), (.15, -.12, -.05), (.42, .04, .05)],
                 [(0, 1), (1, 2), (2, 3)]),
    "g0-star4": (0, [(0, 0, 0), (.42, .42, .42), (.42, -.42, -.42),
                     (-.42, .42, -.42), (-.42, -.42, .42)],
                 [(0, 1), (0, 2), (0, 3), (0, 4)]),
    "g1-cycle4": (1, [(-.34, -.34, 0), (.34, -.34, 0), (.34, .34, 0), (-.34, .34, 0)],
                  [(0, 1), (1, 2), (2, 3), (3, 0)]),
    "g1-cycle5-tail": (1, [(.36, 0, 0), (.11, .34, 0), (-.29, .21, 0),
                           (-.29, -.21, 0), (.11, -.34, 0), (.56, .14, .18)],
                       [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0), (0, 5)]),
    "g2-square-diagonal": (2, [(-.34, -.34, 0), (.34, -.34, 0), (.34, .34, 0), (-.34, .34, 0)],
                           [(0, 1), (1, 2), (2, 3), (3, 0), (0, 2)]),
    "g2-figure-eight": (2, [(0, 0, 0), (.42, 0, 0), (.20, .32, 0),
                            (0, 0, .42), (0, .32, .20)],
                        [(0, 1), (1, 2), (2, 0), (0, 3), (3, 4), (4, 0)]),
    "g3-k4": (3, [(.39, .39, .39), (.39, -.39, -.39),
                  (-.39, .39, -.39), (-.39, -.39, .39)],
              [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]),
    "g3-three-loops": (3, [(0, 0, 0), (.40, 0, 0), (.20, .34, 0),
                           (-.34, .20, 0), (-.40, 0, .02),
                           (.02, -.36, .20), (.02, -.19, .40)],
                       [(0, 1), (1, 2), (2, 0), (0, 3), (3, 4), (4, 0),
                        (0, 5), (5, 6), (6, 0)]),
}


def _seed(identifier: str, attempt: int) -> int:
    payload = json.dumps([VERSION, "pilot8", identifier, attempt], separators=(",", ":"))
    return int.from_bytes(hashlib.sha256(payload.encode()).digest()[:8], "little")


@dataclass(frozen=True)
class SmoothGraphSpec:
    identifier: str
    template: str
    genus: int
    nodes: tuple[tuple[float, float, float], ...]
    edges: tuple[tuple[int, int], ...]
    tube_radius: float
    junction_radius_ratio: float
    fillet_radius: float
    generation_seed: int
    generation_attempt: int
    structure_sha256: str

    def as_dict(self) -> dict:
        return asdict(self)

    @property
    def bounding_radius_bound(self) -> float:
        xyz = np.asarray(self.nodes, dtype=np.float64)
        center = xyz.mean(axis=0)
        return float(np.max(np.linalg.norm(xyz - center, axis=1))
                     + self.junction_radius_ratio * self.tube_radius
                     + self.fillet_radius)


def _segment_distance(a0, a1, b0, b1) -> float:
    """Distance between two finite 3-D segments (Ericson closest points)."""
    u, v, w = a1 - a0, b1 - b0, a0 - b0
    aa, bb, cc = float(u @ u), float(u @ v), float(v @ v)
    dd, ee = float(u @ w), float(v @ w)
    den = aa * cc - bb * bb
    s_num, s_den = (0.0, den) if den > 1e-15 else (0.0, 1.0)
    t_num, t_den = (0.0, den) if den > 1e-15 else (ee, cc)
    if den > 1e-15:
        s_num, t_num = bb * ee - cc * dd, aa * ee - bb * dd
        if s_num < 0:
            s_num, t_num, t_den = 0.0, ee, cc
        elif s_num > s_den:
            s_num, t_num, t_den = s_den, ee + bb, cc
    if t_num < 0:
        t_num = 0.0
        if -dd < 0:
            s_num = 0.0
        elif -dd > aa:
            s_num = s_den
        else:
            s_num, s_den = -dd, aa
    elif t_num > t_den:
        t_num = t_den
        if -dd + bb < 0:
            s_num = 0.0
        elif -dd + bb > aa:
            s_num = s_den
        else:
            s_num, s_den = -dd + bb, aa
    sc = 0.0 if abs(s_num) < 1e-15 else s_num / s_den
    tc = 0.0 if abs(t_num) < 1e-15 else t_num / t_den
    return float(np.linalg.norm(w + sc * u - tc * v))


def _embedding_is_separated(nodes: np.ndarray, edges: tuple[tuple[int, int], ...],
                            tube_radius: float) -> bool:
    junction = 1.0 * tube_radius
    clearance = 0.025
    n = len(nodes)
    for i in range(n):
        for j in range(i + 1, n):
            if np.linalg.norm(nodes[i] - nodes[j]) < 2 * junction + clearance:
                return False
    edge_set = {tuple(sorted(edge)) for edge in edges}
    for eidx, (a, b) in enumerate(edges):
        for node in range(n):
            if node in (a, b):
                continue
            if _segment_distance(nodes[a], nodes[b], nodes[node], nodes[node]) < junction + tube_radius + clearance:
                return False
        for c, d in edges[eidx + 1:]:
            if len({a, b, c, d}) < 4:
                continue
            if _segment_distance(nodes[a], nodes[b], nodes[c], nodes[d]) < 2 * tube_radius + clearance:
                return False
    # Every abstract graph cycle is embedded with its intended edge set; this
    # also rejects duplicate/reversed edges before any CAD operation.
    return len(edge_set) == len(edges)


def generate_pilot8() -> tuple[list[SmoothGraphSpec], list[dict]]:
    specs, rejected = [], []
    ordered = sorted(_TEMPLATES.items(), key=lambda item: (item[1][0], item[0]))
    for template_name, (genus, base_nodes, edges) in ordered:
        identifier = f"pilot-g{genus}-{template_name}"
        accepted = None
        for attempt in range(128):
            seed = _seed(identifier, attempt)
            rng = np.random.default_rng(seed)
            base = np.asarray(base_nodes, dtype=np.float64)
            # A bounded perturbation changes the embedding while preserving the
            # graph; rotate and anisotropically deform for structural diversity.
            jitter = rng.normal(0.0, 0.012, size=base.shape)
            scale = rng.uniform(0.88, 1.12, size=3)
            rotation = Rotation.random(random_state=rng).as_matrix()
            nodes = (base + jitter) * scale
            nodes = nodes @ rotation.T
            radius = float(rng.uniform(0.092, 0.108))
            if not _embedding_is_separated(nodes, tuple(edges), radius):
                rejected.append({"id": identifier, "attempt": attempt,
                                 "seed": seed, "reason": "embedding-clearance"})
                continue
            center = nodes.mean(axis=0)
            nodes = nodes - center
            external_bound = float(np.max(np.linalg.norm(nodes, axis=1)) + radius)
            if external_bound >= 0.9:
                rejected.append({"id": identifier, "attempt": attempt,
                                 "seed": seed, "reason": "bounding-radius"})
                continue
            body = {
                "version": VERSION, "id": identifier, "template": template_name,
                "genus": genus, "nodes": np.round(nodes, 12).tolist(),
                "edges": [list(edge) for edge in edges], "tube_radius": radius,
                "junction_radius_ratio": 1.0, "fillet_radius": 0.0,
                "generation_seed": seed, "generation_attempt": attempt,
            }
            token = hashlib.sha256(json.dumps(body, sort_keys=True,
                                               separators=(",", ":")).encode()).hexdigest()
            accepted = SmoothGraphSpec(
                identifier, template_name, genus,
                tuple(tuple(map(float, row)) for row in nodes),
                tuple(tuple(map(int, edge)) for edge in edges), radius,
                1.0, 0.0, seed, attempt, token,
            )
            break
        if accepted is None:
            raise RuntimeError(f"embedding rejection budget exhausted: {identifier}")
        accepted_edges = len(accepted.edges)
        cycle_rank = accepted_edges - len(accepted.nodes) + 1
        if cycle_rank != genus:
            raise AssertionError(f"graph cycle rank mismatch for {identifier}")
        specs.append(accepted)
    if len(specs) != 8 or {g: sum(s.genus == g for s in specs) for g in range(4)} != GENUS_QUOTAS:
        raise AssertionError("pilot must contain exactly two structures per genus 0--3")
    return specs, rejected


def make_solid(spec: SmoothGraphSpec):
    """Fuse round junctions and straight tubes, then fillet every CAD edge."""
    from netgen.occ import Cylinder, Fuse, Pnt, Sphere, gp_Dir

    nodes = np.asarray(spec.nodes, dtype=np.float64)
    junction_r = spec.junction_radius_ratio * spec.tube_radius
    pieces = [Sphere(tuple(node), junction_r) for node in nodes]
    for first, second in spec.edges:
        start, end = nodes[first], nodes[second]
        direction = end - start
        length = float(np.linalg.norm(direction))
        pieces.append(Cylinder(Pnt(*map(float, start)), gp_Dir(*map(float, direction)),
                               spec.tube_radius, length))
    solid = Fuse(pieces)
    if len(solid.solids) != 1 or float(solid.mass) <= 0:
        raise RuntimeError("fused graph tubes are not one positive-volume solid")
    # A tube with the same radius as its spherical junction meets tangentially:
    # the union is G1 at the joint and has no sharp seam. Global OCC filleting
    # made tiny sliver faces in pilot1 and caused overlapping surface meshes.
    smooth = (solid.MakeFillet(solid.edges, spec.fillet_radius)
              if spec.fillet_radius > 0 else solid)
    if len(smooth.solids) != 1 or float(smooth.mass) <= 0:
        raise RuntimeError("filleted graph tubes are not one positive-volume solid")
    return smooth


def spec_to_dict(spec: SmoothGraphSpec) -> dict:
    return spec.as_dict()
