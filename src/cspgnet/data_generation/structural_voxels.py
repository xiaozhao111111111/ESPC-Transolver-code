"""Random connected 3-D block structures, not resized primitive families.

Only design descriptions are produced here.  Numerical release requires a
separate graded-FEM/BEM gate.  Structural equivalence removes translation,
lattice rotations, reflection, and uniform size changes; genus may repeat.
"""

from __future__ import annotations

import hashlib
import itertools
import json
from collections import Counter
from dataclasses import asdict, dataclass

import networkx as nx
import numpy as np

VERSION = "structural-voxel-pool-v1"
FACE_OFFSETS = (
    ((1, 0, 0), ((1, -1, -1), (1, 1, -1), (1, 1, 1), (1, -1, 1))),
    ((-1, 0, 0), ((-1, -1, -1), (-1, -1, 1), (-1, 1, 1), (-1, 1, -1))),
    ((0, 1, 0), ((-1, 1, -1), (-1, 1, 1), (1, 1, 1), (1, 1, -1))),
    ((0, -1, 0), ((-1, -1, -1), (1, -1, -1), (1, -1, 1), (-1, -1, 1))),
    ((0, 0, 1), ((-1, -1, 1), (1, -1, 1), (1, 1, 1), (-1, 1, 1))),
    ((0, 0, -1), ((-1, -1, -1), (-1, 1, -1), (1, 1, -1), (1, -1, -1))),
)
NEIGHBORS = tuple(item[0] for item in FACE_OFFSETS)


def seed_for(component: str, index: int, base_seed: int = 20261005) -> int:
    material = json.dumps([VERSION, base_seed, component, index], separators=(",", ":"))
    return int.from_bytes(hashlib.sha256(material.encode()).digest()[:8], "little")


def canonical_cells(cells) -> tuple[tuple[int, int, int], ...]:
    xyz = np.asarray(sorted(cells), dtype=np.int64)
    candidates = []
    for permutation in itertools.permutations(range(3)):
        for signs in itertools.product((-1, 1), repeat=3):
            transformed = xyz[:, permutation] * np.asarray(signs)
            transformed -= transformed.min(axis=0)
            candidates.append(tuple(sorted(tuple(map(int, row)) for row in transformed)))
    return min(candidates)


def structural_hash(cells) -> str:
    canonical = canonical_cells(cells)
    return hashlib.sha256(json.dumps(canonical, separators=(",", ":")).encode()).hexdigest()


def cell_graph(cells) -> nx.Graph:
    occupied = set(map(tuple, cells))
    graph = nx.Graph()
    graph.add_nodes_from(occupied)
    for cell in occupied:
        for offset in NEIGHBORS:
            neighbor = tuple(cell[axis] + offset[axis] for axis in range(3))
            if neighbor in occupied:
                graph.add_edge(cell, neighbor)
    return graph


def lattice_boundary(cells) -> tuple[np.ndarray, np.ndarray]:
    """Exact exterior faces of the block union, with outward orientation."""
    occupied = set(map(tuple, cells))
    vertex_index = {}
    triangles = []
    for cell in sorted(occupied):
        for offset, corners in FACE_OFFSETS:
            neighbor = tuple(cell[axis] + offset[axis] for axis in range(3))
            if neighbor in occupied:
                continue
            face = []
            for corner in corners:
                key = tuple(2 * cell[axis] + corner[axis] for axis in range(3))
                if key not in vertex_index:
                    vertex_index[key] = len(vertex_index)
                face.append(vertex_index[key])
            triangles.extend(((face[0], face[1], face[2]), (face[0], face[2], face[3])))
    return np.asarray(list(vertex_index), dtype=np.float64) / 2, np.asarray(triangles, dtype=np.int64)


def audit_surface(vertices: np.ndarray, triangles: np.ndarray) -> dict:
    """Reject edge/vertex contacts, cavities and non-manifold boundaries."""
    from .ood_geometry_audit import _surface_topology

    euler, watertight, oriented, count = _surface_topology(triangles)
    face_graph = nx.Graph()
    face_graph.add_nodes_from(range(len(triangles)))
    edge_faces = {}
    links = {}
    for face_index, face in enumerate(triangles):
        for first, second in ((face[0], face[1]), (face[1], face[2]), (face[2], face[0])):
            edge_faces.setdefault(tuple(sorted((int(first), int(second)))), []).append(face_index)
        for index, vertex in enumerate(face):
            links.setdefault(int(vertex), []).append((int(face[(index + 1) % 3]), int(face[(index + 2) % 3])))
    for faces in edge_faces.values():
        if len(faces) == 2:
            face_graph.add_edge(*faces)
    components = nx.number_connected_components(face_graph)
    valid_links = all(
        nx.is_connected(link := nx.Graph(edges)) and all(degree == 2 for _, degree in link.degree())
        for edges in links.values()
    )
    genus = (2 - euler) // 2 if components == 1 and euler <= 2 and euler % 2 == 0 else None
    cross = np.cross(vertices[triangles[:, 1]] - vertices[triangles[:, 0]],
                     vertices[triangles[:, 2]] - vertices[triangles[:, 0]])
    volume = float(np.sum(vertices[triangles[:, 0]] * cross) / 6)
    checks = {
        "closed_edges": watertight,
        "consistent_orientation": oriented,
        "single_boundary_component": components == 1,
        "manifold_vertex_links": valid_links,
        "positive_oriented_volume": volume > 0,
        "valid_genus": genus is not None,
    }
    return {"euler_characteristic": euler, "genus": genus,
            "boundary_components": components, "vertices": count,
            "triangles": len(triangles), "oriented_volume": volume,
            "checks": checks, "passed": all(checks.values())}


@dataclass(frozen=True)
class StructuralSpec:
    id: str
    cells: tuple[tuple[int, int, int], ...]
    cell_width: float
    structure_sha256: str
    generation_seed: int
    genus: int

    @property
    def center(self) -> np.ndarray:
        xyz = np.asarray(self.cells)
        return (xyz.min(axis=0) + xyz.max(axis=0)) / 2

    def boundary(self) -> tuple[np.ndarray, np.ndarray]:
        points, faces = lattice_boundary(self.cells)
        return (points - self.center) * self.cell_width, faces

    def contains(self, points: np.ndarray, tolerance: float = 1.0e-10) -> np.ndarray:
        points = np.asarray(points, dtype=np.float64)
        centers = (np.asarray(self.cells) - self.center) * self.cell_width
        result = np.zeros(points.shape[:-1], dtype=bool)
        for center in centers:
            result |= np.all(np.abs(points - center) <= self.cell_width / 2 + tolerance, axis=-1)
        return result


def _candidate(rng: np.random.Generator, genus: int):
    if genus == 0:
        cells = {(0, 0, 0)}
        protected = set()
    elif genus == 1:
        cells = {(x, y, 0) for x in range(-1, 2) for y in range(-1, 2) if (x, y) != (0, 0)}
        protected = {(0, 0, z) for z in range(-2, 3)}
    else:
        cells = {(x, y, 0) for x in range(-2, 3) for y in range(-1, 2) if (x, y) not in ((-1, 0), (1, 0))}
        protected = {(x, 0, z) for x in (-1, 1) for z in range(-2, 3)}
    target = int(rng.integers(max(len(cells), 8), 29))
    while len(cells) < target:
        frontier = sorted({tuple(cell[i] + offset[i] for i in range(3))
                           for cell in cells for offset in NEIGHBORS}
                          - cells - protected)
        frontier = [cell for cell in frontier if max(map(abs, cell)) <= 2]
        if not frontier:
            break
        cells.add(frontier[int(rng.integers(len(frontier)))])
    return cells


def generate_design(count: int = 500, base_seed: int = 20261005) -> dict:
    """Pre-CAD rejection is logged; solver failures never trigger resampling."""
    if count < 3:
        raise ValueError("design requires at least three structures")
    genus_counts = [round(count * .6), round(count * .3)]
    genus_counts.append(count - sum(genus_counts))
    targets = [genus for genus, size in enumerate(genus_counts) for _ in range(size)]
    order_rng = np.random.default_rng(seed_for("genus-order", 0, base_seed))
    order_rng.shuffle(targets)
    entries, rejections, seen = [], [], set()
    for index, genus in enumerate(targets):
        seed = seed_for("structure", index, base_seed)
        rng = np.random.default_rng(seed)
        for attempt in range(10000):
            cells = _candidate(rng, genus)
            points, faces = lattice_boundary(cells)
            topology = audit_surface(points, faces)
            signature = structural_hash(cells)
            reason = None
            if not topology["passed"]:
                reason = "non-manifold-or-disconnected-boundary"
            elif topology["genus"] != genus:
                reason = "genus-differs-from-design-stratum"
            elif signature in seen:
                reason = "rotation-reflection-translation-equivalent-structure"
            elif np.ptp(np.asarray(sorted(cells)), axis=0).min() == 0:
                reason = "no-three-dimensional-branch-layout"
            if reason:
                rejections.append({"index": index, "attempt": attempt, "seed": seed,
                                   "reason": reason, "cells": sorted(cells),
                                   "observed_genus": topology["genus"]})
                continue
            center = (points.min(axis=0) + points.max(axis=0)) / 2
            lattice_radius = float(np.linalg.norm(points - center, axis=1).max())
            width_upper = min(.24, .86 / lattice_radius)
            if width_upper < .18:
                raise RuntimeError("design grid exceeds the permitted resolvable feature range")
            width = float(rng.uniform(.18, width_upper))
            spec = StructuralSpec(f"structural-{index:04d}", tuple(sorted(cells)), width, signature, seed, genus)
            graph = cell_graph(cells)
            actual, _ = spec.boundary()
            entry = {**asdict(spec), "design_attempt": attempt,
                     "topology": topology, "physical_volume": len(cells) * width**3,
                     "bounding_radius": float(np.linalg.norm(actual, axis=1).max()),
                     "block_count": len(cells),
                     "adjacency_degrees": sorted(dict(graph.degree()).values()),
                     "adjacency_wl_hash": nx.weisfeiler_lehman_graph_hash(graph),
                     "numerical_status": "not-solved"}
            entries.append(entry)
            seen.add(signature)
            break
        else:
            raise RuntimeError(f"pre-CAD design exhausted for {index}; no reduced count allowed")
    return {"schema": VERSION, "type": "geometry-design-not-training-data",
            "status": "design-complete-numerical-validation-pending",
            "seed": base_seed, "instances": count,
            "deduplication": "all-48-lattice-isometries-plus-translation; ignores-cell-width",
            "structural_uniqueness_not_distinct_homeomorphism_types": True,
            "genus_counts": dict(Counter(entry["genus"] for entry in entries)),
            "entries": entries, "pre_cad_rejections": rejections,
            "rejection_counts": dict(Counter(item["reason"] for item in rejections)),
            "production_released": False}


def spec_from_entry(entry: dict) -> StructuralSpec:
    return StructuralSpec(entry["id"], tuple(tuple(map(int, cell)) for cell in entry["cells"]),
                          float(entry["cell_width"]), entry["structure_sha256"],
                          int(entry["generation_seed"]), int(entry["genus"]))


def make_solid(spec: StructuralSpec):
    from netgen.occ import Box

    solid = None
    for cell in spec.cells:
        center = (np.asarray(cell) - spec.center) * spec.cell_width
        block = Box(tuple(center - spec.cell_width / 2), tuple(center + spec.cell_width / 2))
        solid = block if solid is None else solid + block
    if solid is None or len(solid.solids) != 1:
        raise RuntimeError("CAD union is not one solid; do not substitute another structure")
    expected_volume = len(spec.cells) * spec.cell_width**3
    if not np.isclose(float(solid.mass), expected_volume, rtol=1.0e-8, atol=1.0e-12):
        raise RuntimeError("CAD volume differs from the structural manifest")
    return solid
