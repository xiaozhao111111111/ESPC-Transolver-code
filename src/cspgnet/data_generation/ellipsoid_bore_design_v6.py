"""Deterministic, constraint-aware parameter designs for the V6 bore family.

This module proposes CAD parameters only. It performs no CAD construction,
mesh generation, FEM/BEM solve, or numerical acceptance decision.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import itertools
import json
import math
from typing import Iterable

import numpy as np


SCHEMA = "ellipsoid-through-bore-v6-proposals-v1"
GENUS_COUNTS = {0: 200, 1: 150, 2: 100, 3: 50}
AXIS_RANGE = (0.58, 0.70)
BORE_RADIUS_RANGE = (0.075, 0.130)
MIN_CLEARANCE = 0.10
FILLET_RANGE = (0.012, 0.025)
MASTER_SEED = 20261007
LHS_MAXIMIN_REPLICATES = 12


def _stable_seed(*parts: object) -> int:
    payload = json.dumps(parts, ensure_ascii=True, sort_keys=True,
                         separators=(",", ":"), allow_nan=False).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little", signed=False)


def _unit_design_dimensions(genus: int) -> int:
    if genus not in GENUS_COUNTS:
        raise ValueError("genus must be 0, 1, 2, or 3")
    layout_dimensions = {0: 0, 1: 2, 2: 4, 3: 8}[genus]
    return 3 + genus + layout_dimensions + int(genus > 0)


def _lhs_once(sample_count: int, dimension_count: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    unit = np.empty((sample_count, dimension_count), dtype=np.float64)
    for column in range(dimension_count):
        unit[:, column] = (rng.permutation(sample_count) + rng.random(sample_count)) / sample_count
    return unit


def _maximin_score(unit: np.ndarray) -> float:
    if len(unit) < 2:
        return math.inf
    delta = unit[:, None, :] - unit[None, :, :]
    squared = np.einsum("ijk,ijk->ij", delta, delta)
    upper = squared[np.triu_indices(len(unit), k=1)]
    return float(np.min(upper))


def latin_hypercube_unit_design(sample_count: int, dimension_count: int, *,
                                seed: int, maximin_replicates: int = LHS_MAXIMIN_REPLICATES) -> np.ndarray:
    """Return a deterministic maximin-selected randomized Latin hypercube."""
    if sample_count < 1 or dimension_count < 1 or maximin_replicates < 1:
        raise ValueError("sample_count, dimension_count and maximin_replicates must be positive")
    root = np.random.SeedSequence(int(seed))
    children = root.spawn(maximin_replicates)
    best = None
    best_score = -1.0
    for child in children:
        child_seed = int(child.generate_state(1, dtype=np.uint64)[0])
        candidate = _lhs_once(sample_count, dimension_count, child_seed)
        score = _maximin_score(candidate)
        if score > best_score:
            best, best_score = candidate, score
    assert best is not None
    return best


def _canonical_structure_signature(axes: Iterable[float], bores: Iterable[Iterable[float]],
                                   fillet: float) -> str:
    axes = tuple(float(value) for value in axes)
    bores = tuple(tuple(float(value) for value in bore) for bore in bores)
    scale = float(np.prod(axes) ** (1.0 / 3.0))
    normalized_axes = tuple(round(value / scale, 8) for value in axes)
    candidates = []
    for sign_x, sign_y in itertools.product((-1.0, 1.0), repeat=2):
        normalized_bores = tuple(sorted((round(sign_x * x / scale, 8),
                                         round(sign_y * y / scale, 8),
                                         round(radius / scale, 8))
                                        for x, y, radius in bores))
        candidates.append((normalized_axes, normalized_bores, round(fillet / scale, 8)))
    canonical = min(candidates)
    return hashlib.sha256(json.dumps(canonical, separators=(",", ":")).encode("ascii")).hexdigest()


def _disk_point(unit_radius: float, unit_angle: float, max_radius: float) -> tuple[float, float]:
    radius = max_radius * math.sqrt(float(unit_radius))
    angle = 2.0 * math.pi * float(unit_angle)
    return radius * math.cos(angle), radius * math.sin(angle)


def _build_parameters(genus: int, unit: np.ndarray) -> dict:
    if unit.shape != (_unit_design_dimensions(genus),) or np.any(unit <= 0.0) or np.any(unit >= 1.0):
        raise ValueError("unit design row has the wrong shape or leaves the open unit cube")
    axes = tuple(sorted((AXIS_RANGE[0] + float(value) * (AXIS_RANGE[1] - AXIS_RANGE[0])
                         for value in unit[:3]), reverse=True))
    axial, transverse, _shortest = axes
    cursor = 3
    radii = [BORE_RADIUS_RANGE[0] + float(unit[cursor + index])
             * (BORE_RADIUS_RANGE[1] - BORE_RADIUS_RANGE[0])
             for index in range(genus)]
    cursor += genus
    bores: list[tuple[float, float, float]] = []

    if genus == 1:
        max_radius = transverse - radii[0] - MIN_CLEARANCE
        x, y = _disk_point(unit[cursor], unit[cursor + 1], max_radius)
        bores.append((x, y, radii[0]))
        cursor += 2
    elif genus == 2:
        minimum_separation = radii[0] + radii[1] + MIN_CLEARANCE
        center_margin = min(transverse - radius - MIN_CLEARANCE for radius in radii)
        maximum_separation = 2.0 * center_margin
        if maximum_separation < minimum_separation:
            raise ValueError("genus-2 bore packing infeasible under clearance constraints")
        separation = minimum_separation + float(unit[cursor]) * (maximum_separation - minimum_separation)
        offset_max = max(0.0, center_margin - separation / 2.0)
        ox, oy = _disk_point(unit[cursor + 1], unit[cursor + 2], offset_max)
        axis_angle = math.pi * float(unit[cursor + 3])
        dx, dy = math.cos(axis_angle) * separation / 2.0, math.sin(axis_angle) * separation / 2.0
        bores.extend(((ox - dx, oy - dy, radii[0]), (ox + dx, oy + dy, radii[1])))
        cursor += 4
    elif genus == 3:
        pair_separations = [radii[i] + radii[j] + MIN_CLEARANCE
                            for i in range(3) for j in range(i)]
        minimum_ring_radius = max(pair_separations) / math.sqrt(3.0)
        maximum_ring_radius = min(transverse - radius - MIN_CLEARANCE for radius in radii)
        # Keep fixed safety margins for bounded radial/angular irregularity.
        lower = minimum_ring_radius + 0.03
        upper = maximum_ring_radius - 0.02
        if upper < lower:
            # The frozen size/clearance ranges should make this very rare; the
            # symmetric limiting design remains strictly separated and valid.
            lower, upper = minimum_ring_radius, maximum_ring_radius
        if upper < lower:
            raise ValueError("genus-3 bore packing infeasible under clearance constraints")
        ring_radius = lower + float(unit[cursor]) * (upper - lower)
        phase = 2.0 * math.pi * float(unit[cursor + 1])
        angle_jitter = (unit[cursor + 2:cursor + 5] - 0.5) * 0.10
        radial_jitter = (unit[cursor + 5:cursor + 8] - 0.5) * 0.02
        candidate = [
            ((ring_radius + float(radial_jitter[index]))
             * math.cos(phase + 2.0 * math.pi * index / 3.0 + float(angle_jitter[index])),
             (ring_radius + float(radial_jitter[index]))
             * math.sin(phase + 2.0 * math.pi * index / 3.0 + float(angle_jitter[index])),
             radii[index])
            for index in range(3)
        ]
        if _clearances_pass(candidate, transverse):
            bores.extend(candidate)
        else:
            # A deterministic, bounded feasibility repair: remove only the
            # small layout perturbations; do not change size or gate values.
            bores.extend((ring_radius * math.cos(phase + 2.0 * math.pi * index / 3.0),
                          ring_radius * math.sin(phase + 2.0 * math.pi * index / 3.0),
                          radii[index]) for index in range(3))
        cursor += 8

    if genus == 0:
        fillet = 0.0
    else:
        min_bore_radius = min(radii)
        fillet_low = max(FILLET_RANGE[0], 0.15 * min_bore_radius)
        fillet_high = min(FILLET_RANGE[1], 0.22 * min_bore_radius)
        if fillet_high < fillet_low:
            raise ValueError("bore size admits no fillet in the frozen pilot-supported range")
        fillet = fillet_low + float(unit[cursor]) * (fillet_high - fillet_low)

    return {"axes": axes, "bores": tuple(bores), "fillet": float(fillet)}


def _clearances_pass(bores: Iterable[Iterable[float]], transverse: float,
                     tolerance: float = 1.0e-12) -> bool:
    bores = tuple(tuple(float(value) for value in bore) for bore in bores)
    for x, y, radius in bores:
        if transverse - math.hypot(x, y) - radius < MIN_CLEARANCE - tolerance:
            return False
    for first in range(len(bores)):
        for second in range(first):
            x1, y1, r1 = bores[first]
            x2, y2, r2 = bores[second]
            if math.hypot(x1 - x2, y1 - y2) - r1 - r2 < MIN_CLEARANCE - tolerance:
                return False
    return True


def validate_parameters(genus: int, parameters: dict) -> dict:
    axes = tuple(float(value) for value in parameters["axes"])
    bores = tuple(tuple(float(value) for value in bore) for bore in parameters["bores"])
    fillet = float(parameters["fillet"])
    if len(axes) != 3 or not (AXIS_RANGE[0] <= axes[2] <= axes[1] <= axes[0] <= AXIS_RANGE[1]):
        raise ValueError("parent semiaxes violate the frozen range/order")
    if len(bores) != genus:
        raise ValueError("bore count must match genus")
    for _, _, radius in bores:
        if not BORE_RADIUS_RANGE[0] <= radius <= BORE_RADIUS_RANGE[1]:
            raise ValueError("bore radius leaves the frozen range")
    if not _clearances_pass(bores, axes[1]):
        raise ValueError("bore-wall or bore-to-bore clearance is below 0.10")
    if genus == 0 and fillet != 0.0:
        raise ValueError("genus-0 parent ellipsoid must not have a fillet")
    if genus > 0:
        if not FILLET_RANGE[0] <= fillet <= FILLET_RANGE[1]:
            raise ValueError("fillet radius leaves the frozen range")
        if fillet >= min(radius for _, _, radius in bores):
            raise ValueError("fillet radius must be smaller than each bore radius")
    bounding_radius = axes[0]
    if not 0.0 < bounding_radius < 0.9:
        raise ValueError("candidate violates the strict outer-radius limit")
    wall_clearances = [axes[1] - math.hypot(x, y) - radius for x, y, radius in bores]
    bore_clearances = [
        math.hypot(bores[i][0] - bores[j][0], bores[i][1] - bores[j][1])
        - bores[i][2] - bores[j][2]
        for i in range(len(bores)) for j in range(i)
    ]
    return {"passed": True, "genus": genus,
            "minimum_wall_clearance": min(wall_clearances) if wall_clearances else None,
            "minimum_bore_bore_clearance": min(bore_clearances) if bore_clearances else None,
            "bounding_radius": bounding_radius}


def generate_stratum(genus: int, candidate_indices: Iterable[int], *, master_seed: int,
                     distribution: str) -> list[dict]:
    indices = [int(value) for value in candidate_indices]
    if not indices or any(value < 1 for value in indices) or len(indices) != len(set(indices)):
        raise ValueError("candidate indices must be nonempty, unique positive integers")
    dimensions = _unit_design_dimensions(genus)
    design_seed = _stable_seed(SCHEMA, int(master_seed), genus,
                               min(indices), max(indices), len(indices), "parameter-design")
    if distribution == "stratified_lhs_maximin":
        unit_design = latin_hypercube_unit_design(len(indices), dimensions, seed=design_seed)
    elif distribution == "iid_uniform":
        rng = np.random.default_rng(design_seed)
        unit_design = rng.random((len(indices), dimensions))
    else:
        raise ValueError("distribution must be stratified_lhs_maximin or iid_uniform")

    rows = []
    for candidate_index, unit in zip(indices, unit_design):
        parameters = _build_parameters(genus, unit)
        validation = validate_parameters(genus, parameters)
        design_row = {
            "id": f"ellipsoid-bore-v6-{candidate_index:06d}",
            "candidate_index": candidate_index,
            "genus": genus,
            "status": "proposed-not-cad-audited",
            **parameters,
            "structure_sha256": _canonical_structure_signature(
                parameters["axes"], parameters["bores"], parameters["fillet"]),
            "sampling": {"distribution": distribution, "master_seed": int(master_seed),
                         "design_seed": design_seed, "design_unit_row": unit.tolist()},
            "parameter_preflight": validation,
        }
        rows.append(design_row)
    signatures = [row["structure_sha256"] for row in rows]
    if len(signatures) != len(set(signatures)):
        raise RuntimeError(f"duplicate scale/rotation-invariant structure in genus-{genus} design batch")
    return rows


def generate_initial_proposals(*, master_seed: int = MASTER_SEED,
                               distribution: str) -> dict:
    """Propose the exact initial quota; rows remain unbuilt and unaccepted."""
    entries = []
    next_index = 1
    for genus, count in GENUS_COUNTS.items():
        rows = generate_stratum(genus, range(next_index, next_index + count),
                                master_seed=master_seed, distribution=distribution)
        entries.extend(rows)
        next_index += count
    counts = Counter(row["genus"] for row in entries)
    if counts != Counter(GENUS_COUNTS):
        raise RuntimeError(f"proposal quota mismatch: {dict(counts)}")
    run_order_seed = _stable_seed(SCHEMA, int(master_seed), "cad-audit-order")
    permutation = np.random.default_rng(run_order_seed).permutation(len(entries))
    for rank, index in enumerate(permutation, 1):
        entries[int(index)]["deterministic_cad_run_order"] = rank
    entries.sort(key=lambda row: row["candidate_index"])
    return {
        "schema": SCHEMA,
        "status": "500-proposals-not-cad-or-numerically-audited",
        "production_released": False,
        "accepted_count": 0,
        "new_labeled_training_samples": 0,
        "expected_initial_proposals": 500,
        "genus_counts": {str(key): value for key, value in GENUS_COUNTS.items()},
        "master_seed": int(master_seed),
        "distribution": distribution,
        "parameter_ranges": {"parent_semiaxes": AXIS_RANGE,
                             "through_bore_radius": BORE_RADIUS_RANGE,
                             "minimum_wall_and_bore_spacing": MIN_CLEARANCE,
                             "local_fillet_radius": FILLET_RANGE},
        "entries": entries,
    }
