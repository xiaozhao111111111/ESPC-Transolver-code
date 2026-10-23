"""Deterministic, non-overlapping geometry and scene parameters for v3.

This module defines sealed tests but never opens or evaluates them. Generation
must be invoked only after the numerical protocol and checkpoint-selection
rule have been frozen.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict

import numpy as np

from cspgnet.data_generation import formal_neumann_3d as formal


VERSION = "canonical-response-v3"
SPLIT_SIZES = {
    "id_test": 1000,
    "n8_test": 1000,
    "shape_ood": 500,
    "topology_ood": 500,
    "shape_n8_ood": 250,
    "topology_n8_ood": 250,
}


def derived_seed(split: str, component: str, index: int) -> int:
    token = json.dumps([VERSION, split, component, int(index)], separators=(",", ":"))
    return int.from_bytes(hashlib.sha256(token.encode()).digest()[:8], "little")


def _identity():
    return tuple(tuple(float(value) for value in row) for row in np.eye(3))


def new_shape_entries(split: str) -> list[dict]:
    """24 new base shapes per applicable family, independent of old libraries."""
    if split == "id_test":
        families = ("sphere", "ellipsoid", "box")
    elif split == "shape_ood":
        families = ("peanut",)
    elif split == "topology_ood":
        families = ("torus",)
    else:
        raise ValueError("only genuinely new geometry libraries are generated here")
    entries = []
    for family in families:
        for index in range(24):
            rng = np.random.default_rng(derived_seed(split, family, index))
            if family == "sphere":
                size = (float(rng.uniform(0.25, 0.55)),)
            elif family == "ellipsoid":
                size = tuple(float(x) for x in rng.uniform(0.25, 0.55, 3))
            elif family == "box":
                for _ in range(10000):
                    candidate = rng.uniform(0.22, 0.50, 3)
                    if np.linalg.norm(candidate) <= 0.9:
                        size = tuple(float(x) for x in candidate)
                        break
                else:
                    raise RuntimeError("unable to sample an admissible box")
            elif family == "peanut":
                r1, r2 = rng.uniform(0.30, 0.45, 2)
                distance = rng.uniform(0.45, min(0.70, 0.8 * (r1 + r2)))
                size = (float(r1), float(r2), float(distance))
            else:
                size = (float(rng.uniform(0.45, 0.60)), float(rng.uniform(0.14, 0.24)))
            spec = formal.ObjectSpec(
                family, size, _identity(),
                axis_permutation=tuple(range(len(size))) if family in ("ellipsoid", "box") else (),
                smoothing_beta=30.0 if family == "peanut" else None,
            )
            entries.append({
                "id": f"{split}-{family}-{index:02d}",
                "geometry": asdict(spec),
                "generation_seed": derived_seed(split, family, index),
            })
    return entries


def scene_parameters(split: str, index: int, seen_library: list[dict],
                     new_library: list[dict]) -> dict:
    if split not in SPLIT_SIZES or not 0 <= index < SPLIT_SIZES[split]:
        raise ValueError("unknown split or scene index")
    is_n8 = "n8" in split
    half = SPLIT_SIZES[split] // 2
    mixed = ("ood" in split and index < half)
    object_count = 8 if is_n8 else 2 + ((index % half if "ood" in split else index) % 3)
    rng_select = np.random.default_rng(derived_seed(split, "selection", index))
    rng_center = np.random.default_rng(derived_seed(split, "centers", index))
    rng_rotation = np.random.default_rng(derived_seed(split, "rotation", index))
    rng_incident = np.random.default_rng(derived_seed(split, "incident", index))
    by_seen_shape = {
        family: [entry for entry in seen_library if entry["shape"] == family]
        for family in ("sphere", "ellipsoid", "box")
    }
    if any(not candidates for candidates in by_seen_shape.values()):
        raise ValueError("seen geometry library is missing a family")
    if split == "id_test":
        pool = new_library
    elif split == "n8_test":
        pool = seen_library
    else:
        pool = new_library
    if not pool:
        raise ValueError("selected geometry library is empty")
    selected = []
    ood_slot = int(rng_select.integers(0, object_count)) if mixed else None
    for obj_index in range(object_count):
        if mixed and obj_index != ood_slot:
            family = ("sphere", "ellipsoid", "box")[int(rng_select.integers(0, 3))]
            choices = by_seen_shape[family]
        else:
            choices = pool
        entry = choices[int(rng_select.integers(0, len(choices)))]
        shape = entry.get("shape", entry.get("geometry", {}).get("shape"))
        size = entry.get("size", entry.get("geometry", {}).get("size"))
        rotation = (_identity() if shape == "sphere" else tuple(
            tuple(float(v) for v in row)
            for row in formal.random_rotation_matrix(rng_rotation)
        ))
        selected.append({
            "library_id": entry["id"], "shape": shape, "size": size,
            "axis_permutation": entry.get("axis_permutation", entry.get("geometry", {}).get("axis_permutation", [])),
            "smoothing_beta": entry.get("geometry", {}).get("smoothing_beta"),
            "meridian_points": entry.get("geometry", {}).get("meridian_points", 65),
            "rotation": rotation,
        })
    centers = formal.random_centers(rng_center, object_count, minimum_distance=3.6)
    direction = rng_incident.normal(size=3)
    direction /= np.linalg.norm(direction)
    scene = {
        "schema": VERSION,
        "split": split, "index": index,
        "count": object_count,
        "ood_composition": "mixed" if mixed else ("all" if "ood" in split else "seen"),
        "centers": centers.tolist(),
        "incident_direction": direction.tolist(),
        "objects": selected,
        "seeds": {component: derived_seed(split, component, index)
                  for component in ("selection", "centers", "rotation", "incident", "remote_edges")},
    }
    scene["parameter_sha256"] = hashlib.sha256(json.dumps(
        scene, sort_keys=True, separators=(",", ":"),
    ).encode()).hexdigest()
    return scene
