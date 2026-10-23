"""Geometric validity checks for registered peanut and torus OOD solids."""

from __future__ import annotations

import numpy as np
from scipy.spatial import ConvexHull

from . import formal_neumann_3d as formal


def _surface_topology(triangles: np.ndarray) -> tuple[int, bool, bool, int]:
    edges = np.concatenate((triangles[:, (0, 1)], triangles[:, (1, 2)], triangles[:, (2, 0)]))
    unique_edges, multiplicity = np.unique(np.sort(edges, axis=1), axis=0, return_counts=True)
    used_vertices = np.unique(triangles)
    _, inverse = np.unique(np.sort(edges, axis=1), axis=0, return_inverse=True)
    orientation_balance = np.bincount(
        inverse, weights=np.where(edges[:, 0] < edges[:, 1], 1, -1),
    )
    return (
        int(len(used_vertices) - len(unique_edges) + len(triangles)),
        bool(np.all(multiplicity == 2)),
        bool(np.all(orientation_balance == 0)),
        int(len(used_vertices)),
    )


def _peanut_nonconvexity(spec: formal.ObjectSpec, cad_volume: float) -> float:
    meridian = formal.peanut_meridian(formal.ObjectSpec(
        "peanut", spec.size, spec.rotation,
        smoothing_beta=spec.smoothing_beta, meridian_points=257,
    ))
    azimuth = np.arange(64) * (2.0 * np.pi / 64)
    ring = np.column_stack((
        np.outer(meridian[:, 0], np.cos(azimuth)).ravel(),
        np.outer(meridian[:, 0], np.sin(azimuth)).ravel(),
        np.repeat(meridian[:, 1], len(azimuth)),
    ))
    return float(ConvexHull(ring).volume / cad_volume - 1.0)


def audit_ood_solid(spec: formal.ObjectSpec, maxh: float = 0.2) -> dict:
    if spec.shape not in ("peanut", "torus"):
        raise ValueError("OOD geometry audit expects peanut or torus")
    solid = formal.make_obstacle(spec)
    cad_volume = float(solid.mass)
    vertices, triangles = formal._netgen_surface_arrays(spec, (0.0, 0.0, 0.0), maxh)
    euler, watertight, consistently_oriented, surface_vertices = _surface_topology(triangles)
    used = vertices[np.unique(triangles)]
    local = used @ np.asarray(spec.rotation, dtype=np.float64)
    radius = float(np.linalg.norm(local, axis=1).max())
    if spec.shape == "peanut":
        r1, r2, distance = map(float, spec.size)
        sdf = formal.peanut_sdf(
            np.linalg.norm(local[:, :2], axis=1), local[:, 2],
            r1, r2, distance, 30.0,
        )
        sdf_deviation = float(np.max(np.abs(sdf)))
        nonconvexity = _peanut_nonconvexity(spec, cad_volume)
        expected_euler = 2
    else:
        major, tube = map(float, spec.size)
        sdf = np.hypot(np.linalg.norm(local[:, :2], axis=1) - major, local[:, 2]) - tube
        sdf_deviation = float(np.max(np.abs(sdf)))
        nonconvexity = None
        expected_euler = 0
    checks = {
        "positive_volume": cad_volume > 0,
        "enclosed_by_radius_0_9": radius < 0.9,
        "closed_triangle_mesh": watertight,
        "consistent_outward_orientation": consistently_oriented,
        "expected_euler_characteristic": euler == expected_euler,
        "cad_sdf_deviation": sdf_deviation < 0.002,
    }
    if nonconvexity is not None:
        checks["nonconvex"] = nonconvexity > 0.001
    return {
        "shape": spec.shape,
        "cad_volume": cad_volume,
        "bounding_radius": radius,
        "surface_vertices": surface_vertices,
        "surface_triangles": int(len(triangles)),
        "euler_characteristic": euler,
        "watertight": watertight,
        "consistently_oriented": consistently_oriented,
        "max_cad_sdf_deviation": sdf_deviation,
        "convex_hull_volume_excess": nonconvexity,
        "checks": checks,
        "passed": bool(all(checks.values())),
    }
