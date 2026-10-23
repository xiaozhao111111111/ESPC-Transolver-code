"""Per-geometry curved-mesh occupancy candidate with rotation-batch reuse.

This is explicitly a discretized mesh classifier, not exact CAD containment.
The caller must supply its approved mesh resolution and separate refinement
audit before using its masks in any released dataset.
"""

from __future__ import annotations

import hashlib
import json
import time
from typing import Any

import numpy as np


def plan_grouped_occupancy_queries(scenes: list[dict]) -> dict:
    """Group layout object queries by base geometry for mesh reuse.

    Identical (geometry ID, structure hash, SO(3) rotation) placements share
    one query record, while every scene/object reference is retained. The
    returned rotations are the exact float64 values from the frozen layout;
    query code may report any numerical SO(3) projection it applies.
    """
    if not isinstance(scenes, list):
        raise ValueError("scenes must be a list")
    geometry_groups: dict[str, dict] = {}
    queries: dict[tuple[str, str], dict] = {}
    scene_ids: set[str] = set()
    for scene_position, scene in enumerate(scenes):
        if not isinstance(scene, dict) or scene.get("index") != scene_position:
            raise ValueError("scene indices must be contiguous and ordered")
        scene_id = scene.get("scene_id")
        if not isinstance(scene_id, str) or not scene_id or scene_id in scene_ids:
            raise ValueError("scene IDs must be nonempty and unique")
        scene_ids.add(scene_id)
        objects = scene.get("objects")
        if not isinstance(objects, list) or not objects:
            raise ValueError(f"scene {scene_id} must contain object placements")
        for object_index, obj in enumerate(objects):
            if not isinstance(obj, dict):
                raise ValueError(f"scene {scene_id} object {object_index} is invalid")
            geometry_id = obj.get("library_id")
            structure_sha = str(obj.get("structure_sha256", "")).lower()
            if not isinstance(geometry_id, str) or not geometry_id:
                raise ValueError(f"scene {scene_id} object {object_index} lacks a geometry ID")
            if len(structure_sha) != 64:
                raise ValueError(f"scene {scene_id} object {object_index} has an invalid structure SHA-256")
            try:
                int(structure_sha, 16)
            except ValueError as error:
                raise ValueError("structure_sha256 must be hexadecimal") from error
            rotation = np.asarray(obj.get("rotation"), dtype=np.float64)
            if (rotation.shape != (3, 3) or not np.isfinite(rotation).all()
                    or not np.allclose(rotation.T @ rotation, np.eye(3), atol=2e-8, rtol=0)
                    or abs(float(np.linalg.det(rotation)) - 1.0) > 2e-8):
                raise ValueError(f"scene {scene_id} object {object_index} rotation is not SO(3)")
            rotation_bytes = np.ascontiguousarray(rotation, dtype="<f8").tobytes()
            rotation_sha = hashlib.sha256(rotation_bytes).hexdigest()
            group = geometry_groups.setdefault(geometry_id, {
                "geometry_id": geometry_id,
                "structure_sha256": structure_sha,
                "queries": [],
            })
            if group["structure_sha256"] != structure_sha:
                raise ValueError(f"geometry ID {geometry_id} maps to multiple structure hashes")
            key = (geometry_id, rotation_sha)
            query = queries.get(key)
            if query is None:
                query_id = hashlib.sha256(
                    json.dumps([geometry_id, structure_sha, rotation_sha],
                               separators=(",", ":")).encode("utf-8")
                ).hexdigest()
                query = {
                    "query_id": query_id,
                    "rotation_sha256": rotation_sha,
                    "rotation": rotation.tolist(),
                    "references": [],
                }
                queries[key] = query
                group["queries"].append(query)
            query["references"].append({"scene_index": scene_position,
                                        "scene_id": scene_id,
                                        "object_index": object_index})

    ordered_groups = []
    total_references = 0
    unique_queries = 0
    for geometry_id in sorted(geometry_groups):
        group = geometry_groups[geometry_id]
        group["queries"].sort(key=lambda row: row["query_id"])
        for query in group["queries"]:
            query["references"].sort(key=lambda row: (row["scene_index"], row["object_index"]))
            total_references += len(query["references"])
            unique_queries += 1
        ordered_groups.append(group)
    return {
        "geometry_count": len(ordered_groups),
        "object_reference_count": total_references,
        "unique_geometry_rotation_query_count": unique_queries,
        "duplicate_query_reuse_count": total_references - unique_queries,
        "groups": ordered_groups,
    }


class MeshContainsOccupancyCandidate:
    """Build one curved solid mesh and reuse it for many rigid rotations.

    One instance belongs to one process and one geometry. Do not share its
    NGSolve mesh across concurrent threads. Geometry-level process parallelism
    is the supported parallel axis; each worker builds its own mesh.
    """

    backend_name = "ngsolve-mesh-contains-curved-mesh-proxy"

    def __init__(self, occ_solid: Any, *, mesh_h: float, curve_order: int = 2):
        if not np.isfinite(mesh_h) or mesh_h <= 0:
            raise ValueError("mesh_h must be finite and positive")
        if curve_order < 1:
            raise ValueError("curve_order must be positive")
        from ngsolve import Mesh
        from netgen.occ import OCCGeometry

        started = time.perf_counter()
        ngmesh = OCCGeometry(occ_solid).GenerateMesh(maxh=float(mesh_h))
        mesh = Mesh(ngmesh)
        mesh.Curve(int(curve_order))
        self._mesh = mesh
        self._ngmesh = ngmesh
        self.mesh_h = float(mesh_h)
        self.curve_order = int(curve_order)
        self.build_seconds = float(time.perf_counter() - started)
        self.mesh_vertices = int(mesh.nv)
        self.mesh_elements = int(mesh.ne)
        self.closed = False

    @staticmethod
    def _so3(rotation: np.ndarray) -> np.ndarray:
        rotation = np.asarray(rotation, dtype=np.float64)
        if (
            rotation.shape != (3, 3)
            or not np.allclose(rotation.T @ rotation, np.eye(3), atol=2.0e-6)
            or not np.isclose(np.linalg.det(rotation), 1.0, atol=2.0e-6)
        ):
            raise ValueError("each rotation must be a proper SO(3) matrix")
        left, _, right = np.linalg.svd(rotation)
        projected = left @ right
        if np.linalg.det(projected) < 0:
            left[:, -1] *= -1
            projected = left @ right
        return projected

    def query(
        self,
        interface_points: np.ndarray,
        rotations: np.ndarray,
        radial_radii: np.ndarray,
    ) -> tuple[np.ndarray, dict]:
        """Return uint8 masks [rotation, interface-node, radius-index].

        For each rotation R, canonical sample points are transformed to the
        object's base CAD frame as row vectors ``x_local = x_canonical @ R``.
        The resolution of the occupancy grid is independent of FEM mesh h.
        """
        if self.closed:
            raise RuntimeError("occupancy mesh has already been closed")
        points = np.asarray(interface_points, dtype=np.float64)
        rotations = np.asarray(rotations, dtype=np.float64)
        radii = np.asarray(radial_radii, dtype=np.float64)
        if points.shape != (128, 3):
            raise ValueError("interface_points must have shape [128,3]")
        if rotations.ndim != 3 or rotations.shape[1:] != (3, 3) or len(rotations) == 0:
            raise ValueError("rotations must have shape [batch,3,3] with batch >= 1")
        if radii.ndim != 1 or len(radii) != 128 or not np.isfinite(radii).all():
            raise ValueError("radial_radii must contain 128 finite sample radii")
        if np.any(radii <= 0) or np.any(radii >= np.max(np.linalg.norm(points, axis=1))):
            raise ValueError("radial_radii must lie strictly inside the canonical sphere")
        point_norms = np.linalg.norm(points, axis=1)
        if not np.isfinite(points).all() or np.any(point_norms <= 0):
            raise ValueError("interface_points must be finite nonzero vectors")
        if len(np.unique(np.round(point_norms, decimals=8))) != 1:
            raise ValueError("interface_points must share one canonical radius")

        base_points = (points / point_norms[:, None])[:, None, :] * radii[None, :, None]
        flat = base_points.reshape(-1, 3)
        masks = np.empty((len(rotations), 128, 128), dtype=np.uint8)
        started = time.perf_counter()
        normalized_rotations = []
        for index, raw_rotation in enumerate(rotations):
            rotation = self._so3(raw_rotation)
            normalized_rotations.append(rotation)
            local_points = flat @ rotation
            masks[index] = np.fromiter(
                (
                    self._mesh.Contains(float(x), float(y), float(z))
                    for x, y, z in local_points
                ),
                dtype=np.uint8,
                count=len(local_points),
            ).reshape(128, 128)
        query_seconds = float(time.perf_counter() - started)
        metadata = {
            "backend": self.backend_name,
            "accuracy_status": "discretized_mesh_proxy_not_exact_cad",
            "mesh_h": self.mesh_h,
            "curve_order": self.curve_order,
            "mesh_vertices": self.mesh_vertices,
            "mesh_elements": self.mesh_elements,
            "rotation_count": int(len(rotations)),
            "points_per_rotation": 128 * 128,
            "query_seconds": query_seconds,
            "queries_per_second": float(len(rotations) * 128 * 128 / max(query_seconds, 1.0e-30)),
            "occupied_counts": [int(np.count_nonzero(row)) for row in masks],
            "mesh_reused_for_rotation_batch": True,
            "rotation_projection_max_abs": float(
                max(np.max(np.abs(raw - normalized)) for raw, normalized in zip(rotations, normalized_rotations))
            ),
        }
        return masks, metadata

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        self._mesh = None
        self._ngmesh = None

    def __enter__(self) -> "MeshContainsOccupancyCandidate":
        return self

    def __exit__(self, *_exc) -> None:
        self.close()
