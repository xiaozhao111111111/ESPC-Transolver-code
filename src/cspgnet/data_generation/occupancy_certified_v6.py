"""Conservative P2 surface cell cache for the SAME curved-mesh classifier.

Quadratic Bezier control hulls enclose every boundary patch. A connected
component of grid cells disjoint from all hull boxes cannot cross the mesh
boundary. One native Contains query labels each such component; all points
in boundary cells still use native Contains. No CAD fillet is approximated
or removed. This remains a curved-mesh proxy, not exact CAD membership.
"""
from __future__ import annotations

import time
import hashlib
import json
import os
from pathlib import Path
import numpy as np
from scipy import ndimage

from .occupancy_mesh_v6 import MeshContainsOccupancyCandidate


class ConservativeContains:
    def __init__(self, mesh, *, resolution=128):
        from ngsolve import BND, ET, ElementId
        started = time.perf_counter()
        if not 16 <= resolution <= 256:
            raise ValueError("cell resolution must be in [16,256]")
        self.mesh = mesh
        self.resolution = int(resolution)
        nodes = [(0., 0.), (1., 0.), (0., 1.), (.5, 0.), (0., .5), (.5, .5)]
        checks = [(.2, .3), (.61, .17), (.12, .72), (1 / 3, 1 / 3)]
        boxes = []
        max_defect = 0.
        for element in mesh.Elements(BND):
            if element.type != ET.TRIG:
                raise ValueError("only quadratic triangular boundary patches are supported")
            transform = mesh.GetTrafo(ElementId(BND, element.nr))
            p = np.asarray([tuple(transform(u, v).point) for u, v in nodes])
            control = np.concatenate([p[:3], 2 * p[3:] - .5 * p[[0, 0, 1]] - .5 * p[[1, 2, 2]]])
            for u, v in checks:
                a = 1 - u - v
                weights = np.array([a*a, u*u, v*v, 2*a*u, 2*a*v, 2*u*v])
                defect = np.max(np.abs(weights @ control - np.asarray(tuple(transform(u, v).point))))
                max_defect = max(max_defect, float(defect))
            boxes.append([control.min(axis=0), control.max(axis=0)])
        if not boxes or max_defect > 1e-10:
            raise ValueError(f"surface is not the expected polynomial P2 map: defect={max_defect}")
        boxes = np.asarray(boxes)
        lo, hi = boxes[:, 0].min(axis=0), boxes[:, 1].max(axis=0)
        self.lower = lo - .05
        self.step = (hi - lo + .10) / resolution
        occupied = np.zeros((resolution,) * 3, dtype=np.bool_)
        start = np.floor((boxes[:, 0] - 1e-10 - self.lower) / self.step).astype(int)
        stop = np.floor((boxes[:, 1] + 1e-10 - self.lower) / self.step).astype(int)
        for first, last in zip(start, stop):
            occupied[first[0]:last[0]+1, first[1]:last[1]+1, first[2]:last[2]+1] = True
        # A full-cell guard includes native near-boundary point-location tolerance.
        occupied = ndimage.binary_dilation(occupied, structure=np.ones((3, 3, 3), bool))
        components, count = ndimage.label(~occupied)
        labels, indices = np.unique(components, return_index=True)
        membership = np.zeros(count + 1, dtype=np.uint8)
        for label, index in zip(labels, indices):
            if label == 0:
                continue
            ijk = np.asarray(np.unravel_index(int(index), components.shape))
            point = self.lower + (ijk + .5) * self.step
            membership[label] = bool(mesh.Contains(*map(float, point)))
        self.cells = membership[components].astype(np.int8)
        self.cells[occupied] = -1
        self.metadata = {
            "method": "P2-Bezier-hull-cell-components-with-native-boundary-fallback-v1",
            "resolution": resolution, "boundary_patches": len(boxes),
            "maximum_P2_map_check_defect": max_defect,
            "constant_membership_components": count,
            "boundary_cell_fraction": float(occupied.mean()),
            "build_seconds": time.perf_counter() - started,
        }

    def contains(self, points):
        points = np.asarray(points, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
            raise ValueError("points must be a finite [n,3] array")
        cells = np.floor((points - self.lower) / self.step).astype(np.int64)
        in_box = np.all((cells >= 0) & (cells < self.resolution), axis=1)
        result = np.zeros(len(points), dtype=np.bool_)
        selected = np.flatnonzero(in_box)
        values = self.cells[tuple(cells[selected].T)]
        result[selected] = values == 1
        fallback = selected[values == -1]
        result[fallback] = np.fromiter(
            (self.mesh.Contains(*map(float, p)) for p in points[fallback]),
            count=len(fallback), dtype=np.bool_,
        )
        return result, len(fallback)


class CertifiedMeshOccupancy(MeshContainsOccupancyCandidate):
    """Drop-in query; keep the original proxy, mesh levels and every query."""
    def __init__(self, occ_solid, *, mesh_h, curve_order=2, cache_root=None,
                 geometry_id=None, structure_sha256=None):
        if cache_root is None:
            super().__init__(occ_solid, mesh_h=mesh_h, curve_order=curve_order)
            return
        from ngsolve import Mesh
        cache_root = Path(cache_root)
        identity = {"geometry_id": str(geometry_id), "structure_sha256": str(structure_sha256),
                    "mesh_h": float(mesh_h), "curve_order": int(curve_order),
                    "format": "netgen-curved-vol-v1"}
        if not geometry_id or len(str(structure_sha256)) != 64 or curve_order != 2:
            raise ValueError("mesh cache requires a source-bound geometry and quadratic surface")
        cache_root.mkdir(parents=True, exist_ok=True)
        tag = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:20]
        path, proof_path = cache_root / f"mesh_{tag}.vol", cache_root / f"mesh_{tag}.json"
        if path.exists() or proof_path.exists():
            if not path.is_file() or not proof_path.is_file():
                raise RuntimeError(f"partial immutable mesh cache: {path}")
            proof = json.loads(proof_path.read_text(encoding="utf-8"))
            hasher = hashlib.sha256()
            with path.open("rb") as stream:
                for block in iter(lambda: stream.read(1024*1024), b""):
                    hasher.update(block)
            if proof.get("identity") != identity or proof.get("sha256") != hasher.hexdigest():
                raise RuntimeError(f"stale or modified mesh cache: {path}")
            self.mesh_cache_sha256 = hasher.hexdigest()
            started = time.perf_counter()
            self._ngmesh = None
            self._mesh = Mesh(str(path))
            self._mesh.Curve(curve_order)
            self.mesh_h, self.curve_order = float(mesh_h), int(curve_order)
            self.closed = False
            self.mesh_vertices, self.mesh_elements = int(self._mesh.nv), int(self._mesh.ne)
            if [self.mesh_vertices, self.mesh_elements] != proof.get("counts"):
                raise RuntimeError("reloaded mesh counts differ from immutable cache")
            self.build_seconds = 0.0
            self.mesh_cache_load_seconds = time.perf_counter()-started
        else:
            super().__init__(occ_solid, mesh_h=mesh_h, curve_order=curve_order)
            pending = path.with_name(path.stem + ".pending.vol")
            if pending.exists():
                raise RuntimeError(f"stale pending mesh cache needs review: {pending}")
            self._ngmesh.Save(str(pending))
            os.replace(pending, path)
            hasher = hashlib.sha256()
            with path.open("rb") as stream:
                for block in iter(lambda: stream.read(1024*1024), b""):
                    hasher.update(block)
            payload = {"identity": identity, "sha256": hasher.hexdigest(),
                       "counts": [self.mesh_vertices, self.mesh_elements]}
            self.mesh_cache_sha256 = hasher.hexdigest()
            pending_proof = proof_path.with_name(proof_path.name + ".pending")
            with pending_proof.open("x", encoding="utf-8") as stream:
                json.dump(payload, stream, sort_keys=True)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(pending_proof, proof_path)
            self.mesh_cache_load_seconds = 0.0
        self.cache_path = str(path)

    def query(self, interface_points, rotations, radial_radii):
        points = np.asarray(interface_points, dtype=np.float64)
        radii = np.asarray(radial_radii, dtype=np.float64)
        rotations = np.asarray(rotations, dtype=np.float64)
        if self.closed or points.shape != (128, 3) or radii.shape != (128,):
            raise ValueError("invalid or closed occupancy query")
        if rotations.ndim != 3 or rotations.shape[1:] != (3, 3) or not len(rotations):
            raise ValueError("invalid rotations")
        # Reuse the original query's complete validation and verify representative
        # rotations against unaccelerated native Contains on EVERY built mesh.
        indices = sorted(set([0, len(rotations)//2, len(rotations)-1]))
        reference, reference_meta = super().query(points, rotations[indices], radii)
        accelerator = ConservativeContains(self._mesh)
        base = ((points / np.linalg.norm(points, axis=1)[:, None])[:, None] * radii[None, :, None]).reshape(-1, 3)
        started = time.perf_counter()
        masks = np.empty((len(rotations), 128, 128), np.uint8)
        fallback = []
        for index, rotation in enumerate(rotations):
            mask, count = accelerator.contains(base @ self._so3(rotation))
            masks[index] = mask.reshape(128, 128)
            fallback.append(count)
        mismatch = int(np.count_nonzero(masks[indices] != reference))
        if mismatch:
            raise RuntimeError(f"cell cache differs from native Contains at {mismatch} verification points")
        metadata = dict(reference_meta)
        metadata.update(rotation_count=len(rotations), occupied_counts=masks.sum(axis=(1,2)).tolist(),
                        query_seconds=time.perf_counter()-started,
                        accelerator=accelerator.metadata,
                        mesh_cache_path=getattr(self, "cache_path", None),
                        mesh_cache_sha256=getattr(self, "mesh_cache_sha256", None),
                        mesh_cache_load_seconds=getattr(self, "mesh_cache_load_seconds", 0.0),
                        mesh_build_seconds=self.build_seconds,
                        native_equivalence_rotations=len(indices), native_equivalence_mismatches=mismatch,
                        native_equivalence_seconds=reference_meta["query_seconds"],
                        fallback_points=sum(fallback), total_points=int(masks.size))
        return masks, metadata
