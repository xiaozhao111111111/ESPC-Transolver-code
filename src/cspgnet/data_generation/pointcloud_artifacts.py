"""NumPy-only verification of immutable, precomputed CAD pointcloud files."""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np


POINTS_PER_OBJECT = 2048


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_precomputed_pointcloud(record: dict, base_dir: Path,
                               memory: dict) -> tuple[np.ndarray, np.ndarray] | None:
    """Load a geometry-bound pointcloud if the manifest supplies its file.

    A missing ``pointcloud_file`` means the caller may fall back to its normal
    analytic-shape path. A declared but invalid artifact always fails closed.
    """
    filename = record.get("pointcloud_file")
    if filename is None:
        return None
    library_id = str(record.get("library_id", "<unknown>"))
    digest = record.get("pointcloud_sha256")
    if (not isinstance(digest, str) or len(digest) != 64
            or any(char not in "0123456789abcdefABCDEF" for char in digest)):
        raise ValueError(f"CAD pointcloud record lacks a valid file hash for {library_id}")
    token = digest.lower()
    cache_key = "manifest-sha256:" + token
    if cache_key in memory:
        return memory[cache_key]
    path = Path(filename)
    if not path.is_absolute():
        path = Path(base_dir) / path
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"precomputed CAD pointcloud missing for {library_id}: {path}")
    if _sha256_file(path) != token:
        raise RuntimeError(f"precomputed CAD pointcloud file hash mismatch for {library_id}")
    with np.load(path, allow_pickle=False) as opened:
        if not {"points", "normals", "structure_sha256"}.issubset(opened.files):
            raise ValueError(f"precomputed CAD pointcloud schema is incomplete: {path}")
        points = np.asarray(opened["points"], dtype=np.float32)
        normals = np.asarray(opened["normals"], dtype=np.float32)
        structure_hash = str(opened["structure_sha256"]).lower()
    if structure_hash != str(record.get("structure_sha256", "")).lower():
        raise RuntimeError(f"precomputed pointcloud geometry hash mismatch for {library_id}")
    if (points.shape != (POINTS_PER_OBJECT, 3) or normals.shape != points.shape
            or not np.isfinite(points).all() or not np.isfinite(normals).all()
            or not np.allclose(np.linalg.norm(normals, axis=1), 1.0, atol=1e-4)):
        raise ValueError(f"precomputed CAD pointcloud dimensions/normals invalid for {library_id}")
    memory[cache_key] = (points, normals)
    return points, normals
