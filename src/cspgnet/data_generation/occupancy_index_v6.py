"""Source-bound, resumable occupancy artifacts for the V6 scene pipeline.

This module contains only array/manifest handling.  It does not build CAD or
FEM meshes; the guarded CLI is responsible for producing independently
compared reference/candidate masks before calling these routines.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping

import numpy as np

from .occupancy_mesh_v6 import plan_grouped_occupancy_queries
from .scene_dataset_v6 import OCCUPANCY_SCHEMA


METHOD_SCREEN_SCHEMA = "structural-smooth-v6-occupancy-hybrid-screen-v1"
METHOD_GATE_SCHEMA = "structural-smooth-v6-occupancy-method-gate-v1"
OCCUPANCY_INDEX_SCHEMA = "structural-smooth-v6-scene-occupancy-index-v1"
GEOMETRY_CACHE_SCHEMA = "structural-smooth-v6-geometry-occupancy-cache-v1"
SCENE_PROOF_SCHEMA = "geometry-occupancy-scene-audit-v1"
MASK_SHAPE = (128, 128)
RESERVE_BYTES = 1024**3


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=False, allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def rotation_sha256(rotation: Any) -> str:
    value = np.asarray(rotation, dtype=np.float64)
    if value.shape != (3, 3) or not np.isfinite(value).all():
        raise ValueError("rotation must be finite [3,3]")
    if (not np.allclose(value.T @ value, np.eye(3), atol=2e-8, rtol=0)
            or abs(float(np.linalg.det(value)) - 1.0) > 2e-8):
        raise ValueError("rotation must be in SO(3)")
    return hashlib.sha256(np.ascontiguousarray(value, dtype="<f8").tobytes()).hexdigest()


def _atomic_json(path: Path, payload: dict) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.",
                                     suffix=".pending", delete=False) as stream:
        temporary = Path(stream.name)
        stream.write((json.dumps(payload, indent=2, sort_keys=True,
                                 ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8"))
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    return sha256_file(path)


def _atomic_npz(path: Path, **arrays: np.ndarray) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.stem}.",
                                     suffix=".pending", delete=False) as stream:
        temporary = Path(stream.name)
        np.savez_compressed(stream, **arrays)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    return sha256_file(path)


def select_method_gate(screen: dict, screen_sha256: str) -> dict:
    """Validate a real two-family screen and choose one common passing band."""
    if (screen.get("schema") != METHOD_SCREEN_SCHEMA
            or screen.get("status") != "passed-screen"
            or screen.get("production_authorized") is not False):
        raise ValueError("occupancy method screen is not a passed, non-release screen")
    thresholds = screen.get("thresholds")
    scope = screen.get("scope")
    cases = screen.get("cases")
    if not isinstance(thresholds, dict) or not isinstance(scope, dict) or not isinstance(cases, list):
        raise ValueError("occupancy method screen is missing scope, cases, or gates")
    mismatch_limit = thresholds.get("per_case_per_rotation_mismatch_fraction_le")
    speed_limit = thresholds.get("per_case_amortized_speedup_at_60_placements_ge")
    if (isinstance(mismatch_limit, bool) or not isinstance(mismatch_limit, (int, float))
            or not 0 < float(mismatch_limit) < 1
            or isinstance(speed_limit, bool) or not isinstance(speed_limit, (int, float))
            or float(speed_limit) <= 1):
        raise ValueError("method screen has invalid registered thresholds")
    if (scope.get("case_ids") != ["ellipsoid-g1-a", "ellipsoid-g3-b"]
            or float(scope.get("mesh_h_reference", np.nan)) != 0.01
            or float(scope.get("mesh_h_candidate", np.nan)) != 0.02
            or type(scope.get("curve_order")) is not int or scope["curve_order"] != 2
            or scope.get("points_per_rotation") != 128 * 128
            or thresholds.get("screen_pass_requires_both_cases") is not True):
        raise ValueError("method screen scope differs from the registered V6 calibration")
    expected_ids = {"ellipsoid-g1-a", "ellipsoid-g3-b"}
    by_id = {row.get("case_id"): row for row in cases if isinstance(row, dict)}
    if set(by_id) != expected_ids:
        raise ValueError("method screen must contain exactly the registered genus-1/genus-3 cases")
    common: dict[float, list[float]] = {}
    for case_id, case in by_id.items():
        if case.get("status") != "completed":
            raise ValueError(f"method-screen case {case_id} did not complete")
        for result in case.get("hybrid_results", []):
            band = result.get("uncertainty_band")
            comparisons = result.get("reference_comparison")
            speedup = result.get("amortized_speedup_at_60_placements")
            if (isinstance(band, bool) or not isinstance(band, (int, float))
                    or isinstance(speedup, bool) or not isinstance(speedup, (int, float))
                    or not isinstance(comparisons, list) or not comparisons):
                continue
            fractions = [row.get("mismatch_fraction") for row in comparisons
                         if isinstance(row, dict)]
            if len(fractions) != len(comparisons):
                continue
            if any(isinstance(value, bool) or not isinstance(value, (int, float))
                   or not np.isfinite(value) or value < 0 for value in fractions):
                continue
            if (max(fractions) <= float(mismatch_limit)
                    and float(speedup) >= float(speed_limit)):
                common.setdefault(float(band), []).append(float(speedup))
    passing_bands = {band: speeds for band, speeds in common.items()
                     if len(speeds) == len(expected_ids)}
    if not passing_bands:
        raise ValueError("no single uncertainty band passed accuracy and speed gates for both families")
    selected_band, speeds = max(passing_bands.items(), key=lambda item: min(item[1]))
    if len(screen_sha256) != 64:
        raise ValueError("method screen SHA-256 is invalid")
    int(screen_sha256, 16)
    return {
        "schema": METHOD_GATE_SCHEMA,
        "status": "passed",
        "occupancy_schema": OCCUPANCY_SCHEMA,
        "method": "analytic-ellipsoid-cylinder-with-mesh-boundary-band-fallback",
        "method_screen_sha256": screen_sha256.lower(),
        "acceptance_threshold": float(mismatch_limit),
        "selected_uncertainty_band": selected_band,
        "minimum_calibration_speedup": float(min(speeds)),
        "calibration_cases": sorted(expected_ids),
        "mesh_h_reference": float(scope.get("mesh_h_reference", np.nan)),
        "mesh_h_candidate": float(scope.get("mesh_h_candidate", np.nan)),
        "curve_order": int(scope.get("curve_order", 0)),
        "production_authorized": False,
    }


def validate_method_gate(gate: dict, gate_sha256: str) -> dict:
    if (gate.get("schema") != METHOD_GATE_SCHEMA or gate.get("status") != "passed"
            or gate.get("occupancy_schema") != OCCUPANCY_SCHEMA
            or gate.get("production_authorized") is not False):
        raise ValueError("method gate must be a passed source-bound occupancy-method receipt")
    if (gate.get("method") != "analytic-ellipsoid-cylinder-with-mesh-boundary-band-fallback"
            or gate.get("calibration_cases") != ["ellipsoid-g1-a", "ellipsoid-g3-b"]
            or gate.get("mesh_h_reference") != 0.01
            or gate.get("mesh_h_candidate") != 0.02
            or gate.get("curve_order") != 2):
        raise ValueError("method gate does not match the registered hybrid occupancy implementation")
    for key in ("method_screen_sha256",):
        value = gate.get(key)
        if not isinstance(value, str) or len(value) != 64:
            raise ValueError(f"method gate {key} must be a SHA-256 digest")
        int(value, 16)
    threshold = gate.get("acceptance_threshold")
    band = gate.get("selected_uncertainty_band")
    if (isinstance(threshold, bool) or not isinstance(threshold, (int, float))
            or not 0 < float(threshold) < 1
            or isinstance(band, bool) or not isinstance(band, (int, float))
            or float(band) <= 0):
        raise ValueError("method gate numeric settings are invalid")
    if len(gate_sha256) != 64:
        raise ValueError("method gate file SHA-256 is invalid")
    int(gate_sha256, 16)
    return {**gate, "acceptance_threshold": float(threshold),
            "selected_uncertainty_band": float(band),
            "method_audit_sha256": gate_sha256.lower()}


def validate_geometry_query_result(
    *, geometry_id: str, structure_sha256: str, queries: list[dict],
    reference_masks: np.ndarray, candidate_masks: np.ndarray,
    method_gate: dict, method_audit_sha256: str,
) -> tuple[np.ndarray, dict]:
    """Validate candidate masks against the independent reference per rotation."""
    reference = np.asarray(reference_masks)
    candidate = np.asarray(candidate_masks)
    expected = (len(queries), *MASK_SHAPE)
    if reference.shape != expected or candidate.shape != expected:
        raise ValueError(f"both occupancy arrays must have shape {expected}")
    if (reference.dtype != np.uint8 or candidate.dtype != np.uint8
            or not np.isin(reference, (0, 1)).all() or not np.isin(candidate, (0, 1)).all()):
        raise ValueError("occupancy masks must be binary uint8 arrays")
    if len(geometry_id) == 0 or len(structure_sha256) != 64 or len(method_audit_sha256) != 64:
        raise ValueError("geometry and method audit identities are invalid")
    for digest in (structure_sha256, method_audit_sha256):
        int(digest, 16)
    threshold = float(method_gate["acceptance_threshold"])
    proof_rows = []
    seen_query_ids: set[str] = set()
    for index, query in enumerate(queries):
        query_id = query.get("query_id")
        rotation_hash = query.get("rotation_sha256")
        if (not isinstance(query_id, str) or len(query_id) != 64
                or query_id in seen_query_ids or rotation_sha256(query.get("rotation")) != rotation_hash):
            raise ValueError("query IDs and rotations must be unique, valid, and exactly bound")
        int(query_id, 16)
        seen_query_ids.add(query_id)
        mask_hash = hashlib.sha256(np.ascontiguousarray(candidate[index]).tobytes()).hexdigest()
        reference_hash = hashlib.sha256(np.ascontiguousarray(reference[index]).tobytes()).hexdigest()
        mismatch = int(np.count_nonzero(reference[index] != candidate[index]))
        fraction = float(mismatch / reference[index].size)
        passed = fraction < threshold
        proof_rows.append({
            "query_id": query_id,
            "rotation_sha256": rotation_hash,
            "reference_mask_sha256": reference_hash,
            "mask_sha256": mask_hash,
            "mismatch_count": mismatch,
            "point_count": int(reference[index].size),
            "max_disagreement_fraction": fraction,
            "acceptance_threshold": threshold,
            "status": "passed" if passed else "failed", 
        })
    proof = {
        "schema": GEOMETRY_CACHE_SCHEMA,
        "geometry_id": geometry_id,
        "structure_sha256": structure_sha256.lower(),
        "occupancy_schema": OCCUPANCY_SCHEMA,
        "method_audit_sha256": method_audit_sha256.lower(),
        "acceptance_threshold": threshold,
        "status": "passed" if all(row["status"] == "passed" for row in proof_rows) else "failed",
        "query_count": len(queries),
        "queries": proof_rows,
    }
    return candidate, proof


def write_geometry_cache(
    cache_root: Path, *, geometry_id: str, structure_sha256: str,
    library_sha256: str, layout_sha256: str, method_audit_sha256: str,
    queries: list[dict], reference_masks: np.ndarray, candidate_masks: np.ndarray,
    method_gate: dict,
) -> dict:
    if not geometry_id or Path(geometry_id).name != geometry_id:
        raise ValueError("geometry_id must be a safe path component")
    for value in (structure_sha256, library_sha256, layout_sha256, method_audit_sha256):
        if len(value) != 64:
            raise ValueError("all cache source bindings must be SHA-256 digests")
        int(value, 16)
    candidate, proof = validate_geometry_query_result(
        geometry_id=geometry_id, structure_sha256=structure_sha256,
        queries=queries, reference_masks=reference_masks,
        candidate_masks=candidate_masks, method_gate=method_gate,
        method_audit_sha256=method_audit_sha256,
    )
    root = Path(cache_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    archive = root / f"{geometry_id}.npz"
    sidecar = root / f"{geometry_id}.json"
    if sidecar.exists():
        raise FileExistsError(f"refusing to overwrite an existing geometry proof: {geometry_id}")
    source = {
        "geometry_id": geometry_id, "structure_sha256": structure_sha256.lower(),
        "library_manifest_sha256": library_sha256.lower(),
        "layout_manifest_sha256": layout_sha256.lower(),
        "method_audit_sha256": method_audit_sha256.lower(),
        "occupancy_schema": OCCUPANCY_SCHEMA,
        "query_ids": [row["query_id"] for row in queries],
        "rotation_sha256": [row["rotation_sha256"] for row in queries],
    }
    expected_arrays = {
        "reference_masks": np.asarray(reference_masks, dtype=np.uint8),
        "candidate_masks": candidate,
        "query_ids": np.asarray(source["query_ids"], dtype="U64"),
        "rotation_sha256": np.asarray(source["rotation_sha256"], dtype="U64"),
    }
    if archive.exists():
        # Recover a crash between the atomic NPZ commit and JSON commit only
        # when a fresh deterministic computation exactly matches the archive.
        with np.load(archive, allow_pickle=False) as data:
            existing_arrays = {key: np.asarray(data[key]) for key in data.files}
        if set(existing_arrays) != set(expected_arrays) or any(
                not np.array_equal(existing_arrays[key], value)
                for key, value in expected_arrays.items()):
            raise ValueError(f"orphan geometry cache does not match recomputed content: {geometry_id}")
        archive_sha = sha256_file(archive)
    else:
        archive_sha = _atomic_npz(archive, **expected_arrays)
    sidecar_sha = _atomic_json(sidecar, {
        **source, **proof,
        "archive_path": archive.name,
        "archive_sha256": archive_sha,
    })
    # The sidecar checksum is kept in run state, avoiding a recursive hash.
    return {"archive_path": str(archive), "archive_sha256": archive_sha,
            "proof_path": str(sidecar), "proof_sha256": sidecar_sha,
            "status": proof["status"], "passed_queries": sum(
                row["status"] == "passed" for row in proof["queries"]),
            "query_count": len(queries)}


def load_geometry_cache(
    archive_path: Path, proof_path: Path, *, geometry_id: str,
    structure_sha256: str, library_sha256: str, layout_sha256: str,
    method_audit_sha256: str,
) -> tuple[dict[str, np.ndarray], dict]:
    # Keep canonicalizing junctions/symlinks, but do not require strict
    # Win32 realpath access before opening. On some managed Windows workspaces
    # strict realpath raises Access Denied for readable files; the following
    # read/hash operations still fail closed when either artifact is missing.
    archive_path, proof_path = Path(archive_path).resolve(), Path(proof_path).resolve()
    proof_bytes = proof_path.read_bytes()
    proof = json.loads(proof_bytes.decode("utf-8"))
    expected = {
        "geometry_id": geometry_id,
        "structure_sha256": structure_sha256.lower(),
        "library_manifest_sha256": library_sha256.lower(),
        "layout_manifest_sha256": layout_sha256.lower(),
        "method_audit_sha256": method_audit_sha256.lower(),
        "occupancy_schema": OCCUPANCY_SCHEMA,
    }
    if proof.get("status") != "passed" or any(proof.get(key) != value for key, value in expected.items()):
        raise ValueError(f"geometry cache proof is stale or not passed: {geometry_id}")
    if proof.get("archive_sha256") != sha256_file(archive_path):
        raise ValueError(f"geometry cache archive checksum mismatch: {geometry_id}")
    with np.load(archive_path, allow_pickle=False) as data:
        arrays = {key: np.asarray(data[key]) for key in data.files}
    query_ids = [str(value) for value in arrays.get("query_ids", [])]
    rotation_hashes = [str(value) for value in arrays.get("rotation_sha256", [])]
    rows = proof.get("queries")
    if not isinstance(rows, list) or query_ids != [row.get("query_id") for row in rows]:
        raise ValueError(f"geometry cache query order mismatch: {geometry_id}")
    if rotation_hashes != [row.get("rotation_sha256") for row in rows]:
        raise ValueError(f"geometry cache rotation binding mismatch: {geometry_id}")
    candidate = arrays.get("candidate_masks")
    reference = arrays.get("reference_masks")
    expected_shape = (len(rows), *MASK_SHAPE)
    if (candidate is None or reference is None or candidate.shape != expected_shape
            or reference.shape != expected_shape or candidate.dtype != np.uint8
            or reference.dtype != np.uint8):
        raise ValueError(f"geometry cache mask arrays are malformed: {geometry_id}")
    for index, row in enumerate(rows):
        candidate_hash = hashlib.sha256(np.ascontiguousarray(candidate[index]).tobytes()).hexdigest()
        reference_hash = hashlib.sha256(np.ascontiguousarray(reference[index]).tobytes()).hexdigest()
        mismatch = float(np.count_nonzero(candidate[index] != reference[index]) / candidate[index].size)
        if (candidate_hash != row.get("mask_sha256")
                or reference_hash != row.get("reference_mask_sha256")
                or mismatch != row.get("max_disagreement_fraction")
                or mismatch >= float(row.get("acceptance_threshold", 0.0))):
            raise ValueError(f"geometry cache per-query audit failed: {geometry_id}/{row.get('query_id')}")
    return arrays, proof


def write_scene_occupancy(
    root: Path, *, scene: dict, geometry_entries: Mapping[str, dict],
    query_masks: Mapping[str, np.ndarray], query_proofs: Mapping[str, dict],
    method_audit_sha256: str, threshold: float,
) -> dict:
    if (not np.isfinite(threshold) or threshold <= 0
            or len(method_audit_sha256) != 64):
        raise ValueError("scene occupancy threshold or method-audit digest is invalid")
    int(method_audit_sha256, 16)
    objects = scene.get("objects")
    if not isinstance(objects, list) or not objects:
        raise ValueError("scene must contain object placements")
    masks = []
    audits = []
    for index, obj in enumerate(objects):
        identifier = str(obj["library_id"])
        entry = geometry_entries[identifier]
        rotation_hash = rotation_sha256(obj["rotation"])
        query_id = hashlib.sha256(json.dumps(
            [identifier, str(entry["structure_sha256"]).lower(), rotation_hash],
            separators=(",", ":")).encode("utf-8")).hexdigest()
        mask = np.asarray(query_masks[query_id], dtype=np.uint8)
        proof = query_proofs[query_id]
        if (mask.shape != MASK_SHAPE or proof.get("status") != "passed"
                or proof.get("rotation_sha256") != rotation_hash
                or not isinstance(proof.get("max_disagreement_fraction"), (int, float))
                or isinstance(proof.get("max_disagreement_fraction"), bool)
                or not np.isfinite(proof["max_disagreement_fraction"])
                or proof["max_disagreement_fraction"] < 0
                or proof["max_disagreement_fraction"] >= threshold):
            raise ValueError(f"scene {scene.get('scene_id')} has a missing/failed rotation query")
        if not np.isin(mask, (0, 1)).all():
            raise ValueError("scene occupancy mask must be binary")
        mask_hash = hashlib.sha256(np.ascontiguousarray(mask).tobytes()).hexdigest()
        if mask_hash != proof.get("mask_sha256"):
            raise ValueError("scene mask bytes do not match the geometry query proof")
        masks.append(mask)
        audits.append({
            "object_index": index, "library_id": identifier,
            "structure_sha256": str(entry["structure_sha256"]).lower(),
            "rotation_sha256": rotation_hash, "mask_sha256": mask_hash,
            "reference_mask_sha256": proof["reference_mask_sha256"],
            "mismatch_count": int(proof["mismatch_count"]),
            "point_count": int(proof["point_count"]),
            "max_disagreement_fraction": float(proof["max_disagreement_fraction"]),
            "acceptance_threshold": float(threshold),
            "method_audit_sha256": method_audit_sha256.lower(),
            "occupancy_schema": OCCUPANCY_SCHEMA,
            "status": "passed",
        })
    scene_id = str(scene["scene_id"])
    layout_sha = str(scene["layout_sha256"]).lower()
    archive = Path(root) / "scenes" / f"occupancy_{int(scene['index']):05d}.npz"
    proof_path = Path(root) / "proofs" / f"occupancy_{int(scene['index']):05d}.json"
    expected_masks = np.stack(masks).astype(np.uint8)
    proof_payload = {
        "schema": SCENE_PROOF_SCHEMA, "status": "passed", "scene_id": scene_id,
        "layout_sha256": layout_sha, "occupancy_schema": OCCUPANCY_SCHEMA,
        "method_audit_sha256": method_audit_sha256.lower(),
        "acceptance_threshold": float(threshold), "objects": audits,
        "max_disagreement_fraction": max(row["max_disagreement_fraction"] for row in audits),
    }
    if archive.exists() or proof_path.exists():
        if not archive.is_file():
            raise ValueError("scene occupancy proof exists without its archive")
        with np.load(archive, allow_pickle=False) as data:
            existing_masks = np.asarray(data["geometry_occupancy"])
        if not np.array_equal(existing_masks, expected_masks):
            raise ValueError("existing scene occupancy masks conflict with current geometry cache")
        archive_sha = sha256_file(archive)
        proof_payload["archive_sha256"] = archive_sha
        if proof_path.exists():
            existing_proof = json.loads(proof_path.read_text(encoding="utf-8"))
            if existing_proof != proof_payload:
                raise ValueError("existing scene occupancy proof conflicts with current source artifacts")
            proof_sha = sha256_file(proof_path)
        else:
            proof_sha = _atomic_json(proof_path, proof_payload)
    else:
        archive_sha = _atomic_npz(archive, geometry_occupancy=expected_masks)
        proof_payload["archive_sha256"] = archive_sha
        proof_sha = _atomic_json(proof_path, proof_payload)
    return {
        "index": int(scene["index"]), "scene_id": scene_id,
        "layout_sha256": layout_sha,
        "archive_path": archive.relative_to(Path(root)).as_posix(),
        "archive_sha256": archive_sha,
        "proof_path": proof_path.relative_to(Path(root)).as_posix(),
        "proof_sha256": proof_sha,
    }


def write_occupancy_index(
    path: Path, *, library_sha256: str, layout_sha256: str,
    method_screen_sha256: str, method_gate_sha256: str,
    threshold: float, entries: list[dict],
) -> str:
    if len(entries) != 10_000:
        raise ValueError("the production V6 occupancy index must contain exactly 10,000 scenes")
    for digest in (library_sha256, layout_sha256, method_screen_sha256, method_gate_sha256):
        if len(digest) != 64:
            raise ValueError("occupancy index source bindings must be SHA-256 digests")
        int(digest, 16)
    if not np.isfinite(threshold) or threshold <= 0:
        raise ValueError("occupancy index threshold must be finite and positive")
    scene_ids: set[str] = set()
    for index, row in enumerate(entries):
        if (row.get("index") != index or not row.get("archive_sha256")
                or not row.get("proof_sha256") or not row.get("scene_id")
                or row["scene_id"] in scene_ids):
            raise ValueError(f"occupancy index has an incomplete/out-of-order row at {index}")
        scene_ids.add(row["scene_id"])
    payload = {
        "schema": OCCUPANCY_INDEX_SCHEMA, "status": "passed",
        "library_manifest_sha256": library_sha256.lower(),
        "layout_manifest_sha256": layout_sha256.lower(),
        "method_screen_sha256": method_screen_sha256.lower(),
        "method_gate_sha256": method_gate_sha256.lower(),
        "occupancy_schema": OCCUPANCY_SCHEMA,
        "acceptance_threshold": float(threshold),
        "scene_count": len(entries), "entries": entries,
    }
    return _atomic_json(Path(path), payload)


def validate_grouped_plan(scenes: list[dict], geometry_entries: Mapping[str, dict]) -> dict:
    plan = plan_grouped_occupancy_queries(scenes)
    for group in plan["groups"]:
        entry = geometry_entries.get(group["geometry_id"])
        if entry is None or str(entry.get("structure_sha256", "")).lower() != group["structure_sha256"]:
            raise ValueError(f"occupancy query plan geometry/hash mismatch for {group['geometry_id']}")
        seen: set[str] = set()
        for query in group["queries"]:
            if query["rotation_sha256"] in seen:
                raise ValueError("grouped plan contains a duplicate rotation query")
            seen.add(query["rotation_sha256"])
    return plan
