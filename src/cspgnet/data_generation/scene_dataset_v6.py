"""Fail-closed V6 scene payload assembly and resumable split writer.

This module joins already-computed scene labels with geometry/occupancy
evidence.  It performs no CAD, FEM, BEM, or label solve.  Production callers
must supply an approved, source-bound release gate and per-object occupancy
proofs before any sample can be committed.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from . import formal_neumann_3d as formal
from . import structural_response


SCHEMA = "cspgnet-neumann-structural-smooth-v6"
OCCUPANCY_SCHEMA = "canonical-rays-r0.9-128-midpoints-v1"
RADIAL_R = 0.9 * (np.arange(128, dtype=np.float64) + 0.5) / 128.0
EXPECTED_COUNTS = {"train": 10_000, "val": 1_000}
REQUIRED_FIELDS = {
    "Z", "u_total_gt", "u_inc", "u_scattered_gt", "q64_gt",
    "probe_values", "probe_modes", "phi_pinv", "interface_points",
    "interface_faces", "interface_weights", "centers", "rotations", "shape_codes",
    "shape_names", "shape_parameters", "incident_direction", "k",
    "object_count", "schema_version", "spherical_harmonic_convention",
    "sample_seed", "geometry_occupancy", "geometry_occupancy_schema",
    "geometry_occupancy_radii", "geometry_genus", "geometry_ids",
}


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


def _atomic_json(path: Path, value: dict) -> None:
    pending = path.with_name(path.name + ".pending")
    if pending.exists():
        raise FileExistsError(f"refusing to overwrite pending JSON transaction: {pending}")
    with pending.open("xb") as stream:
        stream.write((json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False,
                                 allow_nan=False) + "\n").encode("utf-8"))
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(pending, path)


def _atomic_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    pending = path.with_name(path.name + ".pending")
    if pending.exists():
        raise FileExistsError(f"refusing to overwrite pending sample transaction: {pending}")
    with pending.open("xb") as stream:
        np.savez_compressed(stream, **arrays)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(pending, path)


def _npz_matches(path: Path, arrays: Mapping[str, np.ndarray]) -> bool:
    with np.load(path, allow_pickle=False) as opened:
        if set(opened.files) != set(arrays):
            return False
        return all(np.array_equal(opened[key], np.asarray(value)) for key, value in arrays.items())


def _complex64_checked(name: str, value: Any, expected_shape: tuple[int, ...]) -> np.ndarray:
    source = np.asarray(value)
    if source.shape != expected_shape or not np.iscomplexobj(source):
        raise ValueError(f"{name} must be complex with shape {expected_shape}; got {source.shape}")
    source128 = np.asarray(source, dtype=np.complex128)
    if not np.isfinite(source128).all():
        raise FloatingPointError(f"{name} contains non-finite values")
    converted = source128.astype(np.complex64)
    relative = float(np.linalg.norm(converted.astype(np.complex128) - source128)
                     / max(np.linalg.norm(source128), 1e-30))
    if relative > 1e-6:
        raise FloatingPointError(f"{name} complex64 conversion error {relative:.3e} exceeds 1e-6")
    return converted


def _validate_occupancy_proof(scene: dict, geometry_entries: Mapping[str, dict],
                              occupancy: np.ndarray, proof: dict) -> dict:
    count = int(scene["object_count"])
    value = np.asarray(occupancy)
    if value.shape != (count, 128, 128) or value.dtype != np.uint8:
        raise ValueError("geometry_occupancy must be uint8 [N,128,128]")
    if not np.isin(value, (0, 1)).all():
        raise ValueError("geometry_occupancy must be binary")
    if not isinstance(proof, dict) or proof.get("status") != "passed":
        raise ValueError("a passed occupancy accuracy audit is required before sample assembly")
    rows = proof.get("objects")
    objects = scene.get("objects")
    if not isinstance(rows, list) or len(rows) != count or len(objects) != count:
        raise ValueError("occupancy proof must contain one audit row per scene object")
    normalized = []
    for index, (obj, audit) in enumerate(zip(objects, rows)):
        identifier = str(obj["library_id"])
        entry = geometry_entries.get(identifier)
        if entry is None:
            raise KeyError(f"missing geometry-library entry {identifier!r}")
        structure_hash = str(entry.get("structure_sha256", "")).lower()
        rotation = np.asarray(obj["rotation"], dtype=np.float64)
        rotation_hash = hashlib.sha256(np.ascontiguousarray(rotation, dtype="<f8").tobytes()).hexdigest()
        if (not isinstance(audit, dict) or audit.get("status") != "passed"
                or str(audit.get("structure_sha256", "")).lower() != structure_hash
                or str(audit.get("rotation_sha256", "")).lower() != rotation_hash
                or audit.get("occupancy_schema") != OCCUPANCY_SCHEMA):
            raise ValueError(f"occupancy audit is not bound to object {index} ({identifier}) and its rotation")
        error = audit.get("max_disagreement_fraction")
        threshold = audit.get("acceptance_threshold")
        if (isinstance(error, bool) or not isinstance(error, (int, float)) or not np.isfinite(error)
                or error < 0 or isinstance(threshold, bool) or not isinstance(threshold, (int, float))
                or not np.isfinite(threshold) or threshold <= 0 or error >= threshold):
            raise ValueError(f"occupancy audit for {identifier} does not meet its registered numeric gate")
        method_hash = audit.get("method_audit_sha256")
        if (not isinstance(method_hash, str) or len(method_hash) != 64
                or any(char not in "0123456789abcdefABCDEF" for char in method_hash)):
            raise ValueError(f"missing method-audit SHA-256 for {identifier}")
        mask_hash = hashlib.sha256(np.ascontiguousarray(value[index]).tobytes()).hexdigest()
        if str(audit.get("mask_sha256", "")).lower() != mask_hash:
            raise ValueError(f"occupancy audit is not bound to the saved mask for {identifier}")
        normalized.append({**audit, "object_index": index, "library_id": identifier,
                           "structure_sha256": structure_hash,
                           "rotation_sha256": rotation_hash,
                           "mask_sha256": mask_hash})
    return {"schema": "geometry-occupancy-scene-audit-v1", "status": "passed",
            "occupancy_schema": OCCUPANCY_SCHEMA, "objects": normalized,
            "max_disagreement_fraction": max(float(row["max_disagreement_fraction"]) for row in normalized)}


def assemble_scene_payload(
    scene: dict,
    geometry_entries: Mapping[str, dict],
    fields: Mapping[str, np.ndarray],
    *,
    label_metadata: dict,
    interface_points: np.ndarray,
    interface_faces: np.ndarray,
    probe_values: np.ndarray,
    phi_pinv: np.ndarray,
    occupancy: np.ndarray,
    occupancy_proof: dict,
) -> tuple[dict[str, np.ndarray], dict]:
    """Create a schema-checked NPZ payload and provenance record in memory."""
    objects = scene.get("objects")
    count = scene.get("object_count")
    if type(count) is not int or count not in (2, 3, 4) or not isinstance(objects, list) or len(objects) != count:
        raise ValueError("scene must contain 2, 3, or 4 objects")
    if (not isinstance(label_metadata, dict) or label_metadata.get("object_count") != count
            or label_metadata.get("q64_gt_type") != "derived_ZP_u_total_not_independent_flux"):
        raise ValueError("label metadata does not match the frozen V6 scene-label contract")
    solve_residual = label_metadata.get("linear_residual")
    if (isinstance(solve_residual, bool) or not isinstance(solve_residual, (int, float))
            or not np.isfinite(solve_residual) or solve_residual < 0 or solve_residual > 1e-10):
        raise ValueError("Foldy--Lax label residual is missing or exceeds 1e-10")
    if float(scene.get("k", np.nan)) != 2.0:
        raise ValueError("V6 scene labels are frozen to k=2")
    ids = [str(item.get("library_id", "")) for item in objects]
    if len(set(ids)) != count:
        raise ValueError("a scene may not repeat a base geometry")
    points = np.asarray(interface_points, dtype=np.float64)
    faces = np.asarray(interface_faces, dtype=np.int32)
    probes = np.asarray(probe_values, dtype=np.complex128)
    projection = np.asarray(phi_pinv, dtype=np.complex128)
    if points.shape != (128, 3) or faces.shape != (252, 3):
        raise ValueError("canonical surface must use the frozen 128-node/252-face mesh")
    expected_points, expected_faces = formal.canonical_surface(128, 1.5)
    if (np.max(np.abs(points - expected_points)) > 1e-8
            or not np.array_equal(faces, expected_faces)
            or faces.min() < 0 or faces.max() >= 128):
        raise ValueError("interface discretization differs from the frozen canonical surface")
    if not np.isfinite(points).all() or not np.allclose(np.linalg.norm(points, axis=1), 1.5, atol=5e-6, rtol=0):
        raise ValueError("interface points must be finite and lie on radius 1.5")
    if probes.shape != (128, 64) or projection.shape != (64, 128):
        raise ValueError("probe_values/phi_pinv dimensions must be [128,64]/[64,128]")
    expected_probes = formal.spherical_harmonic_matrix(points, formal.spherical_modes(7))
    if np.max(np.abs(probes - expected_probes)) > 2e-5:
        raise ValueError("probe_values do not match the frozen l<=7 harmonics")
    expected_projection = formal.weighted_pseudoinverse(
        expected_probes, formal.equal_area_weights(128, 1.5),
    )
    projection_error = float(np.linalg.norm(projection - expected_projection)
                             / max(np.linalg.norm(expected_projection), 1e-30))
    if projection_error > 2e-5:
        raise ValueError("phi_pinv differs from the frozen equal-area weighted projection")
    if np.max(np.abs(projection @ probes - np.eye(64))) > 2e-5:
        raise ValueError("phi_pinv is inconsistent with the saved 64-mode probe matrix")

    centers = np.asarray(scene["centers"], dtype=np.float64)
    rotations = np.asarray(scene["rotations"], dtype=np.float64)
    direction = np.asarray(scene["incident_direction"], dtype=np.float64)
    if centers.shape != (count, 3) or rotations.shape != (count, 3, 3) or direction.shape != (3,):
        raise ValueError("scene centers/rotations/direction have invalid dimensions")
    if not np.isfinite(centers).all() or not np.isfinite(rotations).all() or not np.isfinite(direction).all():
        raise FloatingPointError("scene geometry/incident direction contains non-finite values")
    if abs(float(np.linalg.norm(direction)) - 1.0) > 1e-5:
        raise ValueError("incident direction must be unit length")
    for rotation in rotations:
        if (np.max(np.abs(rotation @ rotation.T - np.eye(3))) > 1e-5
                or abs(float(np.linalg.det(rotation)) - 1.0) > 1e-5):
            raise ValueError("all object rotations must be in SO(3)")
    center_distance = np.linalg.norm(centers[:, None] - centers[None, :], axis=-1)
    np.fill_diagonal(center_distance, np.inf)
    if count > 1 and center_distance.min() < 3.6 - 1e-8:
        raise ValueError("scene centers violate the frozen minimum separation 3.6")

    expected = {
        "Z": (count, 128, 64), "u_total_gt": (count, 128),
        "u_inc": (count, 128), "u_scattered_gt": (count, 128),
        "q64_gt": (count, 128),
    }
    complex_fields = {key: _complex64_checked(key, fields[key], shape) for key, shape in expected.items()}
    total128 = np.asarray(fields["u_total_gt"], dtype=np.complex128)
    incident128 = np.asarray(fields["u_inc"], dtype=np.complex128)
    scattered128 = np.asarray(fields["u_scattered_gt"], dtype=np.complex128)
    scattered_error = float(np.linalg.norm(total128 - incident128 - scattered128)
                            / max(np.linalg.norm(scattered128), 1e-30))
    if scattered_error > 1e-10:
        raise ValueError(f"u_scattered_gt identity error {scattered_error:.3e}")
    q_recomputed = formal.project_flux(np.asarray(fields["Z"], dtype=np.complex128), total128, projection)
    q_error = float(np.linalg.norm(q_recomputed - np.asarray(fields["q64_gt"], dtype=np.complex128))
                    / max(np.linalg.norm(q_recomputed), np.linalg.norm(fields["q64_gt"]), 1e-30))
    if q_error > 1e-10:
        raise ValueError(f"q64_gt derived-ZP_u-total identity error {q_error:.3e}")

    proof = _validate_occupancy_proof(scene, geometry_entries, occupancy, occupancy_proof)
    genus, parameters, object_records = [], [], []
    for obj in objects:
        identifier = str(obj["library_id"])
        entry = geometry_entries[identifier]
        if int(obj.get("genus", -1)) != int(entry.get("genus", -2)):
            raise ValueError(f"scene/library genus mismatch for {identifier}")
        if type(entry.get("genus")) is not int or entry["genus"] not in (0, 1, 2, 3):
            raise ValueError(f"invalid audited genus for {identifier}")
        semiaxes = np.asarray(entry.get("semiaxes", entry.get("axes")), dtype=np.float64)
        if semiaxes.shape != (3,) or not np.isfinite(semiaxes).all() or np.any(semiaxes <= 0):
            raise ValueError(f"library entry {identifier} must provide positive three-axis semiaxes")
        for field in ("structure_sha256", "operator_sha256", "report_sha256"):
            token = str(entry.get(field, ""))
            if (len(token) != 64 or any(char not in "0123456789abcdefABCDEF" for char in token)):
                raise ValueError(f"library entry {identifier} has invalid {field}")
        genus.append(int(entry["genus"]))
        parameters.append(semiaxes)
        object_records.append({
            "library_id": identifier, "shape": "ellipsoid_bore",
            "size": semiaxes.tolist(), "center": np.asarray(obj["center"], float).tolist(),
            "rotation": np.asarray(obj["rotation"], float).tolist(),
            "genus": int(entry["genus"]),
            "structure_sha256": str(entry["structure_sha256"]).lower(),
            "operator_sha256": str(entry["operator_sha256"]).lower(),
            "report_sha256": str(entry["report_sha256"]).lower(),
            "parameters": entry.get("parameters"),
            "physical_volume": float(entry["physical_volume"]),
            "bounding_radius": float(entry["bounding_radius"]),
            "pointcloud_file": entry.get("pointcloud_file"),
            "pointcloud_sha256": entry.get("pointcloud_sha256"),
        })

    arrays: dict[str, np.ndarray] = {
        **complex_fields,
        "S": structural_response.scattering_response({
            **complex_fields,
            "interface_points": points.astype(np.float32),
            "phi_pinv": projection.astype(np.complex64),
            "k": np.asarray(2.0, dtype=np.float32),
        }),
        "probe_values": probes.astype(np.complex64),
        "probe_modes": np.asarray(formal.spherical_modes(7), dtype=np.int8),
        "phi_pinv": projection.astype(np.complex64),
        "interface_points": points.astype(np.float32), "interface_faces": faces,
        "interface_weights": formal.equal_area_weights(128, 1.5).astype(np.float64),
        "centers": centers.astype(np.float32), "rotations": rotations.astype(np.float32),
        "shape_codes": np.full(count, 6, dtype=np.int8),
        "shape_names": np.full(count, "ellipsoid_bore"),
        "shape_parameters": np.asarray(parameters, dtype=np.float32),
        "incident_direction": direction.astype(np.float32),
        "k": np.asarray(2.0, dtype=np.float32), "object_count": np.asarray(count, dtype=np.int32),
        "schema_version": np.asarray(SCHEMA),
        "spherical_harmonic_convention": np.asarray("ngsolve_no_condon_shortley_positive_m"),
        "sample_seed": np.asarray(int(scene["seeds"]["selection"]) % (2**63), dtype=np.int64),
        "geometry_occupancy": np.asarray(occupancy, dtype=np.uint8),
        "geometry_occupancy_schema": np.asarray(OCCUPANCY_SCHEMA),
        "geometry_occupancy_radii": RADIAL_R.copy(),
        "geometry_genus": np.asarray(genus, dtype=np.int8),
        "geometry_ids": np.asarray(ids, dtype="U64"),
    }
    if arrays["S"].shape != (count, 128, 64) or arrays["S"].dtype != np.complex64:
        raise AssertionError("saved-Z scattering response contract failed")
    if not REQUIRED_FIELDS.issubset(arrays):
        raise AssertionError("assembled V6 payload is missing required fields")
    if not all(np.isfinite(value).all() for value in arrays.values() if np.issubdtype(value.dtype, np.number)):
        raise FloatingPointError("assembled V6 sample contains non-finite values")

    scene_digest = str(scene.get("layout_sha256", "")).lower()
    if (len(scene_digest) != 64
            or any(char not in "0123456789abcdef" for char in scene_digest)):
        raise ValueError("layout_sha256 must bind the physical scene")
    record = {
        "index": int(scene["index"]), "scene_id": str(scene["scene_id"]),
        "physical_scene_sha256": scene_digest, "layout_sha256": scene_digest,
        "k": 2.0, "object_count": count, "seeds": scene["seeds"],
        "incident_direction": direction.tolist(), "centers": centers.tolist(),
        "rotations": rotations.tolist(), "objects": object_records,
        "genus_counts": {str(value): genus.count(value) for value in sorted(set(genus))},
        "solve": dict(label_metadata),
        "occupancy_audit": proof,
        "q64_gt_type": "derived_ZP64_u_total_not_independent_flux",
        "status": "labeled-unreleased",
        "max_complex64_cast_relative_error": max(
            float(np.linalg.norm(arrays[name].astype(np.complex128) - np.asarray(fields[name], np.complex128))
                  / max(np.linalg.norm(fields[name]), 1e-30)) for name in expected),
        "scattered_identity_relative_error_complex128": scattered_error,
        "q64_identity_relative_error_complex128": q_error,
    }
    return arrays, record


class SceneSplitWriterV6:
    """Single-writer, append-only split writer with batched manifest checkpoints.

    NPZ and per-sample JSON are committed for each scene; the large aggregate
    manifest defaults to one checkpoint per 1,000 scenes. On resume, the
    writer verifies all complete sample/sidecar pairs and safely adopts pairs
    committed after the last manifest checkpoint. A half-written pair fails
    closed for manual audit.
    """

    def __init__(self, root: Path, split: str, expected_count: int, *,
                 release_gate_path: Path, source_bindings: dict[str, str],
                 run_config: dict, resume: bool = False,
                 manifest_checkpoint_interval: int = 1_000):
        if split not in EXPECTED_COUNTS or expected_count != EXPECTED_COUNTS[split]:
            raise ValueError(f"V6 split counts are frozen: {EXPECTED_COUNTS}")
        if type(manifest_checkpoint_interval) is not int or manifest_checkpoint_interval < 1:
            raise ValueError("manifest checkpoint interval must be a positive integer")
        if not isinstance(run_config, dict):
            raise ValueError("run_config must be a JSON-compatible object")
        self.root = Path(root).resolve()
        self.split, self.expected_count = split, expected_count
        self.manifest_checkpoint_interval = manifest_checkpoint_interval
        gate_path = Path(release_gate_path).resolve()
        if not gate_path.is_file():
            raise FileNotFoundError(f"source-bound V6 release gate is required: {gate_path}")
        gate = json.loads(gate_path.read_text(encoding="utf-8"))
        if (gate.get("schema") != "structural-smooth-v6-scene-label-release-gate-v1"
                or gate.get("status") != "passed"
                or gate.get("label_generation_authorized") is not True):
            raise ValueError("V6 release gate is not passed; refusing all label writes")
        gate_threshold = gate.get("geometry_occupancy_acceptance_threshold")
        if (isinstance(gate_threshold, bool) or not isinstance(gate_threshold, (int, float))
                or not np.isfinite(gate_threshold) or gate_threshold <= 0):
            raise ValueError("V6 release gate must register the geometry occupancy accuracy threshold")
        configured_threshold = run_config.get("geometry_occupancy_acceptance_threshold")
        if configured_threshold != gate_threshold:
            raise ValueError("run config occupancy threshold differs from the source-bound release gate")
        self.occupancy_threshold = float(gate_threshold)
        gate_hash = sha256_file(gate_path)
        for key, value in source_bindings.items():
            if (not isinstance(key, str) or not isinstance(value, str) or len(value) != 64
                    or any(char not in "0123456789abcdefABCDEF" for char in value)):
                raise ValueError("source_bindings must map names to SHA-256 digests")
        self.manifest_path = self.root / "manifest.json"
        expected_identity = {
            "schema_version": SCHEMA, "split": split, "num_samples": expected_count,
            "mesh_strategy": "graded-local-FEM", "release_gate_path": str(gate_path),
            "release_gate_sha256": gate_hash, "source_bindings": dict(sorted(source_bindings.items())),
            "run_config": run_config,
            "manifest_checkpoint_interval": manifest_checkpoint_interval,
        }
        self.run_identity_sha256 = canonical_sha256(expected_identity)
        if resume:
            if not self.manifest_path.is_file():
                raise FileNotFoundError("resume requires an existing manifest")
            self.manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
            if any(self.manifest.get(key) != value for key, value in expected_identity.items()):
                raise ValueError("resume refused: frozen input identity differs")
            if self.manifest.get("status") not in {"incomplete", "validating"}:
                raise ValueError("resume refused: split is not an uncommitted partial run")
            if (self.root / "training_ready.json").exists():
                raise ValueError("resume refused: split already has a release marker")
            (self.root / "records").mkdir(exist_ok=True)
            self._recover_orphans()
        else:
            if self.root.exists():
                raise FileExistsError(f"V6 output root must be new: {self.root}")
            self.root.mkdir(parents=True)
            (self.root / "records").mkdir()
            self.manifest = {
                **expected_identity, "status": "incomplete", "samples": [],
                "contains_numerical_training_samples": True,
                "training_ready": False,
            }
            _atomic_json(self.manifest_path, self.manifest)
        sample_rows = self.manifest["samples"]
        self.rows = {int(row["index"]): row for row in sample_rows}
        if len(self.rows) != len(sample_rows):
            raise ValueError("manifest contains duplicate sample indices")
        self.pending_checkpoint = 0

    def _recover_orphans(self) -> None:
        rows = {int(row["index"]): row for row in self.manifest.get("samples", [])}
        sample_indices = {int(path.stem.split("_")[-1]) for path in self.root.glob("sample_*.npz")}
        record_indices = {int(path.stem.split("_")[-1]) for path in (self.root / "records").glob("sample_*.json")}
        if sample_indices != record_indices or any(i < 0 or i >= self.expected_count for i in sample_indices):
            raise ValueError("resume refused: incomplete or out-of-range sample/record transaction")
        for index in sorted(sample_indices):
            filename = f"sample_{index:05d}.npz"
            record_name = f"records/sample_{index:05d}.json"
            sample = self.root / filename
            sidecar = self.root / record_name
            record_value = json.loads(sidecar.read_text(encoding="utf-8"))
            actual_record_hash = sha256_file(sidecar)
            if (record_value.get("index") != index or record_value.get("file") != filename
                    or record_value.get("record_file") != record_name
                    or record_value.get("writer_run_identity_sha256") != self.run_identity_sha256
                    or sha256_file(sample) != record_value.get("sha256")):
                raise ValueError(f"resume refused: sample/sidecar identity mismatch at index {index}")
            row = {**record_value, "record_sha256": actual_record_hash}
            previous = rows.get(index)
            if previous is not None and (previous.get("sha256") != row["sha256"]
                                         or previous.get("record_sha256") != actual_record_hash
                                         or previous.get("physical_scene_sha256") != row.get("physical_scene_sha256")):
                raise ValueError(f"resume refused: committed manifest row differs at index {index}")
            rows[index] = row
        physical = [row.get("physical_scene_sha256") for row in rows.values()]
        if len(set(physical)) != len(physical):
            raise ValueError("resume refused: duplicate physical scene IDs in committed rows")
        self.manifest["samples"] = [rows[index] for index in sorted(rows)]
        self.manifest["labeled_samples"] = len(rows)
        _atomic_json(self.manifest_path, self.manifest)
        self.rows = rows

    def commit(self, arrays: Mapping[str, np.ndarray], record: dict) -> dict:
        index = record.get("index")
        if type(index) is not int or not 0 <= index < self.expected_count:
            raise ValueError("sample index is outside the frozen split")
        if not REQUIRED_FIELDS.issubset(arrays):
            raise ValueError(f"sample payload lacks fields {sorted(REQUIRED_FIELDS.difference(arrays))}")
        if ("occupancy_audit" not in record or record["occupancy_audit"].get("status") != "passed"
                or len(record["occupancy_audit"].get("objects", [])) != int(record.get("object_count", -1))):
            raise ValueError("sample record is missing its object-bound passed occupancy audit")
        mask = np.asarray(arrays["geometry_occupancy"])
        if mask.dtype != np.uint8 or mask.shape != (int(record["object_count"]), 128, 128):
            raise ValueError("sample occupancy must be uint8 [N,128,128]")
        if (not np.isin(mask, (0, 1)).all()
                or str(np.asarray(arrays["geometry_occupancy_schema"]).item()) != OCCUPANCY_SCHEMA):
            raise ValueError("sample occupancy mask/schema is invalid")
        if not np.array_equal(np.asarray(arrays["geometry_occupancy_radii"], dtype=np.float64), RADIAL_R):
            raise ValueError("sample occupancy radii differ from the frozen radial grid")
        scene_hash = record.get("physical_scene_sha256")
        if not isinstance(scene_hash, str) or len(scene_hash) != 64:
            raise ValueError("sample record needs a physical_scene_sha256")
        geometry_ids = [str(value) for value in np.asarray(arrays["geometry_ids"]).tolist()]
        object_records = record.get("objects")
        proofs = record["occupancy_audit"]["objects"]
        if (not isinstance(object_records, list) or len(geometry_ids) != len(object_records)
                or len(geometry_ids) != len(proofs)):
            raise ValueError("sample objects, IDs, and occupancy proofs disagree")
        for object_index, (identifier, item, proof) in enumerate(zip(geometry_ids, object_records, proofs)):
            rotation = np.asarray(item.get("rotation"), dtype="<f8")
            if (item.get("library_id") != identifier or rotation.shape != (3, 3)
                    or proof.get("library_id") != identifier
                    or str(proof.get("structure_sha256", "")).lower()
                    != str(item.get("structure_sha256", "")).lower()):
                raise ValueError(f"sample object/proof identity mismatch at object {object_index}")
            rotation_hash = hashlib.sha256(np.ascontiguousarray(rotation).tobytes()).hexdigest()
            mask_hash = hashlib.sha256(np.ascontiguousarray(mask[object_index]).tobytes()).hexdigest()
            if proof.get("rotation_sha256") != rotation_hash or proof.get("mask_sha256") != mask_hash:
                raise ValueError(f"sample occupancy proof is not bound to object {object_index}'s rotation/mask")
            threshold, error = proof.get("acceptance_threshold"), proof.get("max_disagreement_fraction")
            if (proof.get("status") != "passed" or proof.get("occupancy_schema") != OCCUPANCY_SCHEMA
                    or isinstance(threshold, bool) or not isinstance(threshold, (int, float))
                    or not np.isfinite(threshold) or threshold <= 0
                    or float(threshold) != self.occupancy_threshold
                    or isinstance(error, bool) or not isinstance(error, (int, float))
                    or not np.isfinite(error) or error < 0 or error >= threshold):
                raise ValueError(f"sample occupancy audit gate failed for object {object_index}")
            method_hash = str(proof.get("method_audit_sha256", ""))
            if len(method_hash) != 64 or any(char not in "0123456789abcdef" for char in method_hash.lower()):
                raise ValueError(f"sample occupancy method receipt hash invalid for object {object_index}")
        filename = f"sample_{index:05d}.npz"
        record_name = f"records/sample_{index:05d}.json"
        output, record_path = self.root / filename, self.root / record_name
        previous = self.rows.get(index)
        if previous is not None:
            saved_record = json.loads(record_path.read_text(encoding="utf-8"))
            same_record = all(saved_record.get(key) == value for key, value in record.items())
            if (previous.get("physical_scene_sha256") != scene_hash
                    or sha256_file(output) != previous["sha256"]
                    or sha256_file(record_path) != previous["record_sha256"]
                    or not same_record or not _npz_matches(output, arrays)):
                raise ValueError(f"refusing to overwrite committed sample {index}")
            return previous
        if any(row.get("physical_scene_sha256") == scene_hash for row in self.rows.values()):
            raise ValueError("duplicate physical scene in V6 split")
        if output.exists() or record_path.exists():
            raise FileExistsError(f"orphan output exists for sample {index}; refusing overwrite")
        _atomic_npz(output, arrays)
        committed = {**record, "file": filename, "sha256": sha256_file(output),
                     "record_file": record_name, "status": "labeled-unreleased",
                     "writer_run_identity_sha256": self.run_identity_sha256}
        _atomic_json(record_path, committed)
        committed["record_sha256"] = sha256_file(record_path)
        self.rows[index] = committed
        self.pending_checkpoint += 1
        if (self.pending_checkpoint >= self.manifest_checkpoint_interval
                or len(self.rows) == self.expected_count):
            self.checkpoint()
        return committed

    def checkpoint(self) -> None:
        self.manifest["samples"] = [self.rows[index] for index in sorted(self.rows)]
        self.manifest["labeled_samples"] = len(self.rows)
        _atomic_json(self.manifest_path, self.manifest)
        self.pending_checkpoint = 0

    def mark_payload_scan_passed(self, receipt: dict) -> None:
        if len(self.rows) != self.expected_count:
            raise ValueError("cannot validate an incomplete split")
        self.checkpoint()
        if receipt.get("status") != "passed" or receipt.get("scene_count") != self.expected_count:
            raise ValueError("full payload scan receipt must pass and cover the exact split")
        manifest_hash = sha256_file(self.manifest_path)
        if receipt.get("manifest_sha256") != manifest_hash:
            raise ValueError("payload scan receipt is not bound to the current split manifest")
        self.manifest["status"] = "complete"
        self.manifest["payload_scan_receipt"] = receipt
        self.manifest["training_ready"] = False
        self.manifest["contains_numerical_training_samples"] = True
        _atomic_json(self.manifest_path, self.manifest)

