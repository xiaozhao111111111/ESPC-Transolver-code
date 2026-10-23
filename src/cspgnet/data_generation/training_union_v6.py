"""Fail-closed, manifest-only merge for the 10k legacy + 10k V6 train sets.

This creates no sample payloads and never rewrites legacy files. A completed
union manifest is not a training-release certificate; full payload, label,
feature, and normalization checks remain separate release gates.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import time
from datetime import datetime, timezone

import numpy as np


OLD_SCHEMA = "cspgnet-neumann-tmatrix-v2"
NEW_SCHEMA = "cspgnet-neumann-structural-smooth-v6"
TEST_SPLITS = {
    "id_test": "id_test",
    "n8_test": "n8_test",
    "shape_ood": "shape_ood",
    "topology_ood": "topology_ood",
    "shape_n8_ood": "shape_n8_ood",
    "topology_n8_ood": "topology_n8_ood",
}
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _safe_member_manifest(path: Path, root: Path, count: int, split: str, schema: str,
                          verify_files: bool, seen_physical: set[str]) -> dict:
    path = Path(path).resolve()
    root = Path(root).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    data = _json(path)
    if (data.get("status") != "complete" or data.get("split") != split
            or data.get("schema_version") != schema or data.get("num_samples") != count
            or len(data.get("samples", [])) != count):
        raise RuntimeError(f"{path}: split manifest is not the exact complete {schema} {split} set")
    records = sorted(data["samples"], key=lambda row: row.get("index", -1))
    if [row.get("index") for row in records] != list(range(count)):
        raise RuntimeError(f"{path}: sample indices must be contiguous 0..{count - 1}")
    files: set[str] = set()
    for row in records:
        name = row.get("file")
        token = row.get("sha256")
        physical = row.get("physical_scene_sha256")
        if not isinstance(name, str) or not name or name in files:
            raise RuntimeError(f"{path}: empty or duplicate sample filename")
        files.add(name)
        if not isinstance(token, str) or not _SHA256.fullmatch(token):
            raise RuntimeError(f"{path}: invalid sample checksum at index {row['index']}")
        if not isinstance(physical, str) or not _SHA256.fullmatch(physical):
            raise RuntimeError(f"{path}: missing physical-scene identity at index {row['index']}")
        if physical in seen_physical:
            raise RuntimeError(f"duplicate physical training scene at index {row['index']}")
        seen_physical.add(physical)
        sample = (root / name).resolve()
        if not sample.is_relative_to(root) or not sample.is_file():
            raise RuntimeError(f"{path}: sample path escapes split root or is missing: {name}")
        if verify_files and sha256_file(sample) != token:
            raise RuntimeError(f"{path}: sample checksum mismatch: {name}")
    return {"path": path, "manifest_sha256": sha256_file(path), "data": data}


def build_v6_training_union(project_root: Path, inventory_path: Path, old_train_manifest: Path,
                            new_train_manifest: Path, output_path: Path, *,
                            test_root: Path | None = None, new_validation_manifest: Path | None = None,
                            verify_files: bool = True, old_count: int = 10_000,
                            new_count: int = 10_000, validation_count: int = 1_000) -> dict:
    """Validate source manifests and return a reference-only 20k union manifest."""
    project_root = Path(project_root).resolve()
    inventory_path = Path(inventory_path).resolve()
    old_train_manifest = Path(old_train_manifest).resolve()
    new_train_manifest = Path(new_train_manifest).resolve()
    output_path = Path(output_path).resolve()
    inventory = _json(inventory_path)
    if inventory.get("status", "").find("passed") < 0:
        raise RuntimeError("legacy inventory has not passed its preflight")
    legacy_train = inventory.get("train", {})
    legacy_val = inventory.get("validation", {})
    if legacy_train.get("num_samples") != old_count or legacy_val.get("num_samples") != validation_count:
        raise RuntimeError("legacy inventory counts differ from the frozen merge protocol")
    if legacy_train.get("manifest_sha256") != sha256_file(old_train_manifest):
        raise RuntimeError("legacy training manifest differs from the frozen inventory")
    expected_legacy_path = (project_root / legacy_train["root"] / "manifest.json").resolve()
    if expected_legacy_path != old_train_manifest:
        raise RuntimeError("provided legacy train manifest is not the inventoried source")

    seen_physical: set[str] = set()
    old = _safe_member_manifest(old_train_manifest, old_train_manifest.parent, old_count,
                                "train", OLD_SCHEMA, verify_files, seen_physical)
    new_data = _json(new_train_manifest)
    if (new_data.get("schema_version") != NEW_SCHEMA
            or new_data.get("release_gate_sha256") is None
            or not _SHA256.fullmatch(str(new_data.get("release_gate_sha256")))):
        raise RuntimeError("new training manifest lacks the V6 schema or source-bound release-gate hash")
    if new_data.get("mesh_strategy") != "graded-local-FEM":
        raise RuntimeError("V6 training manifest is not explicitly bound to the graded local-FEM data path")
    gate_ref = Path(new_data.get("release_gate_path", ""))
    if not gate_ref.is_absolute():
        gate_ref = new_train_manifest.parent / gate_ref
    gate_ref = gate_ref.resolve()
    if not gate_ref.is_file() or sha256_file(gate_ref) != new_data["release_gate_sha256"]:
        raise RuntimeError("V6 release gate is missing or its source-bound hash does not match")
    gate_data = _json(gate_ref)
    if gate_data.get("status") != "passed":
        raise RuntimeError("V6 source-bound release gate has not passed")
    new = _safe_member_manifest(new_train_manifest, new_train_manifest.parent, new_count,
                                "train", NEW_SCHEMA, verify_files, seen_physical)
    unique_training_scene_hashes = len(seen_physical)

    validation_manifest = (project_root / legacy_val["root"] / "manifest.json").resolve()
    if sha256_file(validation_manifest) != legacy_val.get("manifest_sha256"):
        raise RuntimeError("legacy validation manifest changed")
    _safe_member_manifest(validation_manifest, validation_manifest.parent, validation_count,
                          "val", OLD_SCHEMA, verify_files, seen_physical)

    test_root = Path(test_root or (project_root / "datasets/canonical_response_v3/data")).resolve()
    tests = {}
    inventory_tests = inventory.get("tests", {})
    for key, dirname in TEST_SPLITS.items():
        expected = inventory_tests.get(key, {})
        manifest = (test_root / dirname / "manifest.json").resolve()
        if not manifest.is_file() or sha256_file(manifest) != expected.get("manifest_sha256"):
            raise RuntimeError(f"legacy test manifest changed or missing: {key}")
        count = expected.get("num_samples")
        checked_test = _safe_member_manifest(manifest, manifest.parent, count, key,
                                             "canonical-response-v3", verify_files, seen_physical)
        tests[key] = {"manifest": str(manifest), "sha256": checked_test["manifest_sha256"],
                      "scenes": count}

    validation_members = [{"name": "legacy_validation", "manifest": str(validation_manifest),
                           "sha256": sha256_file(validation_manifest), "scenes": validation_count}]
    if new_validation_manifest is not None:
        new_val_path = Path(new_validation_manifest).resolve()
        checked_new_val = _safe_member_manifest(new_val_path, new_val_path.parent, validation_count,
                                                "val", NEW_SCHEMA, verify_files, seen_physical)
        new_val_data = checked_new_val["data"]
        if (new_val_data.get("mesh_strategy") != "graded-local-FEM"
                or new_val_data.get("release_gate_sha256") != new_data["release_gate_sha256"]):
            raise RuntimeError("V6 validation is not bound to the same graded data-release gate as training")
        validation_members.append({"name": "v6_validation", "manifest": str(new_val_path),
                                   "sha256": checked_new_val["manifest_sha256"], "scenes": validation_count})

    manifest = {
        "schema_version": "canonical-response-v6-union",
        "status": "complete",
        "split": "train",
        "dataset_version": "structural-smooth-v6",
        "num_samples": old_count + new_count,
        "members": [
            {"name": "legacy_graded_train", "manifest": str(old["path"]),
             "sha256": old["manifest_sha256"], "scenes": old_count,
             "schema_version": OLD_SCHEMA},
            {"name": "structural_smooth_v6_train", "manifest": str(new["path"]),
             "sha256": new["manifest_sha256"], "scenes": new_count,
             "schema_version": NEW_SCHEMA},
        ],
        "validation_members": validation_members,
        "test_members": tests,
        "legacy_inventory": {"path": str(inventory_path), "sha256": sha256_file(inventory_path)},
        "v6_release_gate_sha256": new_data["release_gate_sha256"],
        "v6_release_gate_path": str(gate_ref),
        "unique_physical_scene_hashes": unique_training_scene_hashes,
        "audited_unique_scene_hashes_all_splits": len(seen_physical),
        "source_sha256": {
            "training_union_module": sha256_file(Path(__file__)),
            "scene_index_reader": sha256_file(Path(__file__).resolve().parents[1] / "structural_dataset.py"),
        },
        "legacy_validation_and_six_tests_unchanged": True,
        "sample_payloads_copied": False,
        "training_ready": False,
        "required_next_gate": "full-20k-payload-and-training-input-integrity-scan",
    }
    return manifest


def write_union_manifest(manifest: dict, output_path: Path) -> None:
    output_path = Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(manifest, sort_keys=True, indent=2) + "\n").encode("utf-8")
    if output_path.exists():
        if output_path.read_bytes() != encoded:
            raise FileExistsError(f"refusing to replace a different union manifest: {output_path}")
        return
    pending = output_path.with_name(output_path.name + ".pending")
    with pending.open("xb") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(pending, output_path)


def audit_legacy_dataset_payload_hashes(project_root: Path, inventory_path: Path,
                                        test_root: Path | None = None) -> dict:
    """Read-only hash audit of the existing train/val/six test splits."""
    project_root = Path(project_root).resolve()
    inventory_path = Path(inventory_path).resolve()
    inventory = _json(inventory_path)
    if "passed" not in inventory.get("status", ""):
        raise RuntimeError("legacy inventory did not pass its manifest/file preflight")
    test_root = Path(test_root or (project_root / "datasets/canonical_response_v3/data")).resolve()
    split_specs = []
    for split, entry in (("train", inventory["train"]), ("val", inventory["validation"])):
        manifest = (project_root / entry["root"] / "manifest.json").resolve()
        split_specs.append((split, manifest, entry["num_samples"], OLD_SCHEMA, entry["manifest_sha256"]))
    for split, dirname in TEST_SPLITS.items():
        entry = inventory["tests"][split]
        manifest = (test_root / dirname / "manifest.json").resolve()
        split_specs.append((split, manifest, entry["num_samples"], "canonical-response-v3", entry["manifest_sha256"]))

    failures: list[dict] = []
    started = time.perf_counter()
    split_reports: dict[str, dict] = {}
    identities: dict[str, str] = {}
    verified_files = 0
    total_bytes = 0
    for split, manifest_path, expected_count, expected_schema, inventory_hash in split_specs:
        report = {"manifest": str(manifest_path), "expected_scenes": int(expected_count),
                  "sample_files_verified": 0, "sample_bytes_verified": 0}
        split_reports[split] = report
        if not manifest_path.is_file():
            failures.append({"split": split, "error": "manifest missing", "path": str(manifest_path)})
            continue
        manifest_hash = sha256_file(manifest_path)
        report["manifest_sha256"] = manifest_hash
        if manifest_hash != inventory_hash:
            failures.append({"split": split, "error": "manifest hash differs from frozen inventory"})
        try:
            data = _json(manifest_path)
        except Exception as error:
            failures.append({"split": split, "error": f"manifest parse failed: {error}"})
            continue
        records = data.get("samples", [])
        if (data.get("status") != "complete" or data.get("split") != split
                or data.get("schema_version") != expected_schema
                or data.get("num_samples") != expected_count or len(records) != expected_count):
            failures.append({"split": split, "error": "manifest schema/status/count mismatch"})
            continue
        if [r.get("index") for r in records] != list(range(expected_count)):
            failures.append({"split": split, "error": "sample indices are not contiguous and ordered"})
        seen_files: set[str] = set()
        for row in records:
            index_value = row.get("index")
            name, token, physical = row.get("file"), row.get("sha256"), row.get("physical_scene_sha256")
            if not isinstance(name, str) or not name or name in seen_files:
                failures.append({"split": split, "index": index_value, "error": "empty/duplicate sample path"})
                continue
            seen_files.add(name)
            sample_path = (manifest_path.parent / name).resolve()
            if not sample_path.is_relative_to(manifest_path.parent.resolve()) or not sample_path.is_file():
                failures.append({"split": split, "index": index_value, "error": "sample missing or escapes split root"})
                continue
            if not isinstance(token, str) or not _SHA256.fullmatch(token):
                failures.append({"split": split, "index": index_value, "error": "invalid sample SHA-256"})
                continue
            actual_hash = sha256_file(sample_path)
            if actual_hash != token:
                failures.append({"split": split, "index": index_value, "error": "sample SHA-256 mismatch",
                                 "path": str(sample_path)})
                continue
            size = sample_path.stat().st_size
            report["sample_files_verified"] += 1
            report["sample_bytes_verified"] += size
            verified_files += 1
            total_bytes += size
            if not isinstance(physical, str) or not _SHA256.fullmatch(physical):
                failures.append({"split": split, "index": index_value, "error": "invalid physical scene SHA-256"})
            elif physical in identities:
                failures.append({"split": split, "index": index_value,
                                 "error": "physical scene duplicated across legacy splits",
                                 "prior_split": identities[physical]})
            else:
                identities[physical] = split

    return {
        "schema": "structural-smooth-v6-legacy-payload-hash-audit-v1",
        "status": "passed" if not failures else "failed",
        "checked_utc": datetime.now(timezone.utc).isoformat(),
        "legacy_inventory": str(inventory_path),
        "legacy_inventory_sha256": sha256_file(inventory_path),
        "splits": split_reports,
        "expected_sample_count": sum(spec[2] for spec in split_specs),
        "verified_sample_count": verified_files,
        "verified_bytes": total_bytes,
        "physical_scene_ids_unique_across_all_splits": not any(
            "physical scene duplicated" in row.get("error", "") for row in failures),
        "npz_arrays_deserialized": False,
        "legacy_files_modified": False,
        "source_sha256": {"training_union_module": sha256_file(Path(__file__))},
        "elapsed_seconds": time.perf_counter() - started,
        "failures": failures,
    }


def _relative_error(actual: np.ndarray, expected: np.ndarray) -> float:
    numerator = np.linalg.norm(np.asarray(actual) - np.asarray(expected))
    denominator = max(float(np.linalg.norm(actual)), float(np.linalg.norm(expected)), 1e-30)
    return float(numerator / denominator)


def validate_training_union_payloads(union_manifest: Path, expected_count: int = 20_000,
                                     q_relative_tolerance: float = 5e-5,
                                     expected_member_counts: dict[str, int] | None = None) -> dict:
    """Stream all scenes and validate fields/derived targets without releasing training."""
    from cspgnet.data_generation.formal_neumann_3d import project_flux
    from cspgnet.structural_dataset import UnifiedSceneIndex, radial_occupancy

    union_manifest = Path(union_manifest).resolve()
    index = UnifiedSceneIndex(union_manifest, verify_files=True)
    if len(index) != expected_count:
        raise RuntimeError(f"union has {len(index)} scenes, expected {expected_count}")
    members = {row["name"]: row for row in index.manifest["members"]}
    expected_names = {"legacy_graded_train", "structural_smooth_v6_train"}
    if set(members) != expected_names:
        raise RuntimeError("union members do not match the frozen old+V6 two-source protocol")
    if sum(int(row["scenes"]) for row in members.values()) != expected_count:
        raise RuntimeError("union member counts do not sum to expected scene count")
    if expected_member_counts is None:
        expected_member_counts = {"legacy_graded_train": 10_000, "structural_smooth_v6_train": 10_000}
    if {name: int(members[name]["scenes"]) for name in expected_names} != expected_member_counts:
        raise RuntimeError("union members do not have the exact predeclared 10k+10k counts")
    if members["legacy_graded_train"].get("schema_version") != OLD_SCHEMA:
        raise RuntimeError("legacy union member schema changed")
    if members["structural_smooth_v6_train"].get("schema_version") != NEW_SCHEMA:
        raise RuntimeError("V6 union member schema changed")
    gate_path = Path(index.manifest.get("v6_release_gate_path", ""))
    if not gate_path.is_absolute():
        gate_path = union_manifest.parent / gate_path
    gate_path = gate_path.resolve()
    expected_gate_hash = str(index.manifest.get("v6_release_gate_sha256", "")).lower()
    if not gate_path.is_file() or sha256_file(gate_path) != expected_gate_hash:
        raise RuntimeError("V6 occupancy/release gate is missing or its source hash changed")
    gate = _json(gate_path)
    if (gate.get("schema") != "structural-smooth-v6-scene-label-release-gate-v1"
            or gate.get("status") != "passed"
            or gate.get("label_generation_authorized") is not True):
        raise RuntimeError("V6 source-bound gate does not authorize this payload scan")
    occupancy_threshold = gate.get("geometry_occupancy_acceptance_threshold")
    if (isinstance(occupancy_threshold, bool) or not isinstance(occupancy_threshold, (int, float))
            or not np.isfinite(occupancy_threshold) or occupancy_threshold <= 0):
        raise RuntimeError("V6 source-bound gate lacks a valid occupancy error threshold")

    started = time.perf_counter()
    counts = {name: 0 for name in expected_names}
    maximum = {"q64_relative_error": 0.0, "scattered_identity_relative_error": 0.0,
               "probe_projection_identity_abs": 0.0, "canonical_radius_abs_error": 0.0,
               "rotation_orthogonality_abs": 0.0, "rotation_det_abs_error": 0.0}
    for union_index in range(len(index)):
        data, record = index.load(union_index)
        member = record["source_member"]
        if member not in expected_names:
            raise RuntimeError(f"unexpected union member at scene {union_index}: {member}")
        counts[member] += 1
        z = np.asarray(data.get("Z"))
        if z.ndim != 3 or z.shape[1:] != (128, 64) or z.dtype != np.complex64:
            raise RuntimeError(f"scene {union_index}: Z must be complex64 [N,128,64]")
        n = z.shape[0]
        required_shapes = {
            "u_total_gt": (n, 128), "u_inc": (n, 128), "q64_gt": (n, 128),
            "probe_values": (128, 64), "phi_pinv": (64, 128), "interface_points": (128, 3),
            "centers": (n, 3), "rotations": (n, 3, 3), "shape_codes": (n,),
            "incident_direction": (3,), "interface_faces": (252, 3),
        }
        for key, shape in required_shapes.items():
            if key not in data or np.asarray(data[key]).shape != shape:
                raise RuntimeError(f"scene {union_index}: {key} must have shape {shape}")
        for key in ("u_total_gt", "u_inc", "q64_gt", "probe_values", "phi_pinv"):
            if np.asarray(data[key]).dtype != np.complex64:
                raise RuntimeError(f"scene {union_index}: {key} must be complex64")
        for key, value in data.items():
            value = np.asarray(value)
            if np.issubdtype(value.dtype, np.number) and not np.isfinite(value).all():
                raise FloatingPointError(f"scene {union_index}: non-finite numeric field {key}")
        if not np.isclose(float(np.asarray(data["k"])), 2.0, rtol=0, atol=1e-6):
            raise RuntimeError(f"scene {union_index}: wavenumber is not k=2")

        radii = np.linalg.norm(np.asarray(data["interface_points"], dtype=np.float64), axis=1)
        radius_error = float(np.max(np.abs(radii - 1.5)))
        maximum["canonical_radius_abs_error"] = max(maximum["canonical_radius_abs_error"], radius_error)
        if radius_error > 2e-5:
            raise RuntimeError(f"scene {union_index}: interface points are not on radius-1.5 sphere")
        probe_identity = np.asarray(data["phi_pinv"], np.complex128) @ np.asarray(data["probe_values"], np.complex128)
        probe_error = float(np.max(np.abs(probe_identity - np.eye(64))))
        maximum["probe_projection_identity_abs"] = max(maximum["probe_projection_identity_abs"], probe_error)
        if probe_error > 1e-4:
            raise RuntimeError(f"scene {union_index}: stored weighted projection is inconsistent with probes")

        direction = np.asarray(data["incident_direction"], dtype=np.float64)
        if abs(float(np.linalg.norm(direction)) - 1.0) > 1e-5:
            raise RuntimeError(f"scene {union_index}: incident direction is not unit length")
        rotations = np.asarray(data["rotations"], dtype=np.float64)
        for rotation in rotations:
            orth_error = float(np.max(np.abs(rotation @ rotation.T - np.eye(3))))
            det_error = abs(float(np.linalg.det(rotation)) - 1.0)
            maximum["rotation_orthogonality_abs"] = max(maximum["rotation_orthogonality_abs"], orth_error)
            maximum["rotation_det_abs_error"] = max(maximum["rotation_det_abs_error"], det_error)
            if orth_error > 1e-4 or det_error > 1e-4:
                raise RuntimeError(f"scene {union_index}: stored object rotation is not in SO(3)")

        if member == "structural_smooth_v6_train":
            if "u_scattered_gt" not in data or np.asarray(data["u_scattered_gt"]).shape != (n, 128):
                raise RuntimeError(f"scene {union_index}: V6 scene lacks u_scattered_gt [N,128]")
            if "geometry_occupancy" not in data:
                raise RuntimeError(f"scene {union_index}: V6 scene lacks explicit topology-agnostic occupancy")
            occupancy_audit = record.get("occupancy_audit")
            geometry_ids = [str(value) for value in np.asarray(data.get("geometry_ids", [])).tolist()]
            object_records = record.get("objects")
            if (not isinstance(occupancy_audit, dict) or occupancy_audit.get("status") != "passed"
                    or len(occupancy_audit.get("objects", [])) != n
                    or len(geometry_ids) != n or not isinstance(object_records, list)
                    or len(object_records) != n):
                raise RuntimeError(f"scene {union_index}: occupancy provenance does not cover every V6 object")
            masks = np.asarray(data["geometry_occupancy"])
            if (masks.shape != (n, 128, 128) or masks.dtype != np.uint8
                    or not np.isin(masks, (0, 1)).all()
                    or str(np.asarray(data.get("geometry_occupancy_schema", "")).item())
                    != "canonical-rays-r0.9-128-midpoints-v1"):
                raise RuntimeError(f"scene {union_index}: V6 occupancy array/schema is invalid")
            expected_weights = np.full(128, 4.0 * np.pi * 1.5**2 / 128, dtype=np.float64)
            if ("interface_weights" not in data
                    or not np.array_equal(np.asarray(data["interface_weights"], dtype=np.float64),
                                          expected_weights)):
                raise RuntimeError(f"scene {union_index}: V6 canonical interface weights are missing or changed")
            genera = np.asarray(data.get("geometry_genus", []), dtype=np.int64)
            if genera.shape != (n,) or np.any((genera < 0) | (genera > 3)):
                raise RuntimeError(f"scene {union_index}: V6 per-object genus array is invalid")
            proof_rows = occupancy_audit["objects"]
            for object_index, (geometry_id, object_record, proof) in enumerate(
                    zip(geometry_ids, object_records, proof_rows)):
                if (not isinstance(proof, dict) or proof.get("status") != "passed"
                        or object_record.get("library_id") != geometry_id
                        or proof.get("library_id") != geometry_id
                        or int(object_record.get("genus", -1)) != int(genera[object_index])
                        or proof.get("occupancy_schema") != "canonical-rays-r0.9-128-midpoints-v1"
                        or str(object_record.get("structure_sha256", "")).lower()
                        != str(proof.get("structure_sha256", "")).lower()):
                    raise RuntimeError(f"scene {union_index}: occupancy proof geometry binding failed at object {object_index}")
                threshold, error = proof.get("acceptance_threshold"), proof.get("max_disagreement_fraction")
                if (isinstance(threshold, bool) or not isinstance(threshold, (int, float))
                        or not np.isfinite(threshold) or threshold <= 0
                        or float(threshold) != float(occupancy_threshold)
                        or isinstance(error, bool) or not isinstance(error, (int, float))
                        or not np.isfinite(error) or error < 0 or error >= threshold):
                    raise RuntimeError(f"scene {union_index}: occupancy numeric gate failed at object {object_index}")
                rotation = np.asarray(object_record.get("rotation"), dtype="<f8")
                if rotation.shape != (3, 3):
                    raise RuntimeError(f"scene {union_index}: missing object rotation in occupancy provenance")
                rotation_hash = hashlib.sha256(np.ascontiguousarray(rotation).tobytes()).hexdigest()
                if proof.get("rotation_sha256") != rotation_hash:
                    raise RuntimeError(f"scene {union_index}: occupancy proof rotation binding failed at object {object_index}")
                mask_hash = hashlib.sha256(np.ascontiguousarray(masks[object_index]).tobytes()).hexdigest()
                if proof.get("mask_sha256") != mask_hash:
                    raise RuntimeError(f"scene {union_index}: occupancy proof mask binding failed at object {object_index}")
                method_hash = str(proof.get("method_audit_sha256", ""))
                if len(method_hash) != 64 or any(char not in "0123456789abcdef" for char in method_hash.lower()):
                    raise RuntimeError(f"scene {union_index}: invalid occupancy method-audit hash at object {object_index}")
        if "u_scattered_gt" in data:
            scattered = np.asarray(data["u_total_gt"], np.complex128) - np.asarray(data["u_inc"], np.complex128)
            identity_error = _relative_error(data["u_scattered_gt"], scattered)
            maximum["scattered_identity_relative_error"] = max(maximum["scattered_identity_relative_error"], identity_error)
            if identity_error > 5e-5:
                raise RuntimeError(f"scene {union_index}: scattered field is inconsistent with total minus incident")
        if "S" in data and (np.asarray(data["S"]).shape != (n, 128, 64)
                            or np.asarray(data["S"]).dtype != np.complex64):
            raise RuntimeError(f"scene {union_index}: cached S must be complex64 [N,128,64]")

        occupancy = radial_occupancy(data)
        if occupancy.shape != (n, 128, 128) or not np.isfinite(occupancy).all():
            raise RuntimeError(f"scene {union_index}: geometry occupancy contract failed")
        q_expected = project_flux(z.astype(np.complex128),
                                  np.asarray(data["u_total_gt"], np.complex128),
                                  np.asarray(data["phi_pinv"], np.complex128))
        q_error = _relative_error(q_expected, np.asarray(data["q64_gt"], np.complex128))
        maximum["q64_relative_error"] = max(maximum["q64_relative_error"], q_error)
        if q_error > q_relative_tolerance:
            raise RuntimeError(f"scene {union_index}: derived q64 mismatch {q_error:.3e}")
        if (union_index + 1) % 500 == 0:
            print(json.dumps({"payload_scan": union_index + 1, "expected": len(index)}), flush=True)

    for name, member in members.items():
        if counts[name] != int(member["scenes"]):
            raise RuntimeError(f"union iterator count mismatch for {name}")
    return {
        "schema": "canonical-response-v6-full-union-payload-audit-v1",
        "status": "passed",
        "union_manifest": str(union_manifest),
        "union_manifest_sha256": sha256_file(union_manifest),
        "scene_count": len(index),
        "member_counts": counts,
        "member_manifest_sha256": {name: members[name]["sha256"] for name in sorted(members)},
        "v6_release_gate_sha256": index.manifest.get("v6_release_gate_sha256"),
        "geometry_occupancy_acceptance_threshold": float(occupancy_threshold),
        "verified_sample_file_sha256": True,
        "npz_payloads_streamed_and_schema_checked": True,
        "q64_semantics": "derived-ZP64u-total",
        "maximum_errors": maximum,
        "source_sha256": {
            "training_union_module": sha256_file(Path(__file__)),
            "scene_index_reader": sha256_file(Path(__file__).resolve().parents[1] / "structural_dataset.py"),
        },
        "training_ready": False,
        "seconds": time.perf_counter() - started,
    }
