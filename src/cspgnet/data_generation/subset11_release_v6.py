"""Subset-11-specific, fail-closed label release gate and plan builder.

This adapter deliberately does not weaken the generic 500-geometry V6 gate.
It validates the separate eleven-body protocol, its complete FEM/BEM/truncation
evidence, and every saved train/calibration occupancy proof before writing the
same writer-compatible V6 gate schema.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from . import scene_generation_v6 as generation
from .occupancy_index_v6 import load_geometry_cache, rotation_sha256


PROJECT = Path(__file__).resolve().parents[3]
ROOT_DEFAULT = PROJECT / "outputs/v6s11_20261008"
THRESHOLD = 0.03


def _sha(path: Path) -> str:
    return generation.sha256_file(Path(path))


def _read(path: Path) -> dict:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _atomic_json(path: Path, value: dict) -> None:
    from .scene_dataset_v6 import _atomic_json as write_json
    write_json(Path(path), value)


def _safe(root: Path, relative: str, label: str) -> Path:
    # Non-strict canonicalization still resolves existing junctions/symlinks,
    # but avoids Windows strict-realpath access checks that can reject an
    # otherwise readable evidence file. Callers immediately open/hash it.
    canonical_root = Path(root).resolve()
    path = (canonical_root / relative).resolve()
    if not path.is_relative_to(canonical_root):
        raise ValueError(f"{label} escapes its evidence root")
    return path


def _check_bem(root: Path) -> tuple[dict, dict]:
    from audit_subset11_v6 import digest, token
    state = _read(root / "bem/state.json")
    cases_doc = _read(root / "bem/cases.json")
    provenance_path = root / "bem/provenance.json"
    provenance = _read(provenance_path)
    cases = cases_doc.get("cases", [])
    if (state.get("status") != "passed" or len(state.get("passed", {})) != 71
            or cases_doc.get("single_count") != 11 or cases_doc.get("multi_count") != 60
            or len(cases) != 71 or state.get("new_training_labels") != 0
            or state.get("production_released") is not False
            or provenance.get("schema") != "subset11-independent-bem-provenance-v1"):
        raise RuntimeError("independent BEM audit is incomplete or not unreleased")
    rows = []
    for case in cases:
        case_id = case["case_id"]
        report_path = root / "bem/cases" / case_id / "report.json"
        report_sha = digest(report_path)
        report = _read(report_path)
        identity = token({"case": case, "provenance_sha256": digest(provenance_path)})
        if (state["passed"].get(case_id) != report_sha or report.get("status") != "passed"
                or report.get("case_id") != case_id or report.get("identity_sha256") != identity
                or report.get("new_training_labels") != 0 or report.get("production_released") is not False
                or report.get("geometry_ids") != case.get("geometry_ids")):
            raise RuntimeError(f"BEM report/state/source identity mismatch: {case_id}")
        residuals = []
        for level, record in report.get("levels", {}).items():
            if (record.get("solve_status") != "converged"
                    or record.get("independent_of_local_dtn") is not True
                    or int(record.get("gmres_info", -1)) != 0):
                raise RuntimeError(f"BEM level was not an independent converged solve: {case_id}/{level}")
            residual = float(record.get("actual_weak_relative_residual", np.inf))
            residuals.append(residual)
            if not np.isfinite(residual) or residual > 1.0e-8:
                raise RuntimeError(f"BEM actual residual exceeds 1e-8: {case_id}/{level}")
            artifact = report_path.parent / f"field_h{level}.npz"
            if _sha(artifact) != record.get("artifact_sha256"):
                raise RuntimeError(f"BEM field artifact hash mismatch: {case_id}/{level}")
        self_error = float(report.get("bem_self_relative", np.inf))
        label_error = float(report.get("label_scattered_relative", np.inf))
        if (not np.isfinite(self_error) or self_error > 0.02
                or not np.isfinite(label_error) or label_error > 0.02
                or not residuals or not np.isfinite(float(report.get("finest_h", np.nan)))):
            raise RuntimeError(f"BEM convergence/label error gate failed: {case_id}")
        rows.append({
            "id": case_id, "kind": case["kind"], "geometry_ids": case["geometry_ids"],
            "report_sha256": report_sha, "finest_h": float(report["finest_h"]),
            "bem_self_relative": self_error, "label_scattered_relative": label_error,
            "maximum_actual_residual": max(residuals),
        })
    if sum(row["kind"] == "single" for row in rows) != 11 or sum(row["kind"] == "multi" for row in rows) != 60:
        raise RuntimeError("BEM report coverage is not 11 singles plus 60 multi-scenes")
    summary = {
        "schema": "subset11-independent-bem-summary-v1", "status": "passed",
        "scope": "11 single-body plus 60 multi-body independent Burton-Miller audits",
        "expected_case_count": 71, "single_count": 11, "multi_count": 60,
        "results": rows, "maximum_bem_self_relative": max(row["bem_self_relative"] for row in rows),
        "maximum_label_scattered_relative": max(row["label_scattered_relative"] for row in rows),
        "maximum_actual_weak_residual": max(row["maximum_actual_residual"] for row in rows),
        "bem_provenance_sha256": digest(provenance_path),
        "new_training_labels": 0, "production_released": False,
    }
    return summary, provenance


def _check_rotation_and_remesh(root: Path, state: dict) -> None:
    from audit_subset11_v6 import digest
    if (state.get("status") != "rotation-audit-complete-other-audits-required"
            or len(state.get("completed", {})) != 22):
        raise RuntimeError("rotation/remesh audit must contain all 22 passed cases")
    for key, row in state["completed"].items():
        report_path = root / row["report"]
        if digest(report_path) != row.get("sha256"):
            raise RuntimeError(f"rotation/remesh report hash mismatch: {key}")
        report = _read(report_path)
        checks = report.get("checks", {})
        fem, flux = report.get("fem", {}), report.get("flux", {})
        if (report.get("status") != "passed"
                or not checks.get("mesh_and_residual")
                or not checks.get("independent_flux_below_3pct")
                or int(fem.get("elements_nonpositive_volume", -1)) != 0
                or float(fem.get("max_original_system_relative_residual", np.inf)) > 1.0e-8
                or float(flux.get("maximum_projection_error", np.inf)) >= 0.03
                or float(report.get("B64_relative_covariance_error", np.inf)) >= 0.01
                or float(report.get("all_modal_relative_difference", np.inf)) >= 0.03):
            raise RuntimeError(f"rotation/remesh numerical gate failed: {key}")
        if report.get("phase") == "rotation" and not checks.get("B64_covariance_below_1pct"):
            raise RuntimeError(f"SO(3) covariance check is not passed: {key}")


def _check_library(root: Path) -> tuple[dict, dict[str, dict]]:
    from audit_subset11_v6 import digest, verify_sources
    subset = _read(root / "subset.json")
    verify_sources(subset)
    path = root / "geometry-inputs/geometry_library.json"
    library = _read(path)
    entries = library.get("entries", [])
    if (library.get("schema") != "subset11-formal-geometry-library-v1"
            or library.get("status") != "all-11-audited-not-released"
            or library.get("geometry_count") != 11 or len(entries) != 11
            or library.get("training_labels_created") != 0
            or library.get("production_released") is not False
            or library.get("subset_sha256") != digest(root / "subset.json")
            or library.get("bem_state_sha256") != digest(root / "bem/state.json")
            or library.get("truncation_summary_sha256") != digest(root / "truncation/summary.json")):
        raise RuntimeError("formal geometry library is incomplete, stale, or already released")
    by_id = {str(row["id"]): row for row in entries}
    if len(by_id) != 11:
        raise RuntimeError("geometry library IDs are not unique")
    expected_genera = {0: 5, 1: 5, 2: 1}
    actual_genera = {genus: sum(int(row["genus"]) == genus for row in entries) for genus in expected_genera}
    if actual_genera != expected_genera:
        raise RuntimeError(f"subset topology coverage changed: {actual_genera}")
    for relative, expected in library.get("source_sha256", {}).items():
        if digest(PROJECT / relative) != expected:
            raise RuntimeError(f"geometry-library source changed: {relative}")
    for identifier, entry in by_id.items():
        if (entry.get("topology", {}).get("passed") is not True
                or entry.get("local_operator", {}).get("passed") is not True
                or float(entry["local_operator"]["independent_flux_projection_error"]) >= 0.03
                or float(entry.get("bounding_radius", np.inf)) >= 0.9):
            raise RuntimeError(f"local/topology gate failed for {identifier}")
        for key in ("structure_sha256", "operator_sha256", "report_sha256"):
            value = str(entry.get(key, ""))
            if len(value) != 64 or any(ch not in "0123456789abcdef" for ch in value.lower()):
                raise RuntimeError(f"invalid geometry {key}: {identifier}")
        for field, sha_field in (("operator_file", "operator_sha256"), ("report_file", "report_sha256")):
            artifact = Path(entry[field]).resolve()
            if digest(artifact) != entry[sha_field]:
                raise RuntimeError(f"geometry local artifact hash changed: {identifier}/{field}")
        cloud = _safe(root, entry["pointcloud_file"], f"point cloud {identifier}")
        if digest(cloud) != entry.get("pointcloud_sha256"):
            raise RuntimeError(f"point-cloud artifact SHA mismatch: {identifier}")
        with np.load(cloud, allow_pickle=False) as archive:
            points, normals = np.asarray(archive["points"]), np.asarray(archive["normals"])
            if (points.shape != (2048, 3) or normals.shape != (2048, 3)
                    or points.dtype != np.float32 or normals.dtype != np.float32
                    or not np.isfinite(points).all() or not np.isfinite(normals).all()
                    or not np.allclose(np.linalg.norm(normals, axis=1), 1.0, atol=1e-5)
                    or str(np.asarray(archive["structure_sha256"]).item()) != entry["structure_sha256"]):
                raise RuntimeError(f"invalid point-cloud contract: {identifier}")
    return library, by_id


def _check_truncation(root: Path) -> dict:
    summary = _read(root / "truncation/summary.json")
    state = _read(root / "truncation/state.json")
    if (summary.get("status") != "passed" or summary.get("geometry_count") != 11
            or summary.get("calibration_scene_count") != 1_000
            or summary.get("trace_scenes_passed") != 1_000
            or state.get("status") != "passed"
            or state.get("summary_sha256") != _sha(root / "truncation/summary.json")
            or state.get("training_labels_created") != 0
            or state.get("production_released") is not False):
        raise RuntimeError("full 1,000-scene input-truncation audit is not complete")
    if (float(summary.get("pooled_input64_relative_flux_p95", np.inf)) >= 0.05
            or float(summary.get("maximum_independent_full_flux_projection_error", np.inf)) >= 0.03
            or float(summary.get("maximum_direct_linear_system_residual", np.inf)) > 1.0e-8):
        raise RuntimeError("input-64 truncation, independent flux projection, or FEM residual gate failed")
    if len(summary.get("geometry_reports", [])) != 11:
        raise RuntimeError("truncation report does not cover all 11 geometries")
    for report in summary["geometry_reports"]:
        if (report.get("status") != "passed"
                or float(report.get("full_boundary_flux_projection_maximum", np.inf)) >= 0.03
                or float(report.get("max_original_system_relative_residual", np.inf)) > 1.0e-8):
            raise RuntimeError(f"per-geometry input truncation audit failed: {report.get('geometry_id')}")
    return summary


def _verify_occupancy_split(root: Path, *, library: dict, library_by_id: dict[str, dict],
                            layout_path: Path, index_path: Path,
                            cache_root: Path, expected_count: int,
                            expected_geometry_state: dict | None) -> dict:
    from audit_subset11_geometry_inputs_v6 import _layout_scene
    from audit_subset11_v6 import digest

    layout_doc = _read(layout_path)
    rows = layout_doc.get("scenes", [])
    if len(rows) != expected_count:
        raise RuntimeError(f"layout count mismatch: {layout_path}")
    scenes = [_layout_scene(row, library) for row in rows]
    scene_hashes = [row["layout_sha256"] for row in scenes]
    if len(set(scene_hashes)) != expected_count:
        raise RuntimeError(f"duplicate physical scenes in {layout_path.name}")
    library_sha = digest(root / "geometry-inputs/geometry_library.json")
    layout_sha = digest(layout_path)
    screen_path = root / "geometry-inputs/occupancy-screen.json"
    gate_path = root / "geometry-inputs/method_gate.json"
    screen = _read(screen_path)
    screen_sha, gate_sha = digest(screen_path), digest(gate_path)
    method_gate = _read(gate_path)
    selected_band = screen.get("selected_uncertainty_band")
    selected_screen_band = screen.get("bands", {}).get(str(selected_band), {})
    if (screen.get("schema") != "subset11-occupancy-screen-v1"
            or screen.get("status") != "passed-screen"
            or screen.get("production_authorized") is not False
            or screen.get("training_labels_created") != 0
            or len(screen.get("cases", [])) != 11
            or any(row.get("status") != "passed-screen" for row in screen["cases"])
            or selected_band is None
            or selected_screen_band.get("all_11_geometries_all_screen_rotations_pass") is not True
            or float(selected_screen_band.get("maximum_mismatch_fraction", np.inf)) >= THRESHOLD
            or screen.get("scope", {}).get("strict_mismatch_limit") != THRESHOLD
            or method_gate.get("schema") != "subset11-occupancy-method-gate-v1"
            or method_gate.get("status") != "passed"
            or method_gate.get("production_authorized") is not False
            or method_gate.get("method_screen_sha256") != screen_sha
            or method_gate.get("selected_uncertainty_band") != selected_band
            or method_gate.get("mesh_reference_h") != 0.01
            or method_gate.get("mesh_candidate_h") != 0.02
            or method_gate.get("curve_order") != 2
            or float(method_gate.get("acceptance_threshold", -1)) != THRESHOLD):
        raise RuntimeError("occupancy method gate is missing or differs from the strict 3% protocol")
    index = _read(index_path)
    if (index.get("schema") != generation.OCCUPANCY_INDEX_SCHEMA
            or index.get("status") != "passed" or index.get("scene_count") != expected_count
            or len(index.get("entries", [])) != expected_count
            or index.get("library_manifest_sha256") != library_sha
            or index.get("layout_manifest_sha256") != layout_sha
            or index.get("method_screen_sha256") != screen_sha
            or index.get("method_gate_sha256") != gate_sha
            or float(index.get("acceptance_threshold", -1)) != THRESHOLD):
        raise RuntimeError(f"occupancy index is stale or incomplete: {index_path}")
    if expected_geometry_state is not None:
        if (expected_geometry_state.get("status") != "passed"
                or expected_geometry_state.get("occupancy_index_sha256") != digest(index_path)
                or expected_geometry_state.get("completed_scene_count") != expected_count):
            raise RuntimeError("occupancy state/index completeness mismatch")

    cache_by_geometry: dict[str, dict[str, dict]] = {}
    expected_query_count = 0
    if cache_root.is_dir():
        for identifier in library_by_id:
            archive = cache_root / f"{identifier}.npz"
            proof_path = cache_root / f"{identifier}.json"
            arrays, proof = load_geometry_cache(
                archive, proof_path, geometry_id=identifier,
                structure_sha256=library_by_id[identifier]["structure_sha256"],
                library_sha256=library_sha, layout_sha256=layout_sha,
                method_audit_sha256=screen_sha,
            )
            query_rows = {row["query_id"]: row for row in proof["queries"]}
            if len(query_rows) != len(proof["queries"]) or proof.get("status") != "passed":
                raise RuntimeError(f"geometry occupancy cache is incomplete: {identifier}")
            cache_by_geometry[identifier] = query_rows
            expected_query_count += len(query_rows)
    else:
        raise RuntimeError(f"missing occupancy geometry cache directory: {cache_root}")

    seen_scene_ids = set()
    for expected_index, (scene, row) in enumerate(zip(scenes, index["entries"])):
        if (row.get("index") != expected_index or row.get("scene_id") != scene["scene_id"]
                or row.get("layout_sha256") != scene["layout_sha256"]
                or row["scene_id"] in seen_scene_ids):
            raise RuntimeError(f"occupancy index scene identity mismatch: {expected_index}")
        seen_scene_ids.add(row["scene_id"])
        # write_scene_occupancy stores relative to the split's ``occupancy``
        # directory, while the compact index lives one directory above it.
        split_artifact_root = index_path.parent / "occupancy"
        archive = _safe(split_artifact_root, row["archive_path"], "scene occupancy archive")
        proof_path = _safe(split_artifact_root, row["proof_path"], "scene occupancy proof")
        if _sha(archive) != row.get("archive_sha256") or _sha(proof_path) != row.get("proof_sha256"):
            raise RuntimeError(f"scene occupancy artifact checksum mismatch: {expected_index}")
        proof = _read(proof_path)
        if (proof.get("schema") != "geometry-occupancy-scene-audit-v1"
                or proof.get("status") != "passed" or proof.get("scene_id") != scene["scene_id"]
                or proof.get("layout_sha256") != scene["layout_sha256"]
                or proof.get("method_audit_sha256") != screen_sha
                or float(proof.get("acceptance_threshold", -1)) != THRESHOLD
                or proof.get("archive_sha256") != row["archive_sha256"]
                or len(proof.get("objects", [])) != scene["object_count"]):
            raise RuntimeError(f"scene occupancy proof identity mismatch: {expected_index}")
        with np.load(archive, allow_pickle=False) as npz:
            masks = np.asarray(npz["geometry_occupancy"])
            if (set(npz.files) != {"geometry_occupancy"}
                    or masks.shape != (scene["object_count"], 128, 128)
                    or masks.dtype != np.uint8 or not np.isin(masks, (0, 1)).all()):
                raise RuntimeError(f"invalid scene occupancy archive: {expected_index}")
        for object_index, (obj, object_proof) in enumerate(zip(scene["objects"], proof["objects"])):
            identifier = obj["library_id"]
            rotation_sha = rotation_sha256(obj["rotation"])
            query_id = hashlib.sha256(json.dumps(
                [identifier, library_by_id[identifier]["structure_sha256"].lower(), rotation_sha],
                separators=(",", ":")).encode("utf-8")).hexdigest()
            query = cache_by_geometry[identifier].get(query_id)
            mask_sha = hashlib.sha256(np.ascontiguousarray(masks[object_index]).tobytes()).hexdigest()
            if query is None or (object_proof.get("status") != "passed"
                    or object_proof.get("object_index") != object_index
                    or object_proof.get("library_id") != identifier
                    or object_proof.get("structure_sha256") != library_by_id[identifier]["structure_sha256"]
                    or object_proof.get("rotation_sha256") != rotation_sha
                    or object_proof.get("mask_sha256") != mask_sha
                    or object_proof.get("mask_sha256") != query.get("mask_sha256")
                    or object_proof.get("reference_mask_sha256") != query.get("reference_mask_sha256")
                    or object_proof.get("mismatch_count") != query.get("mismatch_count")
                    or object_proof.get("point_count") != 128 * 128
                    or object_proof.get("max_disagreement_fraction") != query.get("max_disagreement_fraction")
                    or object_proof.get("acceptance_threshold") != THRESHOLD
                    or object_proof.get("method_audit_sha256") != screen_sha
                    or float(object_proof.get("max_disagreement_fraction", np.inf)) >= THRESHOLD):
                raise RuntimeError(f"per-object occupancy proof mismatch: {scene['scene_id']}/{identifier}")
    return {
        "index_sha256": digest(index_path), "layout_sha256": layout_sha,
        "scene_count": expected_count, "unique_rotation_queries": expected_query_count,
        "scene_proofs_verified": expected_count,
    }


def build_subset11_release_gate(root: Path = ROOT_DEFAULT) -> dict:
    """Validate complete subset-11 evidence and atomically write a gate."""
    from audit_subset11_v6 import atomic_json, digest, token, verify_sources
    from audit_subset11_truncation_v6 import _require_prior_audits

    root = Path(root).resolve()
    rotation_state, bem_state = _require_prior_audits(root)
    _check_rotation_and_remesh(root, rotation_state)
    subset = _read(root / "subset.json")
    verify_sources(subset)
    truncation = _check_truncation(root)
    bem_summary, bem_provenance = _check_bem(root)
    library, library_by_id = _check_library(root)

    train_layout = root / "train-layout.json"
    cal_layout = root / "calibration.json"
    train_occ_index = root / "geometry-inputs/occupancy_index.json"
    cal_occ_index = root / "calibration-geometry-inputs/occupancy_index.json"
    train_geom_state = _read(root / "geometry-inputs/state.json")
    cal_geom_state = _read(root / "calibration-geometry-inputs/state.json")
    if (cal_geom_state.get("status") != "passed"
            or cal_geom_state.get("completed_geometry_count") != 11
            or cal_geom_state.get("completed_scene_count") != 1_000
            or cal_geom_state.get("training_labels_created") != 0
            or cal_geom_state.get("production_released") is not False):
        raise RuntimeError("independent calibration occupancy audit is not complete")
    if (int(train_geom_state.get("minimum_available_bytes", 0)) < 1024**3
            or int(cal_geom_state.get("minimum_available_bytes", 0)) < 1024**3):
        raise MemoryError("an occupancy audit observed system available memory below the 1-GiB reserve")
    cal_provenance_path = root / "calibration-geometry-inputs/provenance.json"
    cal_provenance = _read(cal_provenance_path)
    if (cal_provenance.get("status") != "passed"
            or cal_provenance.get("occupancy_index_sha256") != _sha(cal_occ_index)
            or cal_provenance.get("calibration_layout_sha256") != _sha(cal_layout)
            or cal_provenance.get("threshold") != THRESHOLD
            or cal_provenance.get("training_labels_created") != 0
            or cal_provenance.get("production_released") is not False):
        raise RuntimeError("calibration geometry-input provenance does not bind the passed artifacts")
    train_occ = _verify_occupancy_split(
        root, library=library, library_by_id=library_by_id, layout_path=train_layout,
        index_path=train_occ_index, cache_root=root / "geometry-inputs/geometry_cache",
        expected_count=10_000, expected_geometry_state=train_geom_state,
    )
    cal_occ = _verify_occupancy_split(
        root, library=library, library_by_id=library_by_id, layout_path=cal_layout,
        index_path=cal_occ_index, cache_root=root / "calibration-geometry-inputs/geometry_cache",
        expected_count=1_000, expected_geometry_state=cal_geom_state,
    )

    # Verify every saved local modal operator and its source report before
    # constructing the small index consumed by the bounded scene workers.
    operator_rows = []
    operator_specs = {}
    for identifier, entry in sorted(library_by_id.items()):
        modal_path = Path(entry["operator_file"]).resolve()
        report_path = Path(entry["report_file"]).resolve()
        modal_sha, report_sha = _sha(modal_path), _sha(report_path)
        if modal_sha != entry.get("operator_sha256") or report_sha != entry.get("report_sha256"):
            raise RuntimeError(f"local operator/report hash mismatch: {identifier}")
        with np.load(modal_path, allow_pickle=False) as archive:
            modal = np.asarray(archive["modal_dtn"])
        if modal.shape != (121, 100) or not np.iscomplexobj(modal) or not np.isfinite(modal).all():
            raise RuntimeError(f"local modal operator contract failed: {identifier}")
        generation.LocalModalOperator.from_modal_dtn(identifier, modal.astype(np.complex128), k=2.0, lmax=9)
        operator_rows.append({
            "geometry_id": identifier, "structure_sha256": entry["structure_sha256"],
            "operator_sha256": entry["operator_sha256"], "report_sha256": entry["report_sha256"],
            "modal_dtn_path": str(modal_path), "modal_dtn_file_sha256": modal_sha,
            "report_path": str(report_path),
        })
        operator_specs[identifier] = {"path": str(modal_path), "file_sha256": modal_sha,
                                      "operator_sha256": entry["operator_sha256"]}

    bem_summary_path = root / "bem/subset11_summary.json"
    if bem_summary_path.exists() and _read(bem_summary_path) != bem_summary:
        raise RuntimeError("retained derived BEM summary differs from the 71 verified reports")
    if not bem_summary_path.exists():
        atomic_json(bem_summary_path, bem_summary, immutable=True)
    operator_index_path = root / "generation/operator_index.json"
    operator_index = {
        "schema": "subset11-operator-artifact-index-v1", "status": "passed-not-released",
        "library_manifest_sha256": _sha(root / "geometry-inputs/geometry_library.json"),
        "operator_count": 11, "entries": operator_rows,
        "production_released": False, "training_labels_created": 0,
    }
    if operator_index_path.exists() and _read(operator_index_path) != operator_index:
        raise RuntimeError("retained subset-11 operator index differs from verified operators")
    if not operator_index_path.exists():
        atomic_json(operator_index_path, operator_index, immutable=True)

    # The generic V6 source-binding function expects the occupancy index's
    # parent to hold state.json and provenance.json. The passed audit already
    # owns state.json; create a deterministic derived provenance record here.
    geometry_provenance_path = root / "geometry-inputs/provenance.json"
    geometry_provenance = {
        "schema": "subset11-train-geometry-inputs-provenance-v1",
        "status": "passed", "identity_sha256": train_geom_state.get("identity_sha256"),
        "library_sha256": _sha(root / "geometry-inputs/geometry_library.json"),
        "layout_sha256": _sha(train_layout), "occupancy_index_sha256": _sha(train_occ_index),
        "occupancy_screen_sha256": _sha(root / "geometry-inputs/occupancy-screen.json"),
        "method_gate_sha256": _sha(root / "geometry-inputs/method_gate.json"),
        "completed_geometry_count": 11, "completed_scene_count": 10_000,
        "new_training_labels": 0, "production_released": False,
        "source_sha256": _sha(Path(__file__)),
    }
    if geometry_provenance_path.exists() and _read(geometry_provenance_path) != geometry_provenance:
        raise RuntimeError("retained training occupancy provenance differs from verified inputs")
    if not geometry_provenance_path.exists():
        atomic_json(geometry_provenance_path, geometry_provenance, immutable=True)

    # Recompute after emitting only the derived indexes/provenance. The gate
    # hash then binds every source the common V6 writer rechecks at runtime.
    from .scene_generation_v6 import _generation_source_bindings
    bindings = _generation_source_bindings(
        library_path=root / "geometry-inputs/geometry_library.json",
        operator_index_path=operator_index_path, layout_path=train_layout,
        occupancy_index_path=train_occ_index,
        occupancy_method_screen_path=root / "geometry-inputs/occupancy-screen.json",
        occupancy_method_gate_path=root / "geometry-inputs/method_gate.json",
        bem_summary_path=bem_summary_path, bem_provenance_path=root / "bem/provenance.json",
    )
    gate_payload = {
        "schema": generation.GATE_SCHEMA, "status": "passed",
        "label_generation_authorized": True,
        "geometry_occupancy_acceptance_threshold": THRESHOLD,
        "source_bindings": bindings,
        "evidence": {
            "protocol": "subset11-k2-Neumann; 5/5/1 genus 0/1/2; no genus-3 claim",
            "geometry_count": 11, "genus_counts": {"0": 5, "1": 5, "2": 1},
            "train_scene_count": 10_000, "calibration_scene_count": 1_000,
            "operator_count": 11, "bem_case_count": 71,
            "bem_summary_sha256": _sha(bem_summary_path),
            "bem_provenance_sha256": _sha(root / "bem/provenance.json"),
            "rotation_state_sha256": _sha(root / "state.json"),
            "truncation_summary_sha256": _sha(root / "truncation/summary.json"),
            "train_occupancy_index_sha256": train_occ["index_sha256"],
            "calibration_occupancy_index_sha256": cal_occ["index_sha256"],
            "calibration_occupancy_provenance_sha256": _sha(cal_provenance_path),
            "calibration_occupancy_query_plan_sha256": _sha(root / "calibration-geometry-inputs/query_plan.json"),
            "calibration_geometry_state_sha256": _sha(root / "calibration-geometry-inputs/state.json"),
            "subset11_release_validator_sha256": _sha(Path(__file__)),
            "subset11_generation_driver_sha256": _sha(PROJECT / "scripts/generate_subset11_data_v6.py"),
            "calibration_layout_sha256": cal_occ["layout_sha256"],
            "train_occupancy_scene_proofs_verified": train_occ["scene_proofs_verified"],
            "calibration_occupancy_scene_proofs_verified": cal_occ["scene_proofs_verified"],
            "train_unique_rotation_queries_verified": train_occ["unique_rotation_queries"],
            "calibration_unique_rotation_queries_verified": cal_occ["unique_rotation_queries"],
            "bem_label_error_limit": 0.02, "bem_adjacent_mesh_error_limit": 0.02,
            "bem_gmres_residual_limit": 1.0e-8,
            "local_independent_flux_error_limit_exclusive": 0.03,
            "input64_flux_truncation_p95_limit_exclusive": 0.05,
            "calibration_dataset_is_separate_from_training": True,
            "six_existing_test_sets_modified": False,
        },
    }
    gate_path = root / "generation/release_gate.json"
    if gate_path.exists():
        if _read(gate_path) != gate_payload:
            raise RuntimeError("existing release gate differs from the complete verified evidence")
    else:
        atomic_json(gate_path, gate_payload, immutable=True)
    return {**gate_payload, "release_gate_path": str(gate_path),
            "release_gate_sha256": _sha(gate_path)}

