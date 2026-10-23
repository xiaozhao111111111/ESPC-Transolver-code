"""Fail-closed, bounded-parallel V6 scene-label generation.

The command layer accepts only frozen 500-operator, 10k-layout, occupancy,
and release-gate artifacts. Each worker loads the operator bank once, computes
labels in memory, and returns one scene to a single parent writer. This module
never builds CAD/FEM/BEM operators and never marks a dataset training-ready.
"""
from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import dataclass
import hashlib
import json
from collections import Counter
import multiprocessing as mp
import os
from pathlib import Path
import platform
import scipy
import tempfile
import threading
import time
from typing import Any

import numpy as np

from . import formal_neumann_3d as formal
from . import tmatrix_neumann_3d as tmatrix
from .scene_dataset_v6 import (
    OCCUPANCY_SCHEMA,
    SceneSplitWriterV6,
    assemble_scene_payload,
    canonical_sha256,
    sha256_file,
)
from .scene_labels_v6 import (
    HarmonicRotationCache,
    LocalModalOperator,
    generate_scene_labels,
)
from .occupancy_index_v6 import (
    load_geometry_cache,
    select_method_gate,
    validate_grouped_plan,
    validate_method_gate,
)


LIBRARY_SCHEMA = "structural-smooth-v6-validated-library-v1"
LAYOUT_SCHEMA = "structural-scene-layout-v6"
OPERATOR_INDEX_SCHEMA = "structural-smooth-v6-operator-artifact-index-v1"
OCCUPANCY_INDEX_SCHEMA = "structural-smooth-v6-scene-occupancy-index-v1"
GATE_SCHEMA = "structural-smooth-v6-scene-label-release-gate-v1"
BEM_SUMMARY_SCHEMA = "structural-smooth-v6-five-case-independent-bem-calibration-v1"
BEM_PROVENANCE_SCHEMA = BEM_SUMMARY_SCHEMA
OCCUPANCY_METHOD_SCREEN_SCHEMA = "structural-smooth-v6-occupancy-hybrid-screen-v1"
GEOMETRY_QUOTAS = {0: 200, 1: 150, 2: 100, 3: 50}
SCENE_COUNT = 10_000
MAX_WORKERS = 4  # Only 2/4 workers have a throughput/RAM benchmark.
WORKER_COUNTS = {1, 2, 4}
START_FLOOR_BYTES = 6 * 1024**3
RESERVE_BYTES = 1024**3

_WORKER: dict[str, Any] | None = None


def available_physical_bytes() -> int:
    """Return currently available physical RAM (Windows and Linux)."""
    if os.name == "nt":
        import ctypes

        class MemoryStatusEx(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        status = MemoryStatusEx()
        status.dwLength = ctypes.sizeof(status)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            raise OSError("GlobalMemoryStatusEx failed")
        return int(status.ullAvailPhys)
    if Path("/proc/meminfo").is_file():
        fields = {}
        for line in Path("/proc/meminfo").read_text(encoding="ascii").splitlines():
            key, _, rest = line.partition(":")
            if key in {"MemAvailable", "MemFree"}:
                fields[key] = int(rest.split()[0]) * 1024
        if "MemAvailable" in fields:
            return fields["MemAvailable"]
    raise OSError("cannot determine available physical memory")


class _MemoryMonitor:
    def __init__(self, interval: float = 0.5):
        self.minimum = available_physical_bytes()
        self._interval = interval
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._sample, daemon=True)

    def _sample(self) -> None:
        while not self._stop.wait(self._interval):
            try:
                self.minimum = min(self.minimum, available_physical_bytes())
            except OSError:
                # A telemetry failure must not weaken the hard reserve check.
                self.minimum = min(self.minimum, 0)

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *_exc):
        self._stop.set()
        self._thread.join()
        self.minimum = min(self.minimum, available_physical_bytes())


def _json_with_sha(path: Path) -> tuple[dict, str]:
    raw = Path(path).read_bytes()
    value = json.loads(raw.decode("utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value, hashlib.sha256(raw).hexdigest()


def _validate_occupancy_method_evidence(screen_path: Path, gate_path: Path) -> tuple[dict, str, str]:
    screen, screen_sha = _json_with_sha(screen_path)
    gate, gate_sha = _json_with_sha(gate_path)
    if screen.get("schema") != OCCUPANCY_METHOD_SCREEN_SCHEMA:
        raise ValueError("occupancy method screen schema is not the registered V6 screen")
    expected = select_method_gate(screen, screen_sha)
    validate_method_gate(gate, gate_sha)
    if gate != expected:
        raise ValueError("occupancy method gate does not exactly match its source screen")
    return gate, screen_sha, gate_sha


def _validate_bem_calibration_summary(summary_path: Path) -> tuple[dict, str, Path, str]:
    """Verify the five-case independent BEM evidence before label release."""
    summary_path = Path(summary_path).resolve(strict=True)
    summary, summary_sha = _json_with_sha(summary_path)
    provenance_path = summary_path.with_name("provenance.json").resolve(strict=True)
    provenance, provenance_sha = _json_with_sha(provenance_path)
    expected_base = [
        "ellipsoid-g0-a", "ellipsoid-g1-a", "ellipsoid-g2-b", "ellipsoid-g3-a",
    ]
    allowed_fifth = {"ellipsoid-g3-b", "ellipsoid-g0-b"}
    case_ids = provenance.get("case_ids")
    rows = summary.get("results")
    if (summary.get("schema") != BEM_SUMMARY_SCHEMA or summary.get("status") != "passed"
            or summary.get("expected_cases") != 5 or summary.get("completed_cases") != 5
            or summary.get("scene_labels_generated") != 0
            or summary.get("production_released") is not False
            or summary.get("provenance_sha256") != provenance_sha
            or provenance.get("schema") != BEM_PROVENANCE_SCHEMA
            or provenance.get("scope") != "nonsealed-calibration-only; no scene labels; no production release"
            or not isinstance(case_ids, list) or len(case_ids) != 5
            or case_ids[:4] != expected_base or case_ids[4] not in allowed_fifth
            or not isinstance(rows, list) or len(rows) != 5):
        raise ValueError("independent BEM summary is incomplete, failed, or not the registered five-case calibration")
    thresholds = provenance.get("thresholds", {})
    if (thresholds.get("bem_adjacent_scattered_relative") != 0.02
            or thresholds.get("label_vs_converged_bem_scattered_relative") != 0.02
            or thresholds.get("gmres_relative_residual") != 1.0e-8):
        raise ValueError("BEM provenance does not use the frozen 2%/2%/1e-8 thresholds")
    physics = provenance.get("physics", {})
    if physics.get("k") != 2.0 or physics.get("canonical_radius") != 1.5:
        raise ValueError("BEM calibration physics do not match the V6 fixed-k canonical-sphere problem")

    project_root = Path(__file__).resolve().parents[3]
    source_files = (
        "scripts/calibrate_ellipsoid_bore_bem_v6.py",
        "scripts/pilot_ellipsoid_bore_8_cad_v6.py",
        "src/cspgnet/data_generation/structural_bem.py",
        "src/cspgnet/data_generation/formal_neumann_3d.py",
        "src/cspgnet/data_generation/tmatrix_neumann_3d.py",
    )
    recorded_sources = provenance.get("source_sha256")
    if not isinstance(recorded_sources, dict):
        raise ValueError("BEM provenance lacks source hashes")
    recorded_sources = {str(key).replace("\\", "/"): value
                        for key, value in recorded_sources.items()}
    for relative in source_files:
        expected_sha = recorded_sources.get(relative)
        source_path = project_root / relative
        if (not isinstance(expected_sha, str) or len(expected_sha) != 64
                or not source_path.is_file() or sha256_file(source_path) != expected_sha):
            raise ValueError(f"BEM calibration source hash is missing or stale: {relative}")

    if [row.get("id") for row in rows if isinstance(row, dict)] != case_ids:
        raise ValueError("BEM result rows do not match the exact provenance case order")
    genera: set[int] = set()
    for row in rows:
        if (row.get("status") != "passed" or row.get("production_released") is not False
                or row.get("provenance_sha256") != provenance_sha):
            raise ValueError(f"BEM case is not a passed non-release calibration: {row.get('id')}")
        genus = row.get("genus")
        if type(genus) is not int or genus not in GEOMETRY_QUOTAS:
            raise ValueError(f"invalid genus in BEM case {row.get('id')}")
        genera.add(genus)
        self_error = row.get("bem_self_convergence_scattered_relative")
        if (isinstance(self_error, bool) or not isinstance(self_error, (int, float))
                or not np.isfinite(self_error) or self_error > 0.02):
            raise ValueError(f"BEM adjacent-mesh scattered-field gate failed: {row.get('id')}")
        levels = row.get("bem_levels")
        if not isinstance(levels, dict) or len(levels) < 2:
            raise ValueError(f"BEM case lacks at least two converged mesh levels: {row.get('id')}")
        level_h = []
        for key, level in levels.items():
            if not isinstance(level, dict):
                raise ValueError(f"malformed BEM mesh-level record: {row.get('id')}/{key}")
            h = level.get("mesh_h")
            residual = level.get("actual_weak_relative_residual")
            field_sha = level.get("field_sha256")
            if (isinstance(h, bool) or not isinstance(h, (int, float)) or h <= 0
                    or level.get("gmres_info") != 0 or level.get("solve_status") != "converged"
                    or level.get("independent_of_local_dtn") is not True
                    or isinstance(residual, bool) or not isinstance(residual, (int, float))
                    or not np.isfinite(residual) or residual > 1.0e-8
                    or not isinstance(field_sha, str) or len(field_sha) != 64):
                raise ValueError(f"BEM residual/independence audit failed: {row.get('id')}/{key}")
            int(field_sha, 16)
            if not np.isclose(float(key), float(h), rtol=0, atol=1e-12):
                raise ValueError(f"BEM level key disagrees with recorded mesh_h: {row.get('id')}/{key}")
            field_path = (summary_path.parent / str(row["id"])
                          / f"bem_h{float(h):.2f}.npz").resolve(strict=True)
            if (summary_path.parent.resolve() not in field_path.parents
                    or sha256_file(field_path) != field_sha):
                raise ValueError(f"BEM field artifact is missing or checksum-invalid: {row.get('id')}/{key}")
            with np.load(field_path, allow_pickle=False) as archive:
                if "u_total" not in archive.files:
                    raise ValueError(f"BEM field artifact lacks u_total: {row.get('id')}/{key}")
                total = np.asarray(archive["u_total"])
            if (total.shape != (2, 128) or not np.iscomplexobj(total)
                    or not np.isfinite(total).all()):
                raise ValueError(f"BEM field artifact has invalid shape/type: {row.get('id')}/{key}")
            level_h.append(float(h))
        if len(set(level_h)) != len(level_h):
            raise ValueError(f"duplicate BEM mesh level in case {row.get('id')}")
        selected = row.get("selected_fem_profile")
        checks = row.get("label_checks")
        if selected not in {"G3", "G4"} or not isinstance(checks, dict):
            raise ValueError(f"BEM label comparison is missing for {row.get('id')}")
        selected_check = checks.get(selected, {})
        label_error = selected_check.get("label_vs_bem_scattered_relative")
        modal_cache = selected_check.get("modal_cache")
        if (isinstance(label_error, bool) or not isinstance(label_error, (int, float))
                or not np.isfinite(label_error) or label_error > 0.02):
            raise ValueError(f"selected FEM label does not pass the independent BEM gate: {row.get('id')}")
        if (not isinstance(modal_cache, dict)
                or modal_cache.get("duplicates_numerically_identical_to_relative_1e-12") is not True
                or type(modal_cache.get("duplicate_cache_count")) is not int
                or modal_cache["duplicate_cache_count"] < 1):
            raise ValueError(f"BEM-selected local modal cache provenance is incomplete: {row.get('id')}")
        cache_relative = modal_cache.get("path")
        cache_sha = modal_cache.get("sha256")
        if not isinstance(cache_relative, str) or not isinstance(cache_sha, str) or len(cache_sha) != 64:
            raise ValueError(f"BEM-selected modal cache reference is malformed: {row.get('id')}")
        int(cache_sha, 16)
        cache_path = (project_root / cache_relative).resolve(strict=True)
        if project_root not in cache_path.parents or sha256_file(cache_path) != cache_sha:
            raise ValueError(f"BEM-selected modal cache is outside the project or checksum-invalid: {row.get('id')}")
        with np.load(cache_path, allow_pickle=False) as archive:
            if "modal_dtn" not in archive.files:
                raise ValueError(f"BEM-selected modal cache lacks modal_dtn: {row.get('id')}")
            modal = np.asarray(archive["modal_dtn"])
        if (modal.shape != (121, 100) or not np.iscomplexobj(modal)
                or not np.isfinite(modal).all()):
            raise ValueError(f"BEM-selected modal cache is not finite complex [121,100]: {row.get('id')}")
    if genera != {0, 1, 2, 3}:
        raise ValueError("the five-case BEM audit must cover genera 0, 1, 2, and 3")
    return summary, summary_sha, provenance_path, provenance_sha


def _safe_artifact_path(root: Path, relative: Any, label: str) -> Path:
    if not isinstance(relative, str) or not relative:
        raise ValueError(f"{label} must be a non-empty relative path")
    raw = Path(relative)
    if raw.is_absolute() or any(part in {"..", ""} for part in raw.parts):
        raise ValueError(f"{label} must stay under its manifest directory")
    base = Path(root).resolve()
    resolved = (base / raw).resolve(strict=True)
    if base != resolved and base not in resolved.parents:
        raise ValueError(f"{label} escapes its manifest directory")
    if not resolved.is_file():
        raise FileNotFoundError(f"{label} is not a file: {resolved}")
    return resolved


def _rotation_sha(rotation: Any) -> str:
    value = np.asarray(rotation, dtype="<f8")
    if value.shape != (3, 3):
        raise ValueError("scene rotation must be [3,3]")
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def _validate_library(library: dict, library_sha: str) -> dict[str, dict]:
    if (library.get("schema") != LIBRARY_SCHEMA
            or library.get("status") != "all-500-local-passed-not-released"
            or library.get("production_released") is not False
            or library.get("new_labeled_training_samples") != 0):
        raise ValueError("input library is not the frozen, unreleased, all-500-passed V6 schema")
    entries = library.get("entries")
    if not isinstance(entries, list) or len(entries) != 500:
        raise ValueError("the V6 geometry library must contain exactly 500 entries")
    for key in ("source_sha256", "provenance_sha256"):
        token = library.get(key)
        if not isinstance(token, str) or len(token) != 64:
            raise ValueError(f"library {key} must be a SHA-256 digest")
        int(token, 16)
    by_id: dict[str, dict] = {}
    genera: dict[int, int] = {}
    for row in entries:
        if not isinstance(row, dict):
            raise ValueError("library entry must be a JSON object")
        identifier = row.get("id")
        genus = row.get("genus")
        if not isinstance(identifier, str) or not identifier or identifier in by_id:
            raise ValueError("library IDs must be non-empty and unique")
        if type(genus) is not int or genus not in GEOMETRY_QUOTAS:
            raise ValueError(f"invalid genus for {identifier}")
        topology = row.get("topology")
        if (not isinstance(topology, dict) or topology.get("passed") is not True
                or topology.get("genus") != genus
                or topology.get("euler_characteristic") != 2 - 2 * genus):
            raise ValueError(f"independent CAD topology audit is not passed for {identifier}")
        local = row.get("local_operator")
        if (not isinstance(local, dict) or local.get("passed") is not True
                or not isinstance(local.get("independent_flux_projection_error"), (float, int))
                or isinstance(local.get("independent_flux_projection_error"), bool)
                or not 0 <= float(local["independent_flux_projection_error"]) < 0.03):
            raise ValueError(f"local operator is not independently accepted (<3%) for {identifier}")
        semiaxes = np.asarray(row.get("semiaxes", row.get("axes")), dtype=np.float64)
        if semiaxes.shape != (3,) or not np.isfinite(semiaxes).all() or np.any(semiaxes <= 0):
            raise ValueError(f"missing ellipsoid semiaxes for {identifier}")
        for key in ("physical_volume", "bounding_radius"):
            value = row.get(key)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value) or value <= 0:
                raise ValueError(f"invalid {key} for {identifier}")
        if float(row["bounding_radius"]) >= 0.9:
            raise ValueError(f"bounding radius must be below 0.9 for {identifier}")
        for key in ("structure_sha256", "operator_sha256", "report_sha256"):
            token = row.get(key)
            if not isinstance(token, str) or len(token) != 64:
                raise ValueError(f"missing {key} for {identifier}")
            int(token, 16)
        by_id[identifier] = row
        genera[genus] = genera.get(genus, 0) + 1
    if genera != GEOMETRY_QUOTAS:
        raise ValueError(f"geometry genus quotas differ from {GEOMETRY_QUOTAS}: {genera}")
    if library.get("manifest_sha256") not in {None, library_sha}:
        raise ValueError("embedded library manifest hash does not match the file")
    return by_id


def _validate_layout(layout: dict, layout_sha: str, library_sha: str,
                     geometry_entries: dict[str, dict]) -> list[dict]:
    if (layout.get("schema") != LAYOUT_SCHEMA
            or layout.get("status") != "layout-only-not-labeled-not-released"
            or layout.get("scene_count") != SCENE_COUNT
            or layout.get("library_sha256") != library_sha
            or layout.get("contains_field_labels") is not False):
        raise ValueError("layout is not the frozen 10,000-scene plan for this exact library")
    if layout.get("object_count_scenes") != {"2": 3334, "3": 3333, "4": 3333}:
        raise ValueError("layout object-count allocation differs from 3334/3333/3333")
    scenes = layout.get("scenes")
    if not isinstance(scenes, list) or len(scenes) != SCENE_COUNT:
        raise ValueError("layout must contain exactly 10,000 scene records")
    physical_hashes: set[str] = set()
    scene_counts: Counter[int] = Counter()
    combinations: dict[int, set[tuple[str, ...]]] = {2: set(), 3: set(), 4: set()}
    for index, scene in enumerate(scenes):
        if (not isinstance(scene, dict) or scene.get("index") != index
                or scene.get("scene_id") != f"structural-train-v6-{index:05d}"):
            raise ValueError(f"layout scene order/ID mismatch at index {index}")
        count = scene.get("object_count")
        objects = scene.get("objects")
        if type(count) is not int or count not in (2, 3, 4) or not isinstance(objects, list) or len(objects) != count:
            raise ValueError(f"invalid object count at scene {index}")
        if scene.get("k") != 2.0 or len(set(str(item.get("library_id")) for item in objects)) != count:
            raise ValueError(f"invalid k or duplicate geometry in scene {index}")
        scene_counts[count] += 1
        combination = tuple(sorted(str(item["library_id"]) for item in objects))
        if combination in combinations[count]:
            raise ValueError(f"duplicate geometry combination within N={count} at scene {index}")
        combinations[count].add(combination)
        for object_index, item in enumerate(objects):
            identifier = item.get("library_id")
            geometry = geometry_entries.get(identifier)
            if (geometry is None or item.get("structure_sha256") != geometry["structure_sha256"]
                    or item.get("genus") != geometry["genus"]):
                raise ValueError(f"scene {index} object {object_index} does not match library")
            center = np.asarray(item.get("center"), dtype=np.float64)
            rotation = np.asarray(item.get("rotation"), dtype=np.float64)
            if center.shape != (3,) or rotation.shape != (3, 3) or not np.isfinite(center).all() or not np.isfinite(rotation).all():
                raise ValueError(f"invalid placement for {identifier} in scene {index}")
            if (not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-8, rtol=0)
                    or abs(float(np.linalg.det(rotation)) - 1.0) > 1e-8):
                raise ValueError(f"non-SO(3) rotation for {identifier} in scene {index}")
        centers = np.asarray(scene.get("centers"), dtype=np.float64)
        rotations = np.asarray(scene.get("rotations"), dtype=np.float64)
        direction = np.asarray(scene.get("incident_direction"), dtype=np.float64)
        if (centers.shape != (count, 3) or rotations.shape != (count, 3, 3)
                or not np.array_equal(centers, np.asarray([o["center"] for o in objects]))
                or not np.array_equal(rotations, np.asarray([o["rotation"] for o in objects]))
                or direction.shape != (3,) or not np.isfinite(direction).all()
                or abs(float(np.linalg.norm(direction)) - 1.0) > 1e-10):
            raise ValueError(f"scene placement arrays disagree at scene {index}")
        distances = np.linalg.norm(centers[:, None, :] - centers[None, :, :], axis=-1)
        np.fill_diagonal(distances, np.inf)
        if float(np.min(distances)) < 3.6 - 1e-10:
            raise ValueError(f"scene {index} violates minimum center distance")
        digest = scene.get("layout_sha256")
        if not isinstance(digest, str) or len(digest) != 64 or digest in physical_hashes:
            raise ValueError(f"missing or duplicate physical scene hash at scene {index}")
        physical = {
            "k": 2.0,
            "incident_direction": scene["incident_direction"],
            "objects": sorted(objects, key=lambda item: item["library_id"]),
        }
        if canonical_sha256(physical) != digest:
            raise ValueError(f"physical layout SHA-256 mismatch at scene {index}")
        physical_hashes.add(digest)
    if scene_counts != Counter({2: 3334, 3: 3333, 4: 3333}):
        raise ValueError(f"scene object-count quotas differ from 3334/3333/3333: {dict(scene_counts)}")
    return scenes


def _validate_operator_index(index: dict, index_sha: str, library_sha: str,
                             geometry_entries: dict[str, dict], index_path: Path) -> dict[str, dict]:
    if (index.get("schema") != OPERATOR_INDEX_SCHEMA
            or index.get("status") != "all-500-local-passed-not-released"
            or index.get("library_manifest_sha256") != library_sha):
        raise ValueError("operator artifact index is not bound to the accepted library")
    rows = index.get("entries")
    if not isinstance(rows, list) or len(rows) != 500:
        raise ValueError("operator artifact index must contain exactly 500 operators")
    result: dict[str, dict] = {}
    root = index_path.parent
    for row in rows:
        identifier = row.get("geometry_id") if isinstance(row, dict) else None
        geometry = geometry_entries.get(identifier)
        if geometry is None or identifier in result:
            raise ValueError(f"unknown or duplicate operator geometry ID: {identifier}")
        for key in ("structure_sha256", "operator_sha256", "report_sha256"):
            if row.get(key) != geometry.get(key):
                raise ValueError(f"operator index {key} mismatches library for {identifier}")
        modal_path = _safe_artifact_path(root, row.get("modal_dtn_path"), f"{identifier}.modal_dtn_path")
        report_path = _safe_artifact_path(root, row.get("report_path"), f"{identifier}.report_path")
        modal_sha = row.get("modal_dtn_file_sha256")
        if not isinstance(modal_sha, str) or len(modal_sha) != 64 or sha256_file(modal_path) != modal_sha:
            raise ValueError(f"modal artifact SHA-256 mismatch for {identifier}")
        if sha256_file(report_path) != geometry["report_sha256"]:
            raise ValueError(f"operator report SHA-256 mismatch for {identifier}")
        report, _ = _json_with_sha(report_path)
        local_error = float(geometry["local_operator"]["independent_flux_projection_error"])
        if (report.get("status") != "passed" or report.get("geometry_id") != identifier
                or report.get("structure_sha256") != geometry["structure_sha256"]
                or report.get("operator_sha256") != geometry["operator_sha256"]
                or report.get("independent_flux_projection_error") != local_error
                or local_error >= 0.03):
            raise ValueError(f"operator report semantics/gate mismatch for {identifier}")
        with np.load(modal_path, allow_pickle=False) as archive:
            if "modal_dtn" not in archive.files:
                raise ValueError(f"modal artifact lacks modal_dtn for {identifier}")
            modal = np.asarray(archive["modal_dtn"])
            if modal.shape != (121, 100) or not np.iscomplexobj(modal) or not np.isfinite(modal).all():
                raise ValueError(f"modal DtN must be finite complex [121,100] for {identifier}")
            # Detect non-convertible/near-singular modal operators before any
            # output split is created, not during the first worker startup.
            LocalModalOperator.from_modal_dtn(
                identifier, np.asarray(modal, dtype=np.complex128), k=2.0, lmax=9,
            )
        result[identifier] = {
            "path": str(modal_path), "file_sha256": modal_sha,
            "operator_sha256": geometry["operator_sha256"],
        }
    if set(result) != set(geometry_entries):
        raise ValueError("operator index does not cover the exact 500-geometry library")
    return result


def _validate_occupancy_run_artifacts(
    index: dict, index_sha: str, index_path: Path, scenes: list[dict],
    geometry_entries: dict[str, dict], query_plan: dict, method_gate: dict,
    method_screen_sha: str, method_gate_sha: str, layout_sha: str,
    library_sha: str,
) -> tuple[dict[str, dict], str, str]:
    """Re-verify the completed occupancy run and its per-geometry cache proofs."""
    root = Path(index_path).parent.resolve(strict=True)
    state_path = (root / "state.json").resolve(strict=True)
    provenance_path = (root / "provenance.json").resolve(strict=True)
    method_gate_path = (root / "method_gate.json").resolve(strict=True)
    state, state_sha = _json_with_sha(state_path)
    provenance, provenance_sha = _json_with_sha(provenance_path)
    recorded_gate, recorded_gate_sha = _json_with_sha(method_gate_path)
    if recorded_gate != method_gate or recorded_gate_sha != method_gate_sha:
        raise ValueError("occupancy run method_gate.json differs from the validated source-bound gate")

    if (state.get("schema") != "structural-smooth-v6-occupancy-index-state-v1"
            or state.get("status") != "passed-not-released"
            or state.get("provenance_sha256") != canonical_sha256(provenance)
            or state.get("method_gate_sha256") != method_gate_sha
            or state.get("expected_geometries") != 500
            or state.get("completed_geometry_count") != 500
            or len(state.get("completed_geometries", {})) != 500
            or state.get("failed_geometry_count") != 0
            or state.get("completed_scene_count") != SCENE_COUNT
            or state.get("production_labels_created") != 0
            or state.get("released") is not False
            or state.get("failed_geometries") != {}
            or state.get("failed_scenes") != {}
            or state.get("occupancy_index_sha256") != index_sha
            or Path(state.get("occupancy_index_path", "")).resolve() != Path(index_path).resolve()):
        raise ValueError("occupancy state.json is incomplete, released, or detached from the final index")
    completed_scenes = state.get("completed_scenes")
    index_rows = index.get("entries")
    if (not isinstance(completed_scenes, dict) or not isinstance(index_rows, list)
            or len(completed_scenes) != SCENE_COUNT
            or any(completed_scenes.get(str(i)) != row for i, row in enumerate(index_rows))):
        raise ValueError("occupancy state scene checkpoints do not exactly match occupancy_index.json")

    expected_provenance = {
        "schema": "structural-smooth-v6-occupancy-index-provenance-v1",
        "library_manifest_sha256": library_sha,
        "layout_manifest_sha256": layout_sha,
        "method_screen_sha256": method_screen_sha,
        "method_gate_sha256": method_gate_sha,
        "method_audit_sha256": method_screen_sha,
        "occupancy_schema": OCCUPANCY_SCHEMA,
        "mesh_h_reference": 0.01,
        "mesh_h_candidate": 0.02,
        "curve_order": 2,
        "selected_uncertainty_band": method_gate["selected_uncertainty_band"],
        "acceptance_threshold": method_gate["acceptance_threshold"],
        "scene_count": SCENE_COUNT,
        "geometry_count": 500,
        "unique_rotation_query_count": query_plan["unique_geometry_rotation_query_count"],
        "production_labels_created": 0,
        "released": False,
    }
    for key, expected in expected_provenance.items():
        if provenance.get(key) != expected:
            raise ValueError(f"occupancy provenance {key} differs from the validated run")

    project = Path(__file__).resolve().parents[3]
    source_paths = {
        "scripts/generate_occupancy_index_v6.py": project / "scripts" / "generate_occupancy_index_v6.py",
        "src/cspgnet/data_generation/occupancy_index_v6.py": Path(__file__).with_name("occupancy_index_v6.py"),
        "src/cspgnet/data_generation/occupancy_mesh_v6.py": Path(__file__).with_name("occupancy_mesh_v6.py"),
        "src/cspgnet/data_generation/occupancy_hybrid_v6.py": Path(__file__).with_name("occupancy_hybrid_v6.py"),
        "src/cspgnet/data_generation/formal_neumann_3d.py": Path(__file__).with_name("formal_neumann_3d.py"),
        "src/cspgnet/data_generation/scene_generation_v6.py": Path(__file__),
        "src/cspgnet/data_generation/scene_dataset_v6.py": Path(__file__).with_name("scene_dataset_v6.py"),
        "scripts/pilot_ellipsoid_bore_8_cad_v6.py": project / "scripts" / "pilot_ellipsoid_bore_8_cad_v6.py",
    }
    recorded_sources = provenance.get("source_sha256")
    if not isinstance(recorded_sources, dict):
        raise ValueError("occupancy provenance lacks source hashes")
    recorded_sources = {str(key).replace("\\", "/"): value
                        for key, value in recorded_sources.items()}
    for relative, source_path in source_paths.items():
        source_sha = recorded_sources.get(relative)
        if (not isinstance(source_sha, str) or len(source_sha) != 64
                or not source_path.is_file() or sha256_file(source_path) != source_sha):
            raise ValueError(f"occupancy source hash is missing or stale: {relative}")

    completed_geometries = state.get("completed_geometries")
    groups = query_plan.get("groups")
    if (not isinstance(completed_geometries, dict) or not isinstance(groups, list)
            or len(completed_geometries) != 500 or len(groups) != 500):
        raise ValueError("occupancy run does not contain exactly 500 completed geometry caches")
    cache_root = root / "geometry_cache"
    query_audits: dict[str, dict] = {}
    for group in groups:
        geometry_id = group["geometry_id"]
        checkpoint = completed_geometries.get(geometry_id)
        archive_path = (cache_root / f"{geometry_id}.npz").resolve(strict=True)
        proof_path = (cache_root / f"{geometry_id}.json").resolve(strict=True)
        if (not isinstance(checkpoint, dict) or checkpoint.get("status") != "passed"
                or Path(checkpoint.get("archive_path", "")).resolve() != archive_path
                or Path(checkpoint.get("proof_path", "")).resolve() != proof_path
                or checkpoint.get("archive_sha256") != sha256_file(archive_path)
                or checkpoint.get("proof_sha256") != sha256_file(proof_path)):
            raise ValueError(f"occupancy state/cache checkpoint mismatch: {geometry_id}")
        arrays, proof = load_geometry_cache(
            archive_path, proof_path, geometry_id=geometry_id,
            structure_sha256=group["structure_sha256"],
            library_sha256=library_sha, layout_sha256=layout_sha,
            method_audit_sha256=method_screen_sha,
        )
        if (checkpoint.get("query_count") != len(group["queries"])
                or checkpoint.get("passed_queries") != len(group["queries"])
                or proof.get("query_count") != len(group["queries"])
                or proof.get("acceptance_threshold") != method_gate["acceptance_threshold"]
                or proof.get("method_audit_sha256") != method_screen_sha):
            raise ValueError(f"occupancy geometry proof is not bound to the selected method: {geometry_id}")
        rows = proof.get("queries")
        if not isinstance(rows, list) or len(rows) != len(group["queries"]):
            raise ValueError(f"occupancy geometry proof query count mismatch: {geometry_id}")
        candidate_masks = arrays.get("candidate_masks")
        reference_masks = arrays.get("reference_masks")
        if (candidate_masks is None or reference_masks is None
                or not np.isin(candidate_masks, (0, 1)).all()
                or not np.isin(reference_masks, (0, 1)).all()):
            raise ValueError(f"occupancy geometry cache masks are not binary: {geometry_id}")
        for expected, row in zip(group["queries"], rows):
            if (row.get("query_id") != expected["query_id"]
                    or row.get("rotation_sha256") != expected["rotation_sha256"]
                    or row.get("status") != "passed"
                    or row.get("acceptance_threshold") != method_gate["acceptance_threshold"]):
                raise ValueError(f"occupancy geometry query differs from frozen plan: {geometry_id}")
            if row["query_id"] in query_audits:
                raise ValueError("occupancy geometry caches contain a duplicate query ID")
            query_audits[row["query_id"]] = row
        del arrays
    if len(query_audits) != query_plan["unique_geometry_rotation_query_count"]:
        raise ValueError("occupancy geometry caches do not cover the exact unique query plan")
    return query_audits, state_sha, provenance_sha


def _validate_occupancy_index(index: dict, index_sha: str, layout_sha: str,
                              library_sha: str, scenes: list[dict],
                              geometry_entries: dict[str, dict], index_path: Path,
                              threshold: float, method_screen_sha: str,
                              method_gate_sha: str,
                              query_audits: dict[str, dict]) -> dict[int, dict]:
    if (index.get("schema") != OCCUPANCY_INDEX_SCHEMA or index.get("status") != "passed"
            or index.get("layout_manifest_sha256") != layout_sha
            or index.get("library_manifest_sha256") != library_sha
            or index.get("occupancy_schema") != OCCUPANCY_SCHEMA
            or index.get("acceptance_threshold") != threshold
            or index.get("scene_count") != SCENE_COUNT):
        raise ValueError("occupancy index is not passed/bound to this layout, library, and gate threshold")
    if index.get("method_screen_sha256") != method_screen_sha:
        raise ValueError("occupancy index method_screen_sha256 mismatch")
    if index.get("method_gate_sha256") != method_gate_sha:
        raise ValueError("occupancy index method_gate_sha256 mismatch")
    rows = index.get("entries")
    if not isinstance(rows, list) or len(rows) != SCENE_COUNT:
        raise ValueError("occupancy index must cover all 10,000 scenes")
    result: dict[int, dict] = {}
    root = index_path.parent
    for expected, row in enumerate(rows):
        scene = scenes[expected]
        if (not isinstance(row, dict) or row.get("index") != expected
                or row.get("scene_id") != scene["scene_id"]
                or row.get("layout_sha256") != scene["layout_sha256"]):
            raise ValueError(f"occupancy index scene identity mismatch at {expected}")
        archive_path = _safe_artifact_path(root, row.get("archive_path"), f"occupancy[{expected}].archive_path")
        proof_path = _safe_artifact_path(root, row.get("proof_path"), f"occupancy[{expected}].proof_path")
        archive_sha = row.get("archive_sha256")
        proof_sha = row.get("proof_sha256")
        if (not isinstance(archive_sha, str) or sha256_file(archive_path) != archive_sha
                or not isinstance(proof_sha, str) or sha256_file(proof_path) != proof_sha):
            raise ValueError(f"occupancy artifact checksum mismatch at scene {expected}")
        proof, _ = _json_with_sha(proof_path)
        if (proof.get("schema") != "geometry-occupancy-scene-audit-v1"
                or proof.get("status") != "passed" or proof.get("scene_id") != scene["scene_id"]
                or proof.get("layout_sha256") != scene["layout_sha256"]
                or proof.get("occupancy_schema") != OCCUPANCY_SCHEMA):
            raise ValueError(f"occupancy proof identity/status mismatch at scene {expected}")
        if proof.get("method_audit_sha256") != method_screen_sha:
            raise ValueError(f"occupancy proof method_audit_sha256 mismatch at scene {expected}")
        if proof.get("acceptance_threshold") != threshold:
            raise ValueError(f"occupancy proof threshold mismatch at scene {expected}")
        if proof.get("archive_sha256") != archive_sha:
            raise ValueError(f"occupancy scene proof archive SHA mismatch at scene {expected}")
        proof_rows = proof.get("objects")
        if not isinstance(proof_rows, list) or len(proof_rows) != scene["object_count"]:
            raise ValueError(f"occupancy proof lacks per-object rows at scene {expected}")
        with np.load(archive_path, allow_pickle=False) as archive:
            if set(archive.files) != {"geometry_occupancy"}:
                raise ValueError(f"unexpected arrays in occupancy archive at scene {expected}")
            masks = np.asarray(archive["geometry_occupancy"])
        if (masks.shape != (scene["object_count"], 128, 128)
                or masks.dtype != np.uint8 or not np.isin(masks, (0, 1)).all()):
            raise ValueError(f"occupancy mask archive is malformed at scene {expected}")
        object_errors = []
        for object_index, (obj, audit) in enumerate(zip(scene["objects"], proof_rows)):
            geometry = geometry_entries[obj["library_id"]]
            rotation_sha = _rotation_sha(obj["rotation"])
            query_id = hashlib.sha256(json.dumps(
                [obj["library_id"], geometry["structure_sha256"].lower(), rotation_sha],
                separators=(",", ":")).encode("utf-8")).hexdigest()
            query = query_audits.get(query_id)
            mask_sha = hashlib.sha256(np.ascontiguousarray(masks[object_index]).tobytes()).hexdigest()
            if query is None:
                raise ValueError(f"scene {expected} references a rotation without a geometry-cache proof")
            if (audit.get("status") != "passed" or audit.get("library_id") != obj["library_id"]
                    or audit.get("structure_sha256") != geometry["structure_sha256"]
                    or audit.get("rotation_sha256") != rotation_sha
                    or audit.get("occupancy_schema") != OCCUPANCY_SCHEMA
                    or audit.get("method_audit_sha256") != method_screen_sha
                    or audit.get("acceptance_threshold") != threshold
                    or audit.get("object_index") != object_index
                    or audit.get("mask_sha256") != query.get("mask_sha256")
                    or audit.get("mask_sha256") != mask_sha
                    or audit.get("reference_mask_sha256") != query.get("reference_mask_sha256")
                    or audit.get("mismatch_count") != query.get("mismatch_count")
                    or audit.get("point_count") != query.get("point_count")
                    or audit.get("max_disagreement_fraction") != query.get("max_disagreement_fraction")
                    or not isinstance(audit.get("max_disagreement_fraction"), (int, float))
                    or isinstance(audit.get("max_disagreement_fraction"), bool)
                    or not 0 <= float(audit["max_disagreement_fraction"]) < threshold):
                raise ValueError(f"occupancy per-object proof fails gate at scene {expected}")
            object_errors.append(float(audit["max_disagreement_fraction"]))
            method_hash = audit.get("method_audit_sha256")
            mask_hash = audit.get("mask_sha256")
            for label, value in (("method", method_hash), ("mask", mask_hash)):
                if not isinstance(value, str) or len(value) != 64:
                    raise ValueError(f"occupancy {label} hash missing at scene {expected}")
                int(value, 16)
        if proof.get("max_disagreement_fraction") != max(object_errors):
            raise ValueError(f"occupancy scene maximum error is inconsistent at scene {expected}")
        result[expected] = {
            "archive_path": str(archive_path), "archive_sha256": archive_sha,
            "proof": proof,
        }
    return result


@dataclass(frozen=True)
class GenerationPlan:
    library_path: Path
    operator_index_path: Path
    layout_path: Path
    occupancy_index_path: Path
    occupancy_method_screen_path: Path
    occupancy_method_gate_path: Path
    bem_summary_path: Path
    bem_provenance_path: Path
    gate_path: Path
    geometry_entries: dict[str, dict]
    scenes: list[dict]
    operator_specs: dict[str, dict]
    occupancy_specs: dict[int, dict]
    source_bindings: dict[str, str]
    gate_sha256: str
    occupancy_threshold: float


def _generation_source_bindings(*, library_path: Path, operator_index_path: Path,
                                layout_path: Path, occupancy_index_path: Path,
                                occupancy_method_screen_path: Path,
                                occupancy_method_gate_path: Path,
                                bem_summary_path: Path, bem_provenance_path: Path) -> dict[str, str]:
    project = Path(__file__).resolve().parents[3]
    source_paths = {
        "scene_labels_source_sha256": Path(__file__).with_name("scene_labels_v6.py"),
        "scene_dataset_source_sha256": Path(__file__).with_name("scene_dataset_v6.py"),
        "scene_generation_source_sha256": Path(__file__),
        "scene_generation_cli_source_sha256": project / "scripts" / "generate_scenes_v6.py",
        "release_gate_builder_source_sha256": project / "scripts" / "build_scene_release_gate_v6.py",
        "formal_neumann_source_sha256": Path(__file__).with_name("formal_neumann_3d.py"),
        "tmatrix_source_sha256": Path(__file__).with_name("tmatrix_neumann_3d.py"),
        "bem_calibration_driver_source_sha256": project / "scripts" / "calibrate_ellipsoid_bore_bem_v6.py",
        "bem_solver_source_sha256": Path(__file__).with_name("structural_bem.py"),
        "occupancy_index_cli_source_sha256": project / "scripts" / "generate_occupancy_index_v6.py",
        "occupancy_index_module_source_sha256": Path(__file__).with_name("occupancy_index_v6.py"),
    }
    bindings = {
        "library_manifest_sha256": sha256_file(library_path),
        "operator_index_sha256": sha256_file(operator_index_path),
        "layout_manifest_sha256": sha256_file(layout_path),
        "occupancy_index_sha256": sha256_file(occupancy_index_path),
        "occupancy_run_state_sha256": sha256_file(occupancy_index_path.parent / "state.json"),
        "occupancy_run_provenance_sha256": sha256_file(occupancy_index_path.parent / "provenance.json"),
        "occupancy_method_screen_sha256": sha256_file(occupancy_method_screen_path),
        "occupancy_method_gate_sha256": sha256_file(occupancy_method_gate_path),
        "bem_calibration_summary_sha256": sha256_file(bem_summary_path),
        "bem_calibration_provenance_sha256": sha256_file(bem_provenance_path),
    }
    bindings.update({key: sha256_file(path) for key, path in source_paths.items()})
    return bindings


def _preflight_generation_sources(library_path: Path, operator_index_path: Path,
                                 layout_path: Path, occupancy_index_path: Path,
                                 occupancy_method_screen_path: Path,
                                 occupancy_method_gate_path: Path,
                                 bem_summary_path: Path) -> dict[str, Any]:
    """Validate all immutable inputs and evidence before building a release gate."""
    library_path = Path(library_path).resolve(strict=True)
    library, library_sha = _json_with_sha(library_path)
    geometry_entries = _validate_library(library, library_sha)
    operator_index_path = Path(operator_index_path).resolve(strict=True)
    layout_path = Path(layout_path).resolve(strict=True)
    occupancy_index_path = Path(occupancy_index_path).resolve(strict=True)
    occupancy_method_screen_path = Path(occupancy_method_screen_path).resolve(strict=True)
    occupancy_method_gate_path = Path(occupancy_method_gate_path).resolve(strict=True)
    bem_summary_path = Path(bem_summary_path).resolve(strict=True)

    layout, layout_sha = _json_with_sha(layout_path)
    scenes = _validate_layout(layout, layout_sha, library_sha, geometry_entries)
    operator_index, operator_index_sha = _json_with_sha(operator_index_path)
    operator_specs = _validate_operator_index(
        operator_index, operator_index_sha, library_sha, geometry_entries, operator_index_path,
    )
    method_gate, method_screen_sha, method_gate_sha = _validate_occupancy_method_evidence(
        occupancy_method_screen_path, occupancy_method_gate_path,
    )
    bem_summary, bem_summary_sha, bem_provenance_path, bem_provenance_sha = (
        _validate_bem_calibration_summary(bem_summary_path)
    )
    occupancy_index, occupancy_index_sha = _json_with_sha(occupancy_index_path)
    threshold = float(method_gate["acceptance_threshold"])
    query_plan = validate_grouped_plan(scenes, geometry_entries)
    query_audits, occupancy_state_sha, occupancy_provenance_sha = (
        _validate_occupancy_run_artifacts(
            occupancy_index, occupancy_index_sha, occupancy_index_path, scenes,
            geometry_entries, query_plan, method_gate, method_screen_sha,
            method_gate_sha, layout_sha, library_sha,
        )
    )
    occupancy_specs = _validate_occupancy_index(
        occupancy_index, occupancy_index_sha, layout_sha, library_sha, scenes,
        geometry_entries, occupancy_index_path, threshold, method_screen_sha,
        method_gate_sha, query_audits,
    )
    source_bindings = _generation_source_bindings(
        library_path=library_path, operator_index_path=operator_index_path,
        layout_path=layout_path, occupancy_index_path=occupancy_index_path,
        occupancy_method_screen_path=occupancy_method_screen_path,
        occupancy_method_gate_path=occupancy_method_gate_path,
        bem_summary_path=bem_summary_path, bem_provenance_path=bem_provenance_path,
    )
    expected_gate = {
        "schema": GATE_SCHEMA,
        "status": "passed",
        "label_generation_authorized": True,
        "geometry_occupancy_acceptance_threshold": threshold,
        "source_bindings": source_bindings,
        "evidence": {
            "geometry_count": len(geometry_entries),
            "scene_count": len(scenes),
            "operator_count": len(operator_specs),
            "occupancy_scene_proofs_verified": len(occupancy_specs),
            "occupancy_unique_geometry_queries_verified": len(query_audits),
            "occupancy_state_sha256": occupancy_state_sha,
            "occupancy_provenance_sha256": occupancy_provenance_sha,
            "bem_case_ids": [row["id"] for row in bem_summary["results"]],
            "bem_summary_sha256": bem_summary_sha,
            "bem_provenance_sha256": bem_provenance_sha,
            "occupancy_method_screen_sha256": method_screen_sha,
            "occupancy_method_gate_sha256": method_gate_sha,
            "occupancy_method": method_gate["method"],
            "occupancy_selected_uncertainty_band": method_gate["selected_uncertainty_band"],
            "bem_label_error_limit": 0.02,
            "bem_adjacent_mesh_error_limit": 0.02,
            "bem_gmres_residual_limit": 1.0e-8,
            "local_independent_flux_error_limit_exclusive": 0.03,
        },
    }
    return {
        "library_path": library_path, "operator_index_path": operator_index_path,
        "layout_path": layout_path, "occupancy_index_path": occupancy_index_path,
        "occupancy_method_screen_path": occupancy_method_screen_path,
        "occupancy_method_gate_path": occupancy_method_gate_path,
        "bem_summary_path": bem_summary_path, "bem_provenance_path": bem_provenance_path,
        "gate_payload": expected_gate, "geometry_entries": geometry_entries,
        "scenes": scenes, "operator_specs": operator_specs,
        "occupancy_specs": occupancy_specs, "source_bindings": source_bindings,
        "occupancy_threshold": threshold,
    }


def build_scene_label_release_gate(library_path: Path, operator_index_path: Path,
                                  layout_path: Path, occupancy_index_path: Path,
                                  occupancy_method_screen_path: Path,
                                  occupancy_method_gate_path: Path,
                                  bem_summary_path: Path, output_path: Path) -> dict:
    """Build a deterministic source-bound gate only after every release input passes."""
    prepared = _preflight_generation_sources(
        library_path, operator_index_path, layout_path, occupancy_index_path,
        occupancy_method_screen_path, occupancy_method_gate_path, bem_summary_path,
    )
    output_path = Path(output_path).resolve()
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite release gate: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = prepared["gate_payload"]
    encoded = (json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")
    with tempfile.NamedTemporaryFile(dir=output_path.parent, prefix=f".{output_path.name}.",
                                     suffix=".pending", delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
    if output_path.exists():
        temporary.unlink(missing_ok=True)
        raise FileExistsError(f"refusing to overwrite release gate: {output_path}")
    os.replace(temporary, output_path)
    return {**payload, "release_gate_path": str(output_path),
            "release_gate_sha256": sha256_file(output_path)}


def validate_generation_inputs(library_path: Path, operator_index_path: Path,
                               layout_path: Path, occupancy_index_path: Path,
                               occupancy_method_screen_path: Path,
                               occupancy_method_gate_path: Path,
                               bem_summary_path: Path, gate_path: Path) -> GenerationPlan:
    """Fully preflight immutable inputs before the output directory is created."""
    prepared = _preflight_generation_sources(
        library_path, operator_index_path, layout_path, occupancy_index_path,
        occupancy_method_screen_path, occupancy_method_gate_path, bem_summary_path,
    )
    gate_path = Path(gate_path).resolve(strict=True)
    gate, gate_sha = _json_with_sha(gate_path)
    if gate != prepared["gate_payload"]:
        raise ValueError("release gate is stale or differs from the exact validated evidence")
    return GenerationPlan(
        prepared["library_path"], prepared["operator_index_path"],
        prepared["layout_path"], prepared["occupancy_index_path"],
        prepared["occupancy_method_screen_path"], prepared["occupancy_method_gate_path"],
        prepared["bem_summary_path"], prepared["bem_provenance_path"], gate_path,
        prepared["geometry_entries"], prepared["scenes"], prepared["operator_specs"],
        prepared["occupancy_specs"], prepared["source_bindings"], gate_sha,
        prepared["occupancy_threshold"],
    )


def _current_source_bindings(plan: GenerationPlan) -> dict[str, str]:
    return _generation_source_bindings(
        library_path=plan.library_path, operator_index_path=plan.operator_index_path,
        layout_path=plan.layout_path, occupancy_index_path=plan.occupancy_index_path,
        occupancy_method_screen_path=plan.occupancy_method_screen_path,
        occupancy_method_gate_path=plan.occupancy_method_gate_path,
        bem_summary_path=plan.bem_summary_path,
        bem_provenance_path=plan.bem_provenance_path,
    )


def _worker_init(operator_specs: dict[str, dict], geometry_entries: dict[str, dict],
                 points: np.ndarray, faces: np.ndarray, probes: np.ndarray, phi_pinv: np.ndarray,
                 angular_nodes: int, angular_order: int, occupancy_threshold: float) -> None:
    global _WORKER
    operators: dict[str, LocalModalOperator] = {}
    for identifier, spec in operator_specs.items():
        path = Path(spec["path"])
        if sha256_file(path) != spec["file_sha256"]:
            raise RuntimeError(f"modal artifact changed after preflight: {identifier}")
        with np.load(path, allow_pickle=False) as archive:
            modal = np.asarray(archive["modal_dtn"], dtype=np.complex128)
        operators[identifier] = LocalModalOperator.from_modal_dtn(identifier, modal, k=2.0, lmax=9)
    quadrature = tmatrix.angular_quadrature(angular_nodes, angular_order)
    _WORKER = {
        "operators": operators, "geometry_entries": geometry_entries,
        "points": np.asarray(points, dtype=np.float64),
        "faces": np.asarray(faces, dtype=np.int32),
        "probes": np.asarray(probes, dtype=np.complex128),
        "phi_pinv": np.asarray(phi_pinv, dtype=np.complex128),
        "quadrature": quadrature,
        "rotation_cache": HarmonicRotationCache(quadrature, 10),
        "occupancy_threshold": float(occupancy_threshold),
    }


def _worker_label(task: tuple[dict, dict]) -> tuple[int, dict, dict]:
    if _WORKER is None:
        raise RuntimeError("scene label worker was not initialized")
    scene, occupancy_spec = task
    path = Path(occupancy_spec["archive_path"])
    if sha256_file(path) != occupancy_spec["archive_sha256"]:
        raise RuntimeError(f"occupancy archive changed after preflight: {scene['scene_id']}")
    with np.load(path, allow_pickle=False) as archive:
        if "geometry_occupancy" not in archive.files:
            raise ValueError(f"occupancy archive lacks geometry_occupancy: {path}")
        occupancy = np.asarray(archive["geometry_occupancy"])
    count = int(scene["object_count"])
    if (occupancy.dtype != np.uint8 or occupancy.shape != (count, 128, 128)
            or not np.isin(occupancy, (0, 1)).all()):
        raise ValueError(f"invalid binary occupancy payload: {scene['scene_id']}")
    proof = occupancy_spec["proof"]
    for object_index, row in enumerate(proof["objects"]):
        actual_mask_hash = hashlib.sha256(
            np.ascontiguousarray(occupancy[object_index]).tobytes()
        ).hexdigest()
        if actual_mask_hash != row["mask_sha256"]:
            raise RuntimeError(f"occupancy mask/proof mismatch: {scene['scene_id']} object {object_index}")

    geometry_ids = [str(item["library_id"]) for item in scene["objects"]]
    centers = np.asarray(scene["centers"], dtype=np.float64)
    rotations = np.asarray(scene["rotations"], dtype=np.float64)
    direction = np.asarray(scene["incident_direction"], dtype=np.float64)
    fields, label_metadata = generate_scene_labels(
        geometry_ids, _WORKER["operators"], centers, rotations, direction,
        _WORKER["points"], _WORKER["probes"], _WORKER["phi_pinv"],
        k=2.0, canonical_radius=1.5, lmax=9,
        quadrature=_WORKER["quadrature"], rotation_cache=_WORKER["rotation_cache"],
    )
    arrays, record = assemble_scene_payload(
        scene, _WORKER["geometry_entries"], fields,
        label_metadata=label_metadata,
        interface_points=_WORKER["points"],
        interface_faces=_WORKER["faces"],
        probe_values=_WORKER["probes"], phi_pinv=_WORKER["phi_pinv"],
        occupancy=occupancy, occupancy_proof=proof,
    )
    return int(scene["index"]), arrays, record


def _atomic_state(path: Path, record: dict) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.pending")
    encoded = (json.dumps(record, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")
    with temporary.open("wb") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def run_generation(plan: GenerationPlan, output_root: Path, *, workers: int = 4,
                   resume: bool = False, min_launch_bytes: int = START_FLOOR_BYTES,
                   reserve_bytes: int = RESERVE_BYTES, checkpoint_interval: int = 1_000) -> dict:
    if type(workers) is not int or workers not in WORKER_COUNTS:
        raise ValueError(f"workers must be one of {sorted(WORKER_COUNTS)}; 3 workers was not benchmarked")
    if _current_source_bindings(plan) != plan.source_bindings:
        raise RuntimeError("one or more bound inputs changed after preflight")
    gate, current_gate_sha = _json_with_sha(plan.gate_path)
    if (current_gate_sha != plan.gate_sha256 or gate.get("source_bindings") != plan.source_bindings
            or gate.get("status") != "passed" or gate.get("label_generation_authorized") is not True):
        raise RuntimeError("source-bound release gate changed or no longer authorizes labels")
    free = available_physical_bytes()
    if free < min_launch_bytes:
        raise MemoryError(f"label workers require >= {min_launch_bytes / 1024**3:.1f} GiB free; got {free / 1024**3:.2f}")
    run_config = {
        "problem": "3d-helmholtz-k2-neumann-ellipsoid-through-bore",
        "mesh_strategy": "graded-local-FEM",
        "geometry_count": len(plan.geometry_entries),
        "scene_count": SCENE_COUNT,
        "scene_object_counts": {"2": 3334, "3": 3333, "4": 3333},
        "input_lmax": 9, "output_lmax": 10, "network_probe_lmax": 7,
        "interface_nodes": 128, "canonical_radius": 1.5,
        "angular_quadrature": {"gauss_legendre_nodes": 24, "azimuthal_nodes": 48},
        "geometry_occupancy_acceptance_threshold": plan.occupancy_threshold,
        "parallel_stage": "CPU-scene-labels-only; bounded-spawn; single-writer",
        "environment": {
            "python": platform.python_version(), "numpy": np.__version__,
            "scipy": scipy.__version__, "logical_cpu_count": os.cpu_count(),
            "blas_threads_per_worker": 1,
        },
    }
    output_root = Path(output_root).resolve()
    writer = SceneSplitWriterV6(
        output_root, "train", SCENE_COUNT, release_gate_path=plan.gate_path,
        source_bindings=plan.source_bindings, run_config=run_config,
        resume=resume, manifest_checkpoint_interval=checkpoint_interval,
    )
    points, faces = formal.canonical_surface(128, 1.5)
    probes = formal.spherical_harmonic_matrix(points, formal.spherical_modes(7))
    phi_pinv = formal.weighted_pseudoinverse(
        probes, formal.equal_area_weights(128, 1.5),
    )
    pending_indices = [i for i in range(SCENE_COUNT) if i not in writer.rows]
    tasks = iter(pending_indices)
    pending = {}
    completed_now = 0
    started = time.perf_counter()
    state_path = output_root / "generation_state.json"
    min_free = free
    context = mp.get_context("spawn")
    status = "running"
    failure = None
    _atomic_state(state_path, {
        "schema": "structural-smooth-v6-scene-generation-state-v1", "status": status,
        "existing_committed": len(writer.rows), "remaining_at_start": len(pending_indices),
        "worker_count": workers, "max_in_flight": 2 * workers,
        "source_bindings": plan.source_bindings, "release_gate_sha256": plan.gate_sha256,
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    })
    try:
        with _MemoryMonitor() as monitor:
            with ProcessPoolExecutor(
                max_workers=workers, mp_context=context, initializer=_worker_init,
                initargs=(plan.operator_specs, plan.geometry_entries, points, faces, probes, phi_pinv,
                          24, 48, plan.occupancy_threshold),
            ) as executor:
                while True:
                    current_free = available_physical_bytes()
                    min_free = min(min_free, current_free)
                    if current_free < reserve_bytes:
                        raise MemoryError("system available RAM crossed the hard reserve before dispatch")
                    while len(pending) < 2 * workers:
                        try:
                            index = next(tasks)
                        except StopIteration:
                            break
                        if available_physical_bytes() < reserve_bytes:
                            raise MemoryError("system available RAM crossed the hard reserve during dispatch")
                        future = executor.submit(
                            _worker_label, (plan.scenes[index], plan.occupancy_specs[index])
                        )
                        pending[future] = index
                    if not pending:
                        break
                    done, _ = wait(pending, return_when=FIRST_COMPLETED, timeout=1.0)
                    if not done:
                        min_free = min(min_free, available_physical_bytes())
                        if min_free < reserve_bytes:
                            raise MemoryError("system available RAM crossed the hard reserve while workers were active")
                        continue
                    for future in done:
                        expected_index = pending.pop(future)
                        index, arrays, record = future.result()
                        if index != expected_index:
                            raise RuntimeError("worker returned the wrong scene index")
                        writer.commit(arrays, record)
                        completed_now += 1
                        min_free = min(min_free, available_physical_bytes())
                    _atomic_state(state_path, {
                        "schema": "structural-smooth-v6-scene-generation-state-v1",
                        "status": "running", "committed_samples": len(writer.rows),
                        "newly_committed_this_run": completed_now,
                        "remaining": SCENE_COUNT - len(writer.rows),
                        "worker_count": workers, "max_in_flight": 2 * workers,
                        "minimum_free_memory_reserve_bytes": int(reserve_bytes),
                        "minimum_available_memory_gib": min(min_free, monitor.minimum) / 1024**3,
                        "source_bindings": plan.source_bindings,
                        "release_gate_sha256": plan.gate_sha256,
                    })
                    if min(min_free, monitor.minimum) < reserve_bytes:
                        raise MemoryError("system available RAM crossed the hard reserve")
        writer.checkpoint()
        if len(writer.rows) != SCENE_COUNT:
            raise RuntimeError(f"generation stopped with {len(writer.rows)}/{SCENE_COUNT} scenes")
        status = "labels-generated-not-payload-scan-passed"
    except BaseException as exc:
        failure = f"{type(exc).__name__}: {exc}"
        status = "interrupted-or-failed"
        for future in pending:
            future.cancel()
        try:
            writer.checkpoint()
        except Exception:
            pass
        raise
    finally:
        elapsed = time.perf_counter() - started
        _atomic_state(state_path, {
            "schema": "structural-smooth-v6-scene-generation-state-v1",
            "status": status, "failure": failure,
            "committed_samples": len(writer.rows),
            "newly_committed_this_run": completed_now,
            "remaining": SCENE_COUNT - len(writer.rows),
            "worker_count": workers, "max_in_flight": 2 * workers,
            "minimum_free_memory_reserve_bytes": int(reserve_bytes),
            "elapsed_wall_seconds": elapsed,
            "minimum_available_memory_gib": min(min_free, getattr(locals().get("monitor", None), "minimum", min_free)) / 1024**3,
            "release_gate_sha256": plan.gate_sha256,
            "training_ready": False,
            "payload_scan_required": True,
        })
    return json.loads(state_path.read_text(encoding="utf-8"))

