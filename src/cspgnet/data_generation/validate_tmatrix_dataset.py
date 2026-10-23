"""Validate versioned T-matrix data, local-library gates and split separation."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np

from . import formal_neumann_3d as formal
from .tmatrix_dataset import PROFILE, RELEASE_GATE_SCHEMA, SCHEMA, SPLITS, _sha256


def validate(
    root: Path, profile: str, verify_hashes: bool = True,
    splits: tuple[str, ...] | None = None,
) -> dict:
    base = root / profile
    selected_splits = SPLITS if splits is None else splits
    if (not selected_splits or len(set(selected_splits)) != len(selected_splits)
            or any(split not in SPLITS for split in selected_splits)):
        raise ValueError(f"invalid split selection: {selected_splits}")
    failures = []
    report = {"schema_version": SCHEMA, "profile": profile, "splits": {}, "failures": failures}
    seen_geometry: dict[tuple, str] = {}
    seen_seeds: dict[int, str] = {}
    seen_scenes: dict[str, str] = {}
    gate_hashes = set()
    unitarity_defects = []
    local_fem_residuals = []
    local_superposition_errors = []
    mesh_quality_minima = []
    points, faces = formal.canonical_surface()
    probe = formal.spherical_harmonic_matrix(points, formal.spherical_modes(7))
    pinv = formal.weighted_pseudoinverse(probe, formal.equal_area_weights(128, 1.5))

    expected_by_split = dict(zip(SPLITS, PROFILE[profile]["samples"]))
    for split in selected_splits:
        expected_count = expected_by_split[split]
        lib_path = base / "library" / split / "manifest.json"
        data_path = base / "data" / split / "manifest.json"
        if not lib_path.exists() or not data_path.exists():
            failures.append(f"{split}: library or data manifest is absent")
            continue
        library = json.loads(lib_path.read_text(encoding="utf-8"))
        data = json.loads(data_path.read_text(encoding="utf-8"))
        if library.get("schema_version") != SCHEMA or data.get("schema_version") != SCHEMA:
            failures.append(f"{split}: schema mismatch")
        if library.get("status") != "complete" or data.get("status") != "complete":
            failures.append(f"{split}: library or data is not complete")
        if library.get("mesh_strategy") != "graded-local-FEM" or data.get("mesh_strategy") != "graded-local-FEM":
            failures.append(f"{split}: dataset/library was not generated with graded local FEM")
        graded = library.get("graded_mesh_calibration", {})
        data_graded = data.get("graded_mesh_calibration", {})
        if not graded.get("path") or graded.get("sha256") != data_graded.get("sha256"):
            failures.append(f"{split}: graded-mesh calibration provenance is missing or inconsistent")
        elif not Path(graded["path"]).is_file() or _sha256(Path(graded["path"])) != graded["sha256"]:
            failures.append(f"{split}: graded-mesh calibration report is missing or corrupt")
        if data.get("label_solver") != "T-matrix Foldy-Lax (not per-scene BEM)":
            failures.append(f"{split}: label solver is not declared correctly")
        gate = data.get("numerical_gate", {})
        if (gate.get("schema_version") != RELEASE_GATE_SCHEMA
                or gate.get("mesh_strategy") != "graded-local-FEM"
                or gate.get("bem_label_audit_passed") is not True
                or gate.get("graded_calibration_sha256") != graded.get("sha256")):
            failures.append(f"{split}: current graded/BEM release gate is absent or inconsistent")
        if data.get("num_samples") != expected_count or len(data.get("samples", [])) != expected_count:
            failures.append(f"{split}: sample count differs from configuration")
        if data.get("library_sha256") != _sha256(lib_path):
            failures.append(f"{split}: library manifest checksum mismatch")
        gate_hashes.add(data.get("numerical_gate", {}).get("sha256"))
        entries = {entry["id"]: entry for entry in library["entries"]}
        if len(entries) != len(library["entries"]):
            failures.append(f"{split}: duplicate library ID")
        family_counts = Counter(entry.get("shape") for entry in library.get("entries", []))
        expected_library_per_shape = dict(zip(
            formal.SHAPE_NAMES,
            [
                PROFILE[profile]["library_per_shape"][SPLITS.index(split)]
                for _ in formal.SHAPE_NAMES
            ],
        ))
        if dict(family_counts) != expected_library_per_shape:
            failures.append(f"{split}: base-shape library counts differ: {dict(family_counts)}")
        for entry in library["entries"]:
            if entry.get("status") != "complete":
                failures.append(f"{split}: incomplete library member {entry['id']}")
                continue
            metrics = entry.get("local_metrics", {})
            if metrics.get("method") != "curved P2 graded weak FEM DtN":
                failures.append(f"{split}: {entry['id']} does not declare graded FEM provenance")
            if metrics.get("z_frobenius", np.inf) >= 0.03 or metrics.get("z_mode_p95", np.inf) >= 0.05:
                failures.append(f"{split}: local mesh gate failed for {entry['id']}")
            profile_names = [item.get("name") for item in metrics.get("mesh_profiles", [])]
            candidate = graded.get("candidate_profiles_by_family", {}).get(entry["shape"])
            if not profile_names or profile_names[0] != candidate:
                failures.append(f"{split}: {entry['id']} does not use its approved graded production profile")
            for level in ("coarse", "fine"):
                mesh_meta = metrics.get(level, {})
                mesh_quality_minima.append(float(mesh_meta.get("element_quality_min", -1.0)))
                if (mesh_meta.get("elements_nonpositive_volume") != 0
                        or mesh_meta.get("element_quality_min", 0.0) <= 0.0
                        or not {"canonical", "scatterer"}.issubset(
                            mesh_meta.get("boundary_size_profile", {})
                        )):
                    failures.append(f"{split}: {entry['id']} {level} mesh is invalid or lacks graded boundary diagnostics")
            direct = metrics.get("direct_fem", {})
            for level in ("coarse", "fine"):
                direct_meta = direct.get(level, {})
                residual = float(direct_meta.get("max_free_dof_relative_residual", np.inf))
                superposition = float(direct_meta.get("modal_superposition_relative_error", np.inf))
                local_fem_residuals.append(residual)
                local_superposition_errors.append(superposition)
                mesh_quality_minima.append(float(direct_meta.get("element_quality_min", -1.0)))
                if (residual >= 1.0e-8 or superposition >= 1.0e-6
                        or direct_meta.get("elements_nonpositive_volume") != 0
                        or direct_meta.get("element_quality_min", 0.0) <= 0.0):
                    failures.append(f"{split}: {entry['id']} {level} direct-FEM residual/superposition/mesh gate failed")
            if entry.get("conversion", {}).get("conversion_residual", np.inf) >= 1.0e-10:
                failures.append(f"{split}: {entry['id']} modal-to-T conversion residual failed")
            geometry = (entry["shape"], *entry["size"])
            if geometry in seen_geometry:
                failures.append(f"{split}: base geometry duplicates {seen_geometry[geometry]}")
            seen_geometry[geometry] = split
            operator_path = base / "library" / split / entry["file"]
            if not operator_path.exists() or (verify_hashes and _sha256(operator_path) != entry.get("sha256")):
                failures.append(f"{split}: missing or corrupt operator {entry['id']}")
            else:
                with np.load(operator_path) as operator:
                    transition = operator["transition"]
                scattering = np.eye(transition.shape[0]) + 2.0 * transition
                defect = float(
                    np.linalg.norm(scattering.conj().T @ scattering - np.eye(len(scattering)))
                    / np.sqrt(len(scattering))
                )
                unitarity_defects.append(defect)
                if not np.isfinite(defect) or defect > 1.0e-3:
                    failures.append(f"{split}: acoustic flux unitarity fails for {entry['id']}: {defect}")

        object_counts = Counter()
        shape_counts = Counter()
        for expected_index, sample in enumerate(data.get("samples", [])):
            if sample.get("index") != expected_index:
                failures.append(f"{split}: sample indices are not contiguous at {expected_index}")
                continue
            seed = sample["sample_seed"]
            if seed in seen_seeds:
                failures.append(f"{split}: seed overlaps with {seen_seeds[seed]}")
            seen_seeds[seed] = split
            physical_scene_sha = sample.get("physical_scene_sha256")
            if not physical_scene_sha:
                failures.append(f"{split}: sample {expected_index} lacks its physical-scene checksum")
            else:
                identity = {
                    "k": sample.get("k"),
                    "incident_direction": sample.get("incident_direction"),
                    "centers": sample.get("centers"),
                    "objects": sample.get("objects"),
                }
                expected_scene_sha = hashlib.sha256(
                    json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
                ).hexdigest()
                if physical_scene_sha != expected_scene_sha:
                    failures.append(f"{split}: sample {expected_index} physical-scene checksum mismatch")
                if physical_scene_sha in seen_scenes:
                    failures.append(f"{split}: scene overlaps with {seen_scenes[physical_scene_sha]}")
                seen_scenes[physical_scene_sha] = f"{split}/{expected_index}"
            n = len(sample["objects"])
            object_counts[n] += 1
            for item in sample["objects"]:
                shape_counts[item["shape"]] += 1
                if item["library_id"] not in entries:
                    failures.append(f"{split}: sample {expected_index} references another library")
                rotation = np.asarray(item["rotation"], dtype=np.float64)
                if (
                    rotation.shape != (3, 3)
                    or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1.0e-6)
                    or not np.isclose(np.linalg.det(rotation), 1.0, atol=1.0e-6)
                ):
                    failures.append(f"{split}: sample {expected_index} invalid SO(3) rotation")
            centers = np.asarray(sample["centers"], dtype=np.float64)
            if n >= 2:
                separation = np.linalg.norm(centers[:, None, :] - centers[None, :, :], axis=-1)
                separation[np.eye(n, dtype=bool)] = np.inf
                if np.min(separation) < 3.6 - 1.0e-8:
                    failures.append(f"{split}: sample {expected_index} has overlapping canonical spheres")
            path = base / "data" / split / sample["file"]
            if not path.exists() or (verify_hashes and _sha256(path) != sample.get("sha256")):
                failures.append(f"{split}: sample {expected_index} is missing or corrupt")
                continue
            with np.load(path) as arrays:
                required = {
                    "Z": (n, 128, 64), "u_total_gt": (n, 128),
                    "u_inc": (n, 128), "q64_gt": (n, 128),
                    "probe_values": (128, 64), "interface_points": (128, 3),
                    "centers": (n, 3), "rotations": (n, 3, 3),
                    "shape_parameters": (n, 3),
                }
                for name, shape in required.items():
                    if name not in arrays or arrays[name].shape != shape:
                        failures.append(f"{split}: sample {expected_index} {name} shape mismatch")
                if any(name not in arrays for name in required):
                    continue
                if any(not np.isfinite(arrays[name]).all() for name in required):
                    failures.append(f"{split}: sample {expected_index} contains non-finite data")
                if not np.isclose(float(arrays["k"]), 2.0):
                    failures.append(f"{split}: sample {expected_index} k mismatch")
                if not np.allclose(arrays["interface_points"], points, atol=1.0e-6):
                    failures.append(f"{split}: sample {expected_index} canonical points mismatch")
                if not np.allclose(arrays["probe_values"], probe, atol=1.0e-6):
                    failures.append(f"{split}: sample {expected_index} probe values mismatch")
                direction = np.asarray(sample["incident_direction"], dtype=np.float64)
                incident = formal.incident_on_interfaces(centers, points, direction, 2.0)
                if not np.allclose(arrays["u_inc"], incident, atol=2.0e-6):
                    failures.append(f"{split}: sample {expected_index} incident wave mismatch")
                q64 = formal.project_flux(arrays["Z"].astype(np.complex128), arrays["u_total_gt"].astype(np.complex128), pinv)
                if not np.allclose(arrays["q64_gt"], q64, atol=3.0e-5, rtol=3.0e-5):
                    failures.append(f"{split}: sample {expected_index} q64 mismatch")
                for row, item in enumerate(sample["objects"]):
                    size = np.asarray(item["size"], dtype=np.float64)
                    if not np.allclose(arrays["shape_parameters"][row, : len(size)], size, atol=1.0e-6):
                        failures.append(f"{split}: sample {expected_index} stored geometry mismatch")
                    if not np.allclose(arrays["rotations"][row], item["rotation"], atol=1.0e-6):
                        failures.append(f"{split}: sample {expected_index} stored rotation mismatch")
        if split != "count_ood_n8":
            quotient, remainder = divmod(expected_count, 3)
            expected_distribution = {2: quotient + (remainder >= 1), 3: quotient + (remainder >= 2), 4: quotient}
        else:
            expected_distribution = {8: expected_count}
        if dict(object_counts) != expected_distribution:
            failures.append(f"{split}: object counts {dict(object_counts)} differ from {expected_distribution}")
        report["splits"][split] = {
            "samples": len(data.get("samples", [])),
            "library_members": len(entries),
            "object_counts": dict(object_counts),
            "shape_counts": dict(shape_counts),
        }
    if len(gate_hashes) != 1 or None in gate_hashes:
        failures.append("numerical gate differs across splits")
    report["scattering_matrix_unitarity_max"] = max(unitarity_defects) if unitarity_defects else None
    report["local_fem_max_free_dof_relative_residual"] = max(local_fem_residuals) if local_fem_residuals else None
    report["local_fem_max_modal_superposition_relative_error"] = max(local_superposition_errors) if local_superposition_errors else None
    report["mesh_element_quality_min"] = min(mesh_quality_minima) if mesh_quality_minima else None
    report["passed"] = not failures
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("datasets/canonical_response_v3"))
    parser.add_argument("--profile", choices=tuple(PROFILE), default="full")
    parser.add_argument("--splits", nargs="+", choices=SPLITS, default=("train", "val"))
    parser.add_argument("--skip-hashes", action="store_true")
    parser.add_argument("--report", type=Path, default=Path("outputs/tmatrix_dataset_validation.json"))
    args = parser.parse_args()
    report = validate(args.root, args.profile, not args.skip_hashes, tuple(args.splits))
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"passed": report["passed"], "splits": report["splits"], "failures": report["failures"][:10]}, indent=2))
    if not report["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
