"""Validate formal CSP-GNet datasets before training."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np

from .formal_neumann_3d import SCHEMA_VERSION, project_flux


def _scene_hash(sample: dict) -> str:
    payload = {
        "k": sample["k"],
        "incident_direction": sample["incident_direction"],
        "centers": sample["centers"],
        "objects": sample["objects"],
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def validate_dataset(root: Path, full_scan: bool = False, allow_unsafe: bool = False) -> dict:
    manifest_path = root / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"missing manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    failures: list[str] = []
    if manifest.get("schema_version") != SCHEMA_VERSION:
        failures.append("schema_version mismatch")
    if manifest.get("status") != "complete":
        failures.append("manifest status is not complete")
    if manifest.get("mesh_strategy") != "graded-local-FEM":
        failures.append("new datasets must use graded local FEM meshes")
    profiles = manifest.get("graded_mesh_profiles_by_family", {})
    if not profiles or any(name not in {"G0", "G1", "G2", "G3"} for name in profiles.values()):
        failures.append("graded mesh profile map is missing")
    calibration = manifest.get("graded_calibration")
    if (not calibration or not calibration.get("path") or not calibration.get("sha256")) and not allow_unsafe:
        failures.append("graded calibration provenance is missing")
    elif calibration and not Path(calibration["path"]).is_file():
        failures.append("graded calibration report is missing")
    elif calibration and hashlib.sha256(Path(calibration["path"]).read_bytes()).hexdigest() != calibration["sha256"]:
        failures.append("graded calibration report SHA-256 mismatch")
    if not allow_unsafe:
        gate = manifest.get("numerical_gate")
        if not manifest.get("formal_evidence") or not gate:
            failures.append("dataset was generated without a passed numerical gate")
        elif gate:
            gate_path = Path(gate.get("path", ""))
            if not gate_path.is_file():
                failures.append("numerical-gate report is missing")
            else:
                digest = hashlib.sha256(gate_path.read_bytes()).hexdigest()
                if digest != gate.get("sha256"):
                    failures.append("numerical-gate report SHA-256 mismatch")
                gate_report = json.loads(gate_path.read_text(encoding="utf-8"))
                if not gate_report.get("passed"):
                    failures.append("referenced numerical-gate report did not pass")
                if (
                    gate_report.get("mesh_strategy") != "graded-local-FEM"
                    or not gate_report.get("bem_label_audit_passed")
                    or gate_report.get("graded_calibration_sha256") != (calibration or {}).get("sha256")
                ):
                    failures.append("release gate does not certify the recorded graded mesh/BEM calibration")
    samples = manifest.get("samples", [])
    if len(samples) != int(manifest.get("num_samples", -1)):
        failures.append("manifest sample count does not match num_samples")

    files = sorted(root.glob("sample_*.npz"))
    if len(files) != len(samples):
        failures.append("npz file count does not match manifest")
    sample_by_file = {item.get("file"): item for item in samples}
    if set(sample_by_file) != {file.name for file in files}:
        failures.append("manifest filenames do not match sample files")
    indices = range(len(files)) if full_scan else sorted(set([0, len(files) // 2, len(files) - 1]))
    shapes = Counter()
    object_counts = Counter()
    maximum_q_error = 0.0
    minimum_center_distance = float("inf")
    reference_phi = None
    for index in indices:
        if index < 0 or not files:
            continue
        with np.load(files[index]) as loaded:
            data = {key: loaded[key] for key in loaded.files}
        required = {
            "Z", "u_total_gt", "u_inc", "q64_gt", "probe_values", "phi_pinv",
            "interface_points", "interface_faces", "centers", "rotations", "shape_names",
            "shape_parameters", "axis_permutations", "k", "incident_direction",
            "object_count", "schema_version", "spherical_harmonic_convention",
        }
        missing = required.difference(data)
        if missing:
            failures.append(f"{files[index].name}: missing {sorted(missing)}")
            continue
        objects = int(data["object_count"])
        object_counts[objects] += 1
        shapes.update(str(name) for name in data["shape_names"].tolist())
        if data["Z"].shape != (objects, 128, 64):
            failures.append(f"{files[index].name}: invalid Z shape {data['Z'].shape}")
        if data["probe_values"].shape != (128, 64) or data["phi_pinv"].shape != (64, 128):
            failures.append(f"{files[index].name}: invalid probe matrix or weighted pseudoinverse")
        if data["interface_points"].shape != (128, 3):
            failures.append(f"{files[index].name}: invalid canonical interface coordinates")
        if data["interface_faces"].ndim != 2 or data["interface_faces"].shape[1] != 3:
            failures.append(f"{files[index].name}: canonical interface faces are not triangles")
        if str(data["spherical_harmonic_convention"].item()) != (
            "ngsolve_no_condon_shortley_positive_m"
        ):
            failures.append(f"{files[index].name}: unsupported spherical-harmonic convention")
        for key in ("u_total_gt", "u_inc", "q64_gt"):
            if data[key].shape != (objects, 128):
                failures.append(f"{files[index].name}: invalid {key} shape {data[key].shape}")
        if not all(np.isfinite(data[key]).all() for key in ("Z", "u_total_gt", "u_inc", "q64_gt")):
            failures.append(f"{files[index].name}: non-finite field values")
        if abs(float(data["k"]) - 2.0) > 1.0e-7:
            failures.append(f"{files[index].name}: k is not 2.0")

        sample = sample_by_file.get(files[index].name)
        if sample is None:
            continue
        sample_profiles = sample.get("local_mesh_profiles_by_object", [])
        if len(sample_profiles) != int(data["object_count"]):
            failures.append(f"{files[index].name}: per-object graded mesh profile provenance is missing")
        if not allow_unsafe:
            digest = hashlib.sha256(files[index].read_bytes()).hexdigest()
            if digest != sample.get("sha256"):
                failures.append(f"{files[index].name}: SHA-256 does not match manifest")
        bem = sample.get("bem", {})
        if int(bem.get("gmres_info", 1)) != 0:
            failures.append(f"{files[index].name}: BEM solve did not converge")
        if float(bem.get("gmres_final_residual", float("inf"))) > float(
            manifest.get("gmres_tolerance", 0.0)
        ):
            failures.append(f"{files[index].name}: BEM residual exceeds manifest tolerance")
        expected_names = np.asarray([obj["shape"] for obj in sample["objects"]])
        if not np.array_equal(data["shape_names"].astype(str), expected_names.astype(str)):
            failures.append(f"{files[index].name}: shape names disagree with manifest")
        expected_parameters = np.full((objects, 3), np.nan, dtype=np.float32)
        expected_permutations = np.full((objects, 3), -1, dtype=np.int8)
        for row, obj in enumerate(sample["objects"]):
            expected_parameters[row, : len(obj["size"])] = obj["size"]
            permutation = obj.get("axis_permutation", [])
            expected_permutations[row, : len(permutation)] = permutation
        if not np.allclose(
            data["shape_parameters"], expected_parameters, equal_nan=True, atol=1.0e-6
        ):
            failures.append(f"{files[index].name}: shape parameters disagree with manifest")
        if not np.array_equal(data["axis_permutations"], expected_permutations):
            failures.append(f"{files[index].name}: axis permutations disagree with manifest")
        if not np.allclose(data["centers"], sample["centers"], atol=1.0e-6):
            failures.append(f"{files[index].name}: centers disagree with manifest")
        expected_rotations = np.asarray([obj["rotation"] for obj in sample["objects"]])
        if not np.allclose(data["rotations"], expected_rotations, atol=1.0e-6):
            failures.append(f"{files[index].name}: rotations disagree with manifest")
        if not np.allclose(
            data["incident_direction"], sample["incident_direction"], atol=1.0e-6
        ):
            failures.append(f"{files[index].name}: incident direction disagrees with manifest")
        if not np.isclose(np.linalg.norm(data["incident_direction"]), 1.0, atol=1.0e-6):
            failures.append(f"{files[index].name}: incident direction is not a unit vector")

        rotations = np.asarray(data["rotations"], dtype=np.float64)
        identity_error = np.max(np.abs(rotations @ rotations.transpose(0, 2, 1) - np.eye(3)))
        determinant_error = np.max(np.abs(np.linalg.det(rotations) - 1.0))
        if identity_error > 1.0e-5 or determinant_error > 1.0e-5:
            failures.append(f"{files[index].name}: invalid SO(3) rotation")
        centers = np.asarray(data["centers"], dtype=np.float64)
        if objects > 1:
            distances = np.linalg.norm(centers[:, None, :] - centers[None, :, :], axis=-1)
            distances += np.eye(objects) * 1.0e9
            nearest = float(distances.min())
            minimum_center_distance = min(minimum_center_distance, nearest)
            if nearest < 3.6 - 1.0e-5:
                failures.append(f"{files[index].name}: canonical surfaces violate spacing")

        recomputed_q = project_flux(
            np.asarray(data["Z"], dtype=np.complex128),
            np.asarray(data["u_total_gt"], dtype=np.complex128),
            np.asarray(data["phi_pinv"], dtype=np.complex128),
        )
        q_error = np.linalg.norm(recomputed_q - data["q64_gt"]) / max(
            np.linalg.norm(data["q64_gt"]), 1.0e-12
        )
        maximum_q_error = max(maximum_q_error, float(q_error))
        if q_error > 2.0e-5:
            failures.append(f"{files[index].name}: stored q64_gt is inconsistent")
        if reference_phi is None:
            reference_phi = data["probe_values"]
        elif not np.allclose(reference_phi, data["probe_values"], atol=1.0e-6, rtol=1.0e-6):
            failures.append(f"{files[index].name}: canonical probe matrix changed")

    hashes = [_scene_hash(sample) for sample in samples]
    duplicate_scenes = len(hashes) - len(set(hashes))
    if duplicate_scenes:
        failures.append(f"{duplicate_scenes} duplicate scene parameter sets")
    expected_counts = [
        index % (int(manifest.get("count_range", [2, 4])[1]) - int(manifest.get("count_range", [2, 4])[0]) + 1)
        + int(manifest.get("count_range", [2, 4])[0])
        for index in range(len(samples))
    ]
    observed_counts = [len(sample.get("objects", [])) for sample in samples]
    if observed_counts != expected_counts:
        failures.append("manifest object counts are not balanced in deterministic order")
    if manifest.get("split") == "count_ood_n8" and any(count != 8 for count in observed_counts):
        failures.append("count_ood_n8 contains a sample whose object count is not eight")
    return {
        "root": str(root.resolve()),
        "passed": not failures,
        "failures": failures,
        "manifest_samples": len(samples),
        "files": len(files),
        "scanned_files": len(list(indices)),
        "object_counts": dict(sorted(object_counts.items())),
        "shapes": dict(sorted(shapes.items())),
        "minimum_center_distance": minimum_center_distance,
        "maximum_q_reconstruction_error": maximum_q_error,
        "scene_hashes": hashes,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", action="append", required=True)
    parser.add_argument("--full-scan", action="store_true")
    parser.add_argument("--allow-unsafe", action="store_true")
    parser.add_argument("--report", required=True)
    args = parser.parse_args()
    reports = [
        validate_dataset(
            Path(root), full_scan=args.full_scan, allow_unsafe=args.allow_unsafe
        )
        for root in args.dataset
    ]
    failures = []
    seen: dict[str, str] = {}
    for report in reports:
        for scene_hash in report.pop("scene_hashes"):
            if scene_hash in seen:
                failures.append(
                    f"cross-split duplicate between {seen[scene_hash]} and {report['root']}"
                )
            else:
                seen[scene_hash] = report["root"]
    passed = all(report["passed"] for report in reports) and not failures
    result = {"passed": passed, "cross_split_failures": failures, "datasets": reports}
    output = Path(args.report)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"passed": passed, "report": str(output)}, ensure_ascii=False))
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
