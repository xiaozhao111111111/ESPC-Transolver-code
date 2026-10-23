"""Versioned, resumable finite-library Neumann T-matrix dataset generation.

Only use after ``scripts/validate_tmatrix_neumann.py`` has passed.  Each new
ellipsoid/box library member independently checks its 128-by-64 fingerprint
against a finer local FEM mesh before any scene can use it.  Labels are from
the T-matrix multiple-scattering solver, not per-scene BEM.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import time
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.stats import qmc

from . import formal_neumann_3d as formal
from . import tmatrix_neumann_3d as tmatrix


SCHEMA = "cspgnet-neumann-tmatrix-v2"
SPLITS = ("train", "val", "test", "count_ood_n8")
SPLIT_SEEDS = {"train": 191_731, "val": 283_849, "test": 379_987, "count_ood_n8": 467_041}
PROFILE = {
    "pilot": {"samples": (300, 60, 60, 30), "library_per_shape": (6, 2, 2, 2)},
    "full": {"samples": (10_000, 1_000, 1_000, 1_000), "library_per_shape": (32, 8, 8, 8)},
}
LOCAL_LIMITS = {"z_frobenius": 0.03, "z_mode_p95": 0.05}
DIRECT_FEM_LIMITS = {"modal_superposition": 1.0e-6, "free_dof_residual": 1.0e-8}
GRADED_CALIBRATION_SCHEMA = "graded-local-calibration-summary-v1"
RELEASE_GATE_SCHEMA = "canonical-response-v3-graded-release-gate-v1"
DEFAULT_RELEASE_GATE = Path("outputs/canonical_response_v3_graded_release_gate_v3_20260926.json")
DEFAULT_DATA_ROOT = Path("datasets/canonical_response_v3")


def _provenance() -> dict:
    versions = {}
    for package in ("numpy", "scipy", "ngsolve", "bempp-cl"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "not installed in generator process"
    repository = Path(__file__).resolve().parents[3]
    git = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repository,
        capture_output=True, text=True, check=False,
    )
    source = {
        "tmatrix_dataset": _sha256(Path(__file__)),
        "tmatrix_neumann_3d": _sha256(Path(tmatrix.__file__)),
        "formal_neumann_3d": _sha256(Path(formal.__file__)),
    }
    return {
        "python": platform.python_version(),
        "packages": versions,
        "git_commit": git.stdout.strip() if git.returncode == 0 else None,
        "source_sha256": source,
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".pending")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(temporary, path)


def _npz_write(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".pending")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    os.replace(temporary, path)


def _identity() -> tuple[tuple[float, float, float], ...]:
    return tuple(tuple(float(value) for value in row) for row in np.eye(3))


def _shape_entries(split: str, count_per_shape: int, profile: str) -> list[dict]:
    entries = []
    for shape_index, shape in enumerate(formal.SHAPE_NAMES):
        dimensions = 1 if shape == "sphere" else 3
        seed_material = f"{SCHEMA}|{profile}|shape-library|{split}|{shape}|{shape_index}"
        seed = int.from_bytes(hashlib.sha256(seed_material.encode()).digest()[:4], "little")
        cube = qmc.LatinHypercube(d=dimensions, seed=seed).random(count_per_shape)
        lower, upper = (0.25, 0.55) if shape != "box" else (0.22, 0.50)
        for index, row in enumerate(cube):
            size = tuple(float(lower + (upper - lower) * value) for value in row)
            if shape == "box" and np.linalg.norm(size) > 0.9:
                raise AssertionError("sampled box violates the canonical obstacle radius")
            entries.append(
                {
                    "id": f"{split}-{shape}-{index:03d}",
                    "shape": shape,
                    "size": list(size),
                    "axis_permutation": [] if shape == "sphere" else [0, 1, 2],
                    "generation_seed": seed,
                    "file": f"{split}-{shape}-{index:03d}.npz",
                    "status": "pending",
                }
            )
    return entries


def _same_specs(existing: list[dict], planned: list[dict]) -> bool:
    keys = ("id", "shape", "size", "axis_permutation", "generation_seed", "file")
    return len(existing) == len(planned) and all(
        all(left[key] == right[key] for key in keys)
        for left, right in zip(existing, planned)
    )


def _gate_report(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"current graded/BEM release gate is missing: {path}")
    gate = json.loads(path.read_text(encoding="utf-8"))
    if (gate.get("schema_version") != RELEASE_GATE_SCHEMA or not gate.get("passed")
            or gate.get("status") != "passed"
            or gate.get("mesh_strategy") != "graded-local-FEM"
            or gate.get("bem_label_audit_passed") is not True):
        raise RuntimeError(f"current graded FEM plus independent BEM release gate has not passed: {path}")
    if not gate.get("tests") or not all(test.get("passed") for test in gate["tests"]):
        raise RuntimeError("reliability report is internally inconsistent")
    for raw_path, expected in gate.get("source_sha256", {}).items():
        source_path = Path(raw_path)
        if not source_path.is_absolute():
            source_path = Path(__file__).resolve().parents[3] / source_path
        if not source_path.is_file() or _sha256(source_path) != expected:
            raise RuntimeError(f"release-gate source changed after validation: {source_path}")
    if not gate.get("graded_calibration_sha256"):
        raise RuntimeError("release gate does not bind the graded calibration evidence")
    return {
        "path": str(path.resolve()), "sha256": _sha256(path), "passed": True,
        "schema_version": gate["schema_version"],
        "mesh_strategy": gate["mesh_strategy"],
        "graded_calibration_sha256": gate["graded_calibration_sha256"],
        "bem_label_audit_passed": True,
    }


def _graded_calibration_report(path: Path) -> dict:
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("schema") != GRADED_CALIBRATION_SCHEMA or report.get("overall_status") != "passed":
        raise RuntimeError(f"graded-mesh calibration has not passed: {path}")
    if (len(report.get("results", [])) != 15
            or not report.get("all_per_geometry_reports_passed")
            or not report.get("adjacent_mesh_gates_passed")
            or not report.get("direct_fem_gates_passed")
            or not report.get("analytic_sphere_gate_passed")):
        raise RuntimeError("graded-mesh summary does not contain the complete 15-geometry release evidence")
    profiles = {
        shape: values.get("candidate_profile")
        for shape, values in report.get("profile_by_family", {}).items()
    }
    if not set(formal.SHAPE_NAMES).issubset(profiles) or any(name not in tmatrix.GRADED_MESH_PROFILES for name in profiles.values()):
        raise RuntimeError("graded calibration is missing a valid candidate profile for every generated shape family")
    return {
        "path": str(path.resolve()),
        "sha256": _sha256(path),
        "passed": True,
        "schema": report["schema"],
        "candidate_profiles_by_family": profiles,
    }


def build_library(
    root: Path, profile: str, gate: dict, graded: dict,
    split_names: tuple[str, ...] = SPLITS,
) -> dict[str, dict]:
    profile_config = PROFILE[profile]
    if gate.get("graded_calibration_sha256") != graded.get("sha256"):
        raise RuntimeError("release gate and graded calibration checksum do not match")
    if not split_names or len(set(split_names)) != len(split_names) or any(name not in SPLITS for name in split_names):
        raise ValueError(f"invalid split selection: {split_names}")
    points, _ = formal.canonical_surface()
    manifests = {}
    split_counts = dict(zip(SPLITS, profile_config["library_per_shape"]))
    for split in split_names:
        per_shape = split_counts[split]
        split_root = root / "library" / split
        manifest_path = split_root / "manifest.json"
        planned = _shape_entries(split, per_shape, profile)
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest.get("status") == "failed" or any(
                entry.get("status") == "failed" for entry in manifest.get("entries", [])
            ):
                raise RuntimeError(
                    f"a prior local-operator failure is retained and blocks this dataset version: {manifest_path}"
                )
            if (
                not _same_specs(manifest["entries"], planned)
                or manifest["numerical_gate"]["sha256"] != gate["sha256"]
                or manifest.get("graded_mesh_calibration", {}).get("sha256") != graded["sha256"]
            ):
                raise RuntimeError(f"existing shape library differs from requested configuration: {manifest_path}")
            if profile == "full" and manifest.get("solver_provenance", {}).get("source_sha256") != _provenance()["source_sha256"]:
                raise RuntimeError(f"source code changed during full library generation: {manifest_path}")
        else:
            manifest = {
                "schema_version": SCHEMA,
                "type": "shape_library",
                "profile": profile,
                "split": split,
                "numerical_gate": gate,
                "mesh_strategy": "graded-local-FEM",
                "graded_mesh_calibration": graded,
                "graded_mesh_profiles_by_family": graded["candidate_profiles_by_family"],
                "solver_provenance": _provenance(),
                "local_mesh_profiles_compared_by_family": {
                    family: [
                        graded["candidate_profiles_by_family"][family],
                        tuple(tmatrix.GRADED_MESH_PROFILES)[
                            tuple(tmatrix.GRADED_MESH_PROFILES).index(
                                graded["candidate_profiles_by_family"][family]
                            ) + 1
                        ],
                    ]
                    for family in formal.SHAPE_NAMES
                },
                "fem_order": 2,
                "local_thresholds": LOCAL_LIMITS,
                "direct_fem_thresholds": DIRECT_FEM_LIMITS,
                "status": "incomplete",
                "entries": planned,
            }
            _json_write(manifest_path, manifest)

        for entry in manifest["entries"]:
            file = split_root / entry["file"]
            if entry["status"] == "complete":
                if not file.exists() or _sha256(file) != entry.get("sha256"):
                    raise RuntimeError(f"cached operator is missing or corrupt: {file}")
                continue
            recover_unlisted_file = file.exists()
            spec = formal.ObjectSpec(
                entry["shape"], tuple(entry["size"]), _identity(),
                tuple(entry["axis_permutation"]),
            )
            started = time.perf_counter()
            coarse_name = graded["candidate_profiles_by_family"][spec.shape]
            profile_names = tuple(tmatrix.GRADED_MESH_PROFILES)
            coarse_index = profile_names.index(coarse_name)
            if coarse_index + 1 >= len(profile_names):
                raise RuntimeError(f"no finer graded profile is available after {coarse_name}")
            coarse_profile = tmatrix.GRADED_MESH_PROFILES[coarse_name]
            fine_profile = tmatrix.GRADED_MESH_PROFILES[profile_names[coarse_index + 1]]
            modal, coarse_meta = tmatrix.weak_modal_dtn(
                spec, coarse_profile.background_h, input_lmax=9, output_lmax=10,
                mesh_profile=coarse_profile,
            )
            finer, fine_meta = tmatrix.weak_modal_dtn(
                spec, fine_profile.background_h, input_lmax=9, output_lmax=10,
                mesh_profile=fine_profile,
            )
            z_coarse = tmatrix.nodal_fingerprint(modal, points)
            z_fine = tmatrix.nodal_fingerprint(finer, points)
            difference = z_coarse - z_fine
            mode_errors = np.linalg.norm(difference, axis=0) / np.maximum(
                np.linalg.norm(z_fine, axis=0), 1.0e-14
            )
            local_metrics = {
                "method": "curved P2 graded weak FEM DtN",
                "mesh_profiles": [coarse_profile.as_dict(), fine_profile.as_dict()],
                "z_frobenius": float(np.linalg.norm(difference) / max(np.linalg.norm(z_fine), 1.0e-14)),
                "z_mode_p95": float(np.percentile(mode_errors, 95)),
                "coarse": coarse_meta,
                "fine": fine_meta,
            }
            coefficient_rng = np.random.default_rng(int(entry["generation_seed"]))
            coefficients = (
                coefficient_rng.standard_normal(100)
                + 1j * coefficient_rng.standard_normal(100)
            ) / np.sqrt(2.0)
            direct_checks = {}
            for label, candidate, mesh in (
                ("coarse", modal, coarse_profile), ("fine", finer, fine_profile),
            ):
                direct_flux, direct_meta = tmatrix.weak_flux_for_harmonic_coefficients(
                    spec, mesh.background_h, coefficients, output_lmax=10,
                    fem_order=2, mesh_profile=mesh,
                )
                predicted_flux = candidate @ coefficients
                direct_error = float(
                    np.linalg.norm(direct_flux[:, 0] - predicted_flux)
                    / max(np.linalg.norm(direct_flux[:, 0]), 1.0e-15)
                )
                direct_checks[label] = {
                    **direct_meta,
                    "modal_superposition_relative_error": direct_error,
                }
            local_metrics["direct_fem"] = direct_checks
            local_pass = (
                all(local_metrics[key] < limit for key, limit in LOCAL_LIMITS.items())
                and all(
                    item["modal_superposition_relative_error"] < DIRECT_FEM_LIMITS["modal_superposition"]
                    and item["max_free_dof_relative_residual"] < DIRECT_FEM_LIMITS["free_dof_residual"]
                    and item["elements_nonpositive_volume"] == 0
                    and item["element_quality_min"] > 0.0
                    for item in direct_checks.values()
                )
                and all(
                    mesh.get("elements_nonpositive_volume") == 0
                    and mesh.get("element_quality_min", 0.0) > 0.0
                    and {"canonical", "scatterer"}.issubset(mesh.get("boundary_size_profile", {}))
                    for mesh in (coarse_meta, fine_meta)
                )
            )
            if not local_pass:
                entry["status"] = "failed"
                entry["local_metrics"] = local_metrics
                manifest["status"] = "failed"
                _json_write(manifest_path, manifest)
                raise RuntimeError(f"local graded FEM/Z/direct-residual gate failed for {entry['id']}: {local_metrics}")
            transition, conversion = tmatrix.transition_matrix(modal, lmax=9)
            if conversion["conversion_residual"] >= 1.0e-10:
                raise ArithmeticError(f"T conversion residual failed for {entry['id']}")
            arrays = {
                "modal_dtn": modal,
                "transition": transition,
                "size": np.asarray(entry["size"], dtype=np.float64),
            }
            if recover_unlisted_file:
                with np.load(file) as stored:
                    if set(stored.files) != set(arrays) or any(
                        not np.array_equal(stored[name], expected)
                        for name, expected in arrays.items()
                    ):
                        raise RuntimeError(f"unlisted operator differs from reconstruction: {file}")
            else:
                _npz_write(file, **arrays)
            entry.update(
                status="complete", sha256=_sha256(file),
                seconds=float(time.perf_counter() - started),
                local_metrics=local_metrics, conversion=conversion,
            )
            _json_write(manifest_path, manifest)
            print(f"library {split} {entry['id']} {entry['seconds']:.1f}s", flush=True)
        manifest["status"] = "complete"
        _json_write(manifest_path, manifest)
        manifests[split] = manifest
    return manifests


def _scene_seed(split: str, index: int, profile: str) -> int:
    material = f"{SCHEMA}|2026-09|{profile}|scene|{split}|{index}"
    return int.from_bytes(hashlib.sha256(material.encode()).digest()[:4], "little")


def _scene_spec(split: str, index: int, profile: str) -> formal.SampleSpec:
    count = 8 if split == "count_ood_n8" else formal.balanced_object_count(index, 2, 4)
    return formal.sample_spec(
        index, split, _scene_seed(split, 0, profile), count, count
    )


def generate_split(root: Path, profile: str, split: str, library: dict, gate: dict) -> dict:
    count = PROFILE[profile]["samples"][SPLITS.index(split)]
    output = root / "data" / split
    manifest_path = output / "manifest.json"
    library_entries = library["entries"]
    by_shape = {
        shape: [entry for entry in library_entries if entry["shape"] == shape]
        for shape in formal.SHAPE_NAMES
    }
    if any(not values for values in by_shape.values()):
        raise RuntimeError("shape library is missing a family")
    library_hash = _sha256(root / "library" / split / "manifest.json")
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            manifest.get("schema_version") != SCHEMA
            or manifest.get("profile") != profile
            or manifest.get("num_samples") != count
            or manifest.get("library_sha256") != library_hash
            or manifest.get("numerical_gate", {}).get("sha256") != gate["sha256"]
            or manifest.get("mesh_strategy") != "graded-local-FEM"
            or manifest.get("graded_mesh_calibration", {}).get("sha256")
                != library.get("graded_mesh_calibration", {}).get("sha256")
        ):
            raise RuntimeError(f"existing dataset differs from requested configuration: {manifest_path}")
        if profile == "full" and manifest.get("solver_provenance", {}).get("source_sha256") != _provenance()["source_sha256"]:
            raise RuntimeError(f"source code changed during full split generation: {manifest_path}")
    else:
        manifest = {
            "schema_version": SCHEMA,
            "status": "incomplete",
            "formal_evidence": False,
            "validated_tmatrix_evidence": False,
            "label_solver": "T-matrix Foldy-Lax (not per-scene BEM)",
            "profile": profile,
            "split": split,
            "num_samples": count,
            "count_range": [8, 8] if split == "count_ood_n8" else [2, 4],
            "seed": _scene_seed(split, 0, profile),
            "seed_derivation": "sha256(canonical-response-v3|2026-09|profile|scene|split|index)",
            "k": 2.0,
            "canonical_radius": 1.5,
            "interface_nodes": 128,
            "probe_lmax": 7,
            "probe_count": 64,
            "tmatrix_lmax": 9,
            "numerical_gate": gate,
            "mesh_strategy": "graded-local-FEM",
            "graded_mesh_calibration": library["graded_mesh_calibration"],
            "graded_mesh_profiles_by_family": library["graded_mesh_profiles_by_family"],
            "solver_provenance": _provenance(),
            "library_manifest": str((root / "library" / split / "manifest.json").resolve()),
            "library_sha256": library_hash,
            "samples": [],
        }
        _json_write(manifest_path, manifest)

    points, faces = formal.canonical_surface()
    probes = formal.spherical_harmonic_matrix(points, formal.spherical_modes(7))
    pinv = formal.weighted_pseudoinverse(
        probes, formal.equal_area_weights(128, formal.DEFAULT_CANONICAL_RADIUS)
    )
    quadrature = tmatrix.angular_quadrature()
    operators = {}
    for entry in library_entries:
        file = root / "library" / split / entry["file"]
        with np.load(file) as data:
            operators[entry["id"]] = (data["modal_dtn"], data["transition"])
    records = {int(record["index"]): record for record in manifest["samples"]}
    started = time.perf_counter()
    for index in range(count):
        path = output / f"sample_{index:05d}.npz"
        if index in records:
            if not path.exists() or _sha256(path) != records[index].get("sha256"):
                raise RuntimeError(f"existing sample is missing or corrupt: {path}")
            continue
        recover_unlisted_file = path.exists()
        spec = _scene_spec(split, index, profile)
        centers = np.asarray(spec.centers, dtype=np.float64)
        direction = np.asarray(spec.incident_direction, dtype=np.float64)
        rng = np.random.default_rng(spec.sample_seed + 17)
        selected = []
        rotations = []
        modal = []
        transitions = []
        for _ in spec.objects:
            shape = formal.SHAPE_NAMES[int(rng.integers(0, len(formal.SHAPE_NAMES)))]
            entry = by_shape[shape][int(rng.integers(0, len(by_shape[shape])))]
            rotation = np.eye(3) if shape == "sphere" else formal.random_rotation_matrix(rng)
            base_modal, base_transition = operators[entry["id"]]
            if shape == "sphere":
                current_modal, current_transition = base_modal, base_transition
            else:
                d9 = tmatrix.rotation_matrix(rotation, 9, quadrature)
                d10 = tmatrix.rotation_matrix(rotation, 10, quadrature)
                current_modal = d10 @ base_modal @ d9.conj().T
                current_transition = d9 @ base_transition @ d9.conj().T
            selected.append(entry)
            rotations.append(rotation)
            modal.append(current_modal)
            transitions.append(current_transition)
        z = np.asarray([tmatrix.nodal_fingerprint(item, points) for item in modal])
        total, solve_meta = tmatrix.solve_scene(
            transitions, centers, direction, points, quadrature=quadrature
        )
        incident = formal.incident_on_interfaces(centers, points, direction, spec.k)
        q64 = formal.project_flux(z, total, pinv)
        if solve_meta["linear_residual"] > 1.0e-10:
            raise ArithmeticError("Foldy-Lax linear residual exceeds threshold")
        for name, array in {"Z": z, "u_total_gt": total, "u_inc": incident, "q64_gt": q64}.items():
            if array.dtype != np.complex128 or not np.isfinite(array).all():
                raise FloatingPointError(f"{name} failed complex128 validation")
        shapes = np.asarray([formal.SHAPE_NAMES.index(item["shape"]) for item in selected], dtype=np.int8)
        sizes = np.zeros((len(selected), 3), dtype=np.float32)
        for row, item in enumerate(selected):
            sizes[row, : len(item["size"])] = item["size"]
        arrays = {
            "Z": z.astype(np.complex64),
            "u_total_gt": total.astype(np.complex64),
            "u_inc": incident.astype(np.complex64),
            "q64_gt": q64.astype(np.complex64),
            "probe_values": probes.astype(np.complex64),
            "phi_pinv": pinv.astype(np.complex64),
            "interface_points": points.astype(np.float32),
            "interface_faces": faces,
            "centers": centers.astype(np.float32),
            "rotations": np.asarray(rotations, dtype=np.float32),
            "shape_parameters": sizes,
            "shape_codes": shapes,
            "incident_direction": direction.astype(np.float32),
            "k": np.asarray(spec.k, dtype=np.float32),
            "sample_seed": np.asarray(spec.sample_seed, dtype=np.int64),
        }
        if recover_unlisted_file:
            with np.load(path) as stored:
                if set(stored.files) != set(arrays) or any(
                    not np.array_equal(stored[name], expected)
                    for name, expected in arrays.items()
                ):
                    raise RuntimeError(f"unlisted sample differs from deterministic reconstruction: {path}")
        else:
            _npz_write(path, **arrays)
        record = {
            "index": index,
            "file": path.name,
            "sha256": _sha256(path),
            "sample_seed": spec.sample_seed,
            "seed_derivation": "sha256-derived split base plus fixed 1000003*index stride",
            "k": spec.k,
            "centers": centers.tolist(),
            "incident_direction": direction.tolist(),
            "objects": [
                {
                    "library_id": item["id"],
                    "shape": item["shape"],
                    "size": item["size"],
                    "axis_permutation": item["axis_permutation"],
                    "rotation": rotation.tolist(),
                }
                for item, rotation in zip(selected, rotations)
            ],
            "solve": solve_meta,
        }
        scene_identity = {
            "k": spec.k,
            "incident_direction": record["incident_direction"],
            "centers": record["centers"],
            "objects": record["objects"],
        }
        record["physical_scene_sha256"] = hashlib.sha256(
            json.dumps(scene_identity, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        manifest["samples"].append(record)
        if (index + 1) % 50 == 0 or index + 1 == count:
            _json_write(manifest_path, manifest)
            print(f"data {profile} {split}: {index + 1}/{count}", flush=True)
    if len(manifest["samples"]) != count:
        raise RuntimeError("split generation ended with an incomplete manifest")
    manifest["status"] = "complete"
    manifest["validated_tmatrix_evidence"] = profile == "full"
    manifest["generation_seconds_this_run"] = float(time.perf_counter() - started)
    manifest["object_counts"] = dict(Counter(len(record["objects"]) for record in manifest["samples"]))
    _json_write(manifest_path, manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--profile", choices=tuple(PROFILE), default="full")
    parser.add_argument("--stage", choices=("library", "data", "all"), default="all")
    parser.add_argument("--splits", nargs="+", choices=SPLITS, default=("train", "val"))
    parser.add_argument("--gate", type=Path, default=DEFAULT_RELEASE_GATE)
    parser.add_argument(
        "--graded-calibration",
        type=Path,
        default=Path("outputs/v3_graded_mesh_calibration_v2/summary.json"),
        help="passing four-level graded local-operator calibration summary",
    )
    args = parser.parse_args()
    gate = _gate_report(args.gate)
    graded = _graded_calibration_report(args.graded_calibration)
    if gate["graded_calibration_sha256"] != graded["sha256"]:
        raise RuntimeError("release gate is bound to a different graded calibration summary")
    if args.stage in ("library", "all"):
        build_library(args.root / args.profile, args.profile, gate, graded, tuple(args.splits))
    if args.stage in ("data", "all"):
        profile_root = args.root / args.profile
        libraries = {}
        for split in args.splits:
            path = profile_root / "library" / split / "manifest.json"
            library = json.loads(path.read_text(encoding="utf-8"))
            if (
                library["status"] != "complete"
                or library["numerical_gate"]["sha256"] != gate["sha256"]
                or library.get("graded_mesh_calibration", {}).get("sha256") != graded["sha256"]
            ):
                raise RuntimeError(f"split shape library is not complete or gate-approved: {path}")
            libraries[split] = library
        for split in args.splits:
            generate_split(profile_root, args.profile, split, libraries[split], gate)


if __name__ == "__main__":
    main()
