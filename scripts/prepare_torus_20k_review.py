"""Prepare an immutable Torus review sidecar using mixed-20k training scales.

No FEM/BEM solve, label edit, or normalization refit is performed. Existing
source-hashed surface clouds and saved-Z response conversion are reused.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import shutil
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import numpy as np
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
import prepare_v6s11_20k_all_methods as packing
from cspgnet.data_generation.structural_response import scattering_response


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def response_features(data, stats):
    response = scattering_response(data)
    scales = np.repeat(np.asarray(stats["s_degree_complex_rms"], np.float64), np.arange(1, 16, 2))
    scaled = response / scales[None, None, :]
    inc = np.asarray(data["u_inc"], np.complex64)
    n = len(inc)
    f = np.empty((n, 128, 140), np.float32)
    f[..., :128] = np.concatenate((scaled.real, scaled.imag), axis=-1).astype(np.float32)
    f[..., 128:130] = np.stack((inc.real, inc.imag), axis=-1)
    f[..., 130:133] = np.asarray(data["interface_points"], np.float32)[None] / 1.5
    centers = np.asarray(data["centers"], np.float32)
    f[..., 133:136] = ((centers - centers.mean(0, keepdims=True)) / 1.5)[:, None, :]
    f[..., 136:139] = np.asarray(data["incident_direction"], np.float32)[None, None, :]
    f[..., 139] = (1 / np.pi - 1 / 6) / (1 / 0.6 - 1 / 6) * 2 - 1
    return f.reshape(-1, 140)


def occupancy(data):
    # Extend, without modifying, the frozen old-family packing predicate.
    codes = np.asarray(data["shape_codes"], int)
    result = np.empty((len(codes), 128, 128), np.uint8)
    points = np.asarray(data["interface_points"], np.float64)
    directions = points / np.linalg.norm(points, axis=-1, keepdims=True)
    xyz = directions[:, None, :] * packing.RADII[None, :, None]
    for j, code in enumerate(codes):
        if code == 4:
            local = xyz @ np.asarray(data["rotations"][j], np.float64)
            size = np.asarray(data["shape_parameters"][j], np.float64)
            result[j] = (np.hypot(np.linalg.norm(local[..., :2], axis=-1) - size[0],
                                  local[..., 2]) <= size[1]).astype(np.uint8)
        else:
            subset = {k: data[k] for k in ("interface_points",)}
            for k in ("shape_codes", "shape_parameters", "rotations"):
                subset[k] = data[k][j:j + 1]
            result[j] = packing.make_old_occupancy(subset)[0]
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--training-cache", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    original, out = args.training_cache.resolve(), args.output.resolve()
    require(not out.exists(), "Output exists; retain prior attempts and choose a new output")
    source_root = ROOT / "datasets/canonical_response_v3/data/topology_ood"
    source, files, records = packing.old_manifest(source_root)
    require(len(files) == 500 and source["num_samples"] == 500, "Torus split is not complete")
    groups = Counter(r["ood_composition"] for r in records.values())
    require(groups == {"mixed": 250, "all": 250}, "Unexpected Torus subgroup counts")
    audit_path = ROOT / "outputs/v3_sealed_bem_audit_current_20260927/topology_ood.json"
    audit = packing.json_read(audit_path)
    require(audit["status"] == "passed" and audit["data_manifest_sha256"] == packing.sha(source_root / "manifest.json"),
            "The existing Torus audit is not source-bound and passing")
    stats = packing.json_read(original / "statistics.json")
    metadata = packing.json_read(original / "manifest.json")
    require(stats["fit_scenes"] == 20000 and metadata["train_scene_count"] == 20000
            and packing.sha(original / "statistics.json") == metadata["normalization_sha256"],
            "Wrong mixed20k training normalization")
    with np.load(original / "shared.npz", allow_pickle=False) as z:
        shared = {k: z[k] for k in z.files}

    # Independently check the on-the-fly S conversion and all 140 channels
    # against the exact ID cache used by the completed remote review.
    old_features = np.load(original / "id_test/features.npy", mmap_mode="r")
    old_offsets = np.load(original / "id_test/object_offsets.npy", mmap_mode="r")
    parity = []
    for i in (0, 47, 999):
        with np.load(ROOT / f"datasets/canonical_response_v3/data/id_test/sample_{i:05d}.npz", allow_pickle=False) as z:
            data = {k: z[k] for k in z.files}
        actual = response_features(data, stats)
        lo, hi = (old_offsets[i:i + 2] * 128).astype(int)
        reference = np.asarray(old_features[lo:hi])
        delta = np.linalg.norm(actual.astype(np.float64) - reference) / max(np.linalg.norm(reference), 1e-15)
        require(delta < 2e-6, f"S conversion/input packing parity failed on ID {i}: {delta}")
        parity.append({"id_scene": i, "relative_feature_difference": float(delta)})
    print(json.dumps({"event": "id_feature_parity_passed", "checks": parity}), flush=True)

    code_sha = next(v for k, v in packing.json_read(packing.OLD_PC_RUN)["source_sha256"].items()
                    if "v3_pointcloud.py" in k)
    clouds, memory, cloud_entries = {}, {}, []
    for rec in records.values():
        for obj in rec["objects"]:
            key, values = packing.old_pointcloud(obj, packing.OLD_PC, code_sha, memory)
            if key not in clouds:
                clouds[key] = len(clouds)
                cloud_entries.append({"library_id": key[0], "shape": key[1], "size": key[2], "cloud_sha256": values[2]})
    points = np.stack([memory[key][0] for key in clouds]).astype(np.float32)
    normals = np.stack([memory[key][1] for key in clouds]).astype(np.float32)
    require(np.isfinite(points).all() and np.allclose(np.linalg.norm(normals, axis=-1), 1, atol=2e-5),
            "Invalid cached geometry clouds/normals")

    start = time.perf_counter()
    out.mkdir(parents=True)
    folder = out / "topology_ood"
    folder.mkdir()
    shutil.copyfile(original / "statistics.json", out / "statistics.json")
    shutil.copyfile(original / "shared.npz", out / "shared.npz")
    shutil.copyfile(source_root / "manifest.json", out / "source_manifest.json")
    np.save(out / "geometry_points.npy", points, allow_pickle=False)
    np.save(out / "geometry_normals.npy", normals, allow_pickle=False)
    packing.json_write(out / "geometry_library.json", cloud_entries)
    counts = [len(records[i]["objects"]) for i in range(500)]
    require(set(counts) == {2, 3, 4}, "Unexpected object counts")
    nobjects, nn = sum(counts), sum(counts) * 128
    specs = {"features": ((nn, 140), np.float32), "labels": ((nn, 2), np.float32),
             "incident": ((nn, 2), np.float32), "q64": ((nn, 2), np.float32),
             "z_real": ((nn, 64), np.float32), "z_imag": ((nn, 64), np.float32),
             "occupancy": ((nobjects, 128, 128), np.uint8), "geometry_index": ((nobjects,), np.int32),
             "rotations": ((nobjects, 3, 3), np.float32), "nearest_indices": ((nobjects, 128, 16), np.uint16),
             "object_offsets": ((501,), np.int64), "source_code": ((500,), np.uint8),
             "source_sample_index": ((500,), np.int32)}
    arrays = {name: np.lib.format.open_memmap(folder / f"{name}.npy", mode="w+", dtype=dtype, shape=shape)
              for name, (shape, dtype) in specs.items()}
    offsets = np.concatenate(([0], np.cumsum(counts))).astype(np.int64)
    arrays["object_offsets"][:] = offsets
    max_q_difference = 0.0
    for i, path in enumerate(files):
        rec = records[i]
        require(packing.sha(path) == rec["sha256"], f"Source sample checksum differs: {path}")
        with np.load(path, allow_pickle=False) as z:
            d = {k: z[k] for k in z.files}
        n = counts[i]
        require(d["Z"].shape == (n, 128, 64) and d["Z"].dtype == np.complex64, "Invalid stored Z")
        require(np.allclose(d["interface_points"], shared["interface_points"], atol=2e-6)
                and np.allclose(d["phi_pinv"], shared["phi_pinv"], rtol=2e-5, atol=2e-6), "Canonical basis mismatch")
        require(np.isclose(float(d["k"]), 2) and np.allclose(d["centers"], rec["centers"], atol=2e-6),
                "Physics/layout metadata mismatch")
        require(np.allclose(d["incident_direction"], rec["incident_direction"], atol=2e-6), "Direction mismatch")
        torus_count = 0
        for j, obj in enumerate(rec["objects"]):
            require(int(d["shape_codes"][j]) == ("sphere", "ellipsoid", "box", "peanut", "torus").index(obj["shape"]),
                    "Shape code mismatch")
            require(np.allclose(d["shape_parameters"][j, :len(obj["size"])], obj["size"], atol=2e-6)
                    and np.allclose(d["rotations"][j], obj["rotation"], atol=2e-6), "Actual geometry mismatch")
            torus_count += obj["shape"] == "torus"
        require(torus_count == (1 if rec["ood_composition"] == "mixed" else n), "Composition mismatch")
        a, b = map(int, offsets[i:i + 2])
        lo, hi = a * 128, b * 128
        arrays["features"][lo:hi] = response_features(d, stats)
        for name, value in (("incident", d["u_inc"]), ("q64", d["q64_gt"])):
            arrays[name][lo:hi] = np.stack((value.real, value.imag), -1).reshape(-1, 2)
        scatter = np.asarray(d["u_total_gt"] - d["u_inc"], np.complex64)
        arrays["labels"][lo:hi] = np.stack((scatter.real, scatter.imag), -1).reshape(-1, 2) / stats["scatter_complex_rms"]
        arrays["z_real"][lo:hi] = d["Z"].real.reshape(-1, 64)
        arrays["z_imag"][lo:hi] = d["Z"].imag.reshape(-1, 64)
        arrays["occupancy"][a:b] = occupancy(d)
        arrays["rotations"][a:b] = d["rotations"]
        arrays["source_code"][i] = 0
        arrays["source_sample_index"][i] = i
        for j, obj in enumerate(rec["objects"]):
            key = (str(obj["library_id"]), str(obj["shape"]), tuple(float(x) for x in obj["size"]))
            gid = clouds[key]
            arrays["geometry_index"][a + j] = gid
            cloud = points[gid] @ np.asarray(d["rotations"][j], np.float32).T
            arrays["nearest_indices"][a + j] = cKDTree(cloud).query(shared["interface_points"], k=16)[1].astype(np.uint16)
        q = np.einsum("onm,om->on", d["Z"].astype(np.complex128),
                      np.einsum("mn,on->om", d["phi_pinv"].astype(np.complex128), d["u_total_gt"]))
        qdiff = np.linalg.norm(q - d["q64_gt"]) / max(np.linalg.norm(d["q64_gt"]), 1e-15)
        require(qdiff < 1e-4, "Derived q64 identity mismatch")
        max_q_difference = max(max_q_difference, float(qdiff))
        if i % 50 == 0:
            print(json.dumps({"event": "torus_pack_progress", "done": i + 1, "total": 500}), flush=True)
    for a in arrays.values():
        a.flush()
        require(np.isfinite(a).all(), "Nonfinite packed array")
    packing.json_write(folder / "records.json", [records[i] for i in range(500)])
    file_hashes = {str(p.relative_to(out)).replace("\\", "/"): packing.sha(p)
                   for p in sorted(out.rglob("*")) if p.is_file()}
    packing.json_write(out / "manifest.json", {
        "schema": "mixed20k-torus-review-sidecar-v1", "status": "complete", "scenes": 500,
        "groups": dict(groups), "objects": nobjects, "object_count_distribution": dict(Counter(counts)),
        "training_cache_manifest_sha256": packing.sha(original / "manifest.json"),
        "training_statistics_sha256": packing.sha(original / "statistics.json"),
        "source_manifest_sha256": packing.sha(source_root / "manifest.json"),
        "existing_bem_audit_sha256": packing.sha(audit_path), "independent_bem_audit_scenes": len(audit["selected_indices"]),
        "s_conversion": "existing structural_response.scattering_response from saved complex64 Z only",
        "id_feature_parity": parity, "max_q64_identity_difference": max_q_difference,
        "geometry_cloud_count": len(points), "source_code_convention": "0, canonical-v3; M3 epoch=0 edge seed unchanged",
        "files_sha256": file_hashes, "source_sha256": {
            name: packing.sha(ROOT / name) for name in ("scripts/prepare_torus_20k_review.py",
                "scripts/prepare_v6s11_20k_all_methods.py", "src/cspgnet/data_generation/structural_response.py",
                "src/cspgnet/data_generation/reduced_z64_solver.py", "src/cspgnet/data_generation/tmatrix_neumann_3d.py")},
        "elapsed_seconds": time.perf_counter() - start, "interpretation": "Previously viewed Torus family cross-run review; training labels partly exploratory",
        "v6s11_scientific_release": False})
    print(json.dumps({"event": "torus_cache_complete", "output": str(out), "scenes": 500,
                      "seconds": time.perf_counter() - start}), flush=True)


if __name__ == "__main__":
    main()
