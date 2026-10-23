"""Build an immutable, source-bound 20k mixed-source training cache.

The cache is intentionally prepared before remote training: it avoids repeated
NPZ decompression in epochs and freezes all feature scales using train only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree


ROOT = Path(__file__).resolve().parents[1]
OLD_DATA = ROOT / "datasets/canonical_response_v3"
OLD_TRAIN = OLD_DATA / "full/data/train"
OLD_VAL = OLD_DATA / "full/data/val"
OLD_ID = OLD_DATA / "data/id_test"
OLD_OOD = OLD_DATA / "data/shape_ood"
OLD_S = ROOT / "outputs/v3_feature_trials_20260927/cache"
OLD_PC = ROOT / "outputs/v3_main_single_run_20260927/pointcloud_cache"
OLD_PC_RUN = ROOT / "outputs/v3_main_single_run_20260927/models/geo-strong-adapted_seed0_100ep/run_manifest.json"
NEW_PACK = ROOT / "datasets/subset11_quick_trainable_v1"
NEW_GEOMETRY = ROOT / "datasets/subset11_quick_geometry_v1"
DEFAULT_OUT = ROOT / "outputs/v6s11_20k_all_methods_20261009/prepared"
NODES = 128
MODES = 64
RADIUS = 1.5
RADII = 0.9 * (np.arange(128, dtype=np.float64) + 0.5) / 128.0
SOURCE_NAMES = ("canonical-v3", "v6s11")


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_read(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def json_write(path: Path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_name(path.name + ".pending")
    pending.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(pending, path)


def files_for(root: Path):
    return sorted(Path(root).glob("sample_*.npz"), key=lambda p: int(p.stem.split("_")[-1]))


def old_manifest(split_root: Path):
    manifest = json_read(split_root / "manifest.json")
    files = files_for(split_root)
    if manifest.get("status") != "complete" or len(files) != len(manifest.get("samples", [])):
        raise RuntimeError(f"Incomplete canonical-v3 split: {split_root}")
    records = {int(item["index"]): item for item in manifest["samples"]}
    if set(records) != set(range(len(files))):
        raise RuntimeError(f"Non-contiguous scene IDs: {split_root}")
    return manifest, files, records


def verify_new_inputs():
    pack = json_read(NEW_PACK / "trainable_manifest.json")
    if pack.get("status") != "complete" or pack.get("scientific_release") is not False:
        raise RuntimeError("Expected the complete exploratory v6s11 pack")
    for relative, item in pack["files"].items():
        path = NEW_PACK / relative
        if path.stat().st_size != item["bytes"] or sha(path) != item["sha256"]:
            raise RuntimeError(f"v6s11 training pack checksum failed: {relative}")
    gm = json_read(NEW_GEOMETRY / "geometry_manifest.json")
    receipt = json_read(NEW_GEOMETRY / "training_interface_validation.json")
    if gm.get("status") != "complete" or gm.get("scientific_release") is not False:
        raise RuntimeError("Expected the complete exploratory v6s11 geometry sidecar")
    if gm.get("base_manifest_sha256") != sha(NEW_PACK / "trainable_manifest.json"):
        raise RuntimeError("v6s11 geometry sidecar belongs to a different data pack")
    if receipt.get("status") != "passed" or receipt.get("geometry_manifest_sha256") != sha(NEW_GEOMETRY / "geometry_manifest.json"):
        raise RuntimeError("v6s11 geometry interface receipt is stale or failed")
    for relative, item in gm["files"].items():
        path = NEW_GEOMETRY / relative
        if path.stat().st_size != item["bytes"] or sha(path) != item["sha256"]:
            raise RuntimeError(f"v6s11 geometry checksum failed: {relative}")
    return pack, gm, receipt


def old_pointcloud(record: dict, cache: Path, code_sha: str, memory: dict):
    key = (str(record["library_id"]), str(record["shape"]), tuple(float(x) for x in record["size"]))
    if key in memory:
        return key, memory[key]
    payload = json.dumps([
        "area-mesh-h0.15-v1", code_sha, key[0], key[1], key[2],
        record.get("smoothing_beta"), int(record.get("meridian_points", 65)),
    ], separators=(",", ":"))
    fingerprint = hashlib.sha256(payload.encode()).hexdigest()
    path = cache / f"{fingerprint}.npz"
    if not path.is_file():
        raise FileNotFoundError(f"Required previously audited surface point cloud missing: {path.name}")
    with np.load(path, allow_pickle=False) as opened:
        if str(opened["fingerprint"]) != fingerprint:
            raise RuntimeError(f"Point cloud fingerprint mismatch: {path}")
        points = np.asarray(opened["points"], dtype=np.float32)
        normals = np.asarray(opened["normals"], dtype=np.float32)
    if points.shape != (2048, 3) or normals.shape != points.shape:
        raise ValueError(f"Invalid old surface point cloud: {path}")
    memory[key] = (points, normals, sha(path))
    return key, memory[key]


def object_cloud_index(record, libraries, pc_memory, pc_code_sha):
    key, value = old_pointcloud(record, OLD_PC, pc_code_sha, pc_memory)
    if key not in libraries["old_key_to_index"]:
        idx = len(libraries["points"])
        libraries["old_key_to_index"][key] = idx
        libraries["points"].append(value[0])
        libraries["normals"].append(value[1])
        libraries["ids"].append(f"old:{key[0]}:{key[1]}:{','.join(format(x,'.17g') for x in key[2])}")
        libraries["hashes"][libraries["ids"][-1]] = value[2]
    return libraries["old_key_to_index"][key]


def make_old_occupancy(data: dict) -> np.ndarray:
    points = np.asarray(data["interface_points"], dtype=np.float64)
    directions = points / np.linalg.norm(points, axis=-1, keepdims=True)
    xyz = directions[:, None, :] * RADII[None, :, None]
    codes = np.asarray(data["shape_codes"], dtype=np.int64)
    sizes = np.asarray(data["shape_parameters"], dtype=np.float64)
    rotations = np.asarray(data["rotations"], dtype=np.float64)
    result = np.empty((len(codes), NODES, len(RADII)), dtype=np.uint8)
    for j, code in enumerate(codes):
        local = xyz @ rotations[j]
        size = sizes[j]
        if code == 0:
            inside = np.linalg.norm(local, axis=-1) <= size[0]
        elif code == 1:
            inside = np.sum((local / size[None, None, :3]) ** 2, axis=-1) <= 1.0
        elif code == 2:
            inside = np.all(np.abs(local) <= size[None, None, :3], axis=-1)
        elif code == 3:
            d1 = np.hypot(np.linalg.norm(local[..., :2], axis=-1), local[..., 2] + size[2] / 2) - size[0]
            d2 = np.hypot(np.linalg.norm(local[..., :2], axis=-1), local[..., 2] - size[2] / 2) - size[1]
            inside = (-np.logaddexp(-30.0 * d1, -30.0 * d2) / 30.0) <= 0.0
        else:
            raise ValueError(f"Unsupported old geometry code {code}")
        result[j] = inside.astype(np.uint8)
    return result


def accum_stats(state, s, total, incident, q, occupancy):
    s = np.asarray(s, dtype=np.complex128)
    for ell in range(8):
        block = s[..., ell * ell:(ell + 1) ** 2]
        state["s_power"][ell] += np.square(np.abs(block)).sum(dtype=np.float64)
        state["s_count"][ell] += block.size
    scatter = np.asarray(total, dtype=np.complex128) - np.asarray(incident, dtype=np.complex128)
    state["scatter_power"] += np.square(np.abs(scatter)).sum(dtype=np.float64)
    state["field_count"] += scatter.size
    state["q_energies"].append(float(np.square(np.abs(np.asarray(q, dtype=np.complex128))).sum(dtype=np.float64)))
    occ = np.asarray(occupancy, dtype=np.float64)
    state["occ_sum"] += occ.sum(axis=(0, 1), dtype=np.float64)
    state["occ_sq"] += np.square(occ).sum(axis=(0, 1), dtype=np.float64)
    state["occ_count"] += occ.shape[0] * occ.shape[1]


def iter_new(root: Path, split: str, geom_root: Path):
    arr = {name: np.load(root / split / f"{name}.npy", mmap_mode="r", allow_pickle=False)
           for name in ("S", "Z", "u_total_gt", "u_inc", "q64_gt", "centers", "rotations", "incident_direction", "offsets", "geometry_ids")}
    ga = {name: np.load(geom_root / split / f"{name}.npy", mmap_mode="r", allow_pickle=False)
          for name in ("occupancy_bits", "nearest_indices", "geometry_index")}
    return arr, ga


def new_occ(bits):
    return np.unpackbits(np.asarray(bits, dtype=np.uint8), axis=-1, bitorder="little").astype(np.uint8, copy=False)


def prepare(args):
    out = args.output.resolve()
    if (out / "manifest.json").exists():
        existing = json_read(out / "manifest.json")
        if existing.get("status") == "complete":
            print(json.dumps({"status": "already_complete", "output": str(out), "scenes": existing["splits"]}))
            return
        raise FileExistsError(f"Partial output exists; preserve and inspect it, use another --output: {out}")
    out.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    old_train_m, old_train_files, old_train_records = old_manifest(OLD_TRAIN)
    old_val_m, old_val_files, old_val_records = old_manifest(OLD_VAL)
    old_id_m, old_id_files, old_id_records = old_manifest(OLD_ID)
    old_ood_m, old_ood_files, old_ood_records = old_manifest(OLD_OOD)
    pack, geom_manifest, receipt = verify_new_inputs()
    st = json_read(OLD_S / "train/cache_manifest.json")
    sv = json_read(OLD_S / "val/cache_manifest.json")
    if st["source_manifest_sha256"] != sha(OLD_TRAIN / "manifest.json") or sv["source_manifest_sha256"] != sha(OLD_VAL / "manifest.json"):
        raise RuntimeError("Old S caches are not bound to the selected canonical-v3 manifests")
    if st.get("files") != 10000 or sv.get("files") != 1000:
        raise RuntimeError("Unexpected old S cache split counts")
    s_train = np.load(OLD_S / "train/S.npy", mmap_mode="r", allow_pickle=False)
    s_val = np.load(OLD_S / "val/S.npy", mmap_mode="r", allow_pickle=False)
    pc_run = json_read(OLD_PC_RUN)
    pc_code_sha = next(v for k, v in pc_run["source_sha256"].items() if "v3_pointcloud.py" in k)
    pc_memory = {}
    libraries = {"old_key_to_index": {}, "new_id_to_index": {}, "points": [], "normals": [], "ids": [], "hashes": {}}
    old_manifests = ((OLD_TRAIN, old_train_records), (OLD_VAL, old_val_records),
                     (OLD_ID, old_id_records), (OLD_OOD, old_ood_records))
    old_geometry_index = {}
    for _, records in old_manifests:
        for rec in records.values():
            for obj in rec["objects"]:
                key, _ = old_pointcloud(obj, OLD_PC, pc_code_sha, pc_memory)
                if key not in libraries["old_key_to_index"]:
                    idx = len(libraries["points"])
                    libraries["old_key_to_index"][key] = idx
                    val = pc_memory[key]
                    libraries["points"].append(val[0]); libraries["normals"].append(val[1])
                    token = f"old:{key[0]}:{key[1]}:{','.join(format(x,'.17g') for x in key[2])}"
                    libraries["ids"].append(token); libraries["hashes"][token] = val[2]
                old_geometry_index[(obj["library_id"], obj["shape"], tuple(float(x) for x in obj["size"]))] = libraries["old_key_to_index"][key]
    with np.load(NEW_GEOMETRY / "pointclouds.npz", allow_pickle=False) as pc:
        new_points, new_normals = pc["points"], pc["normals"]
        new_ids = [str(x) for x in pc["geometry_ids"]]
    new_library = json_read(NEW_PACK / "geometry_library.json")
    lib_entries = {str(entry["id"]): entry for entry in new_library["entries"]}
    for idx, gid in enumerate(new_ids):
        if gid not in lib_entries:
            raise RuntimeError(f"Point cloud geometry ID missing from library: {gid}")
        target = len(libraries["points"])
        libraries["new_id_to_index"][gid] = target
        libraries["points"].append(np.asarray(new_points[idx], dtype=np.float32))
        libraries["normals"].append(np.asarray(new_normals[idx], dtype=np.float32))
        token = f"v6s11:{gid}"
        libraries["ids"].append(token)
        libraries["hashes"][token] = sha(NEW_GEOMETRY / "pointclouds.npz")
    cloud_points = np.stack(libraries["points"]).astype(np.float32)
    cloud_normals = np.stack(libraries["normals"]).astype(np.float32)
    np.save(out / "geometry_points.npy", cloud_points, allow_pickle=False)
    np.save(out / "geometry_normals.npy", cloud_normals, allow_pickle=False)
    json_write(out / "geometry_library.json", {"ids": libraries["ids"], "source_file_sha256": libraries["hashes"],
                                                  "shape": list(cloud_points.shape), "points_per_shape": 2048})
    del cloud_points, cloud_normals

    # Fit every scale from the 20,000 training scenes only.
    state = {"s_power": np.zeros(8, np.float64), "s_count": np.zeros(8, np.int64),
             "scatter_power": 0.0, "field_count": 0, "q_energies": [],
             "occ_sum": np.zeros(128, np.float64), "occ_sq": np.zeros(128, np.float64), "occ_count": 0}
    old_offsets = np.asarray(st["offsets"], dtype=np.int64)
    for i, path in enumerate(old_train_files):
        with np.load(path, allow_pickle=False) as opened:
            data = {k: opened[k] for k in opened.files}
        lo, hi = old_offsets[i:i+2]
        occ = make_old_occupancy(data)
        accum_stats(state, s_train[lo:hi], data["u_total_gt"], data["u_inc"], data["q64_gt"], occ)
        if (i + 1) % 2000 == 0: print(json.dumps({"phase":"fit_stats_old","done":i+1,"total":10000}), flush=True)
    na, nga = iter_new(NEW_PACK, "train", NEW_GEOMETRY)
    for i in range(10000):
        lo, hi = map(int, na["offsets"][i:i+2])
        occ = new_occ(nga["occupancy_bits"][lo:hi])
        accum_stats(state, na["S"][lo:hi], na["u_total_gt"][lo:hi], na["u_inc"][lo:hi], na["q64_gt"][lo:hi], occ)
        if (i + 1) % 2000 == 0: print(json.dumps({"phase":"fit_stats_v6s11","done":i+1,"total":10000}), flush=True)
    s_scale = np.sqrt(np.maximum(state["s_power"] / state["s_count"], 1e-24))
    scatter_scale = float(np.sqrt(state["scatter_power"] / state["field_count"]))
    occ_mean = state["occ_sum"] / state["occ_count"]
    occ_var = np.maximum(state["occ_sq"] / state["occ_count"] - occ_mean**2, 1e-4)
    stats = {"schema":"v6s11-mixed-20k-train-only-statistics-v1", "fit_split":"combined_train",
             "fit_scenes":20000, "source_scenes":{"canonical-v3":10000,"v6s11":10000},
             "s_degree_complex_rms":s_scale.tolist(), "scatter_complex_rms":scatter_scale,
             "q_scene_energy_median":float(np.median(state["q_energies"])),
             "q_loss_stabilizer":float(1e-6*np.median(state["q_energies"])),
             "occupancy_mean":occ_mean.tolist(), "occupancy_std":np.sqrt(occ_var).tolist(),
             "occupancy_variance_floor":1e-4, "canonical_radius":RADIUS,
             "no_complex_mean_subtraction":True}
    json_write(out / "statistics.json", stats)
    with np.load(NEW_PACK / "shared.npz", allow_pickle=False) as shared:
        interface_shared = shared["interface_points"].astype(np.float32)
        phi_shared = shared["phi_pinv"].astype(np.complex64)
    np.savez(out / "shared.npz", interface_points=interface_shared, phi_pinv=phi_shared, k=np.float32(2.0))
    interface = np.load(out / "shared.npz")["interface_points"]
    phi = np.load(out / "shared.npz")["phi_pinv"]
    if interface.shape != (128,3) or phi.shape != (64,128): raise RuntimeError("Shared canonical basis dimensions differ")

    def allocate_split(name, nscenes, nobjects):
        folder = out / name; folder.mkdir(parents=True, exist_ok=False)
        nn = nobjects * NODES
        specs = {"features":((nn,140),np.float32), "labels":((nn,2),np.float32),
                 "incident":((nn,2),np.float32), "q64":((nn,2),np.float32),
                 "z_real":((nn,64),np.float32), "z_imag":((nn,64),np.float32),
                 "occupancy":((nobjects,128,128),np.uint8), "geometry_index":((nobjects,),np.int32),
                 "rotations":((nobjects,3,3),np.float32), "nearest_indices":((nobjects,128,16),np.uint16),
                 "object_offsets":((nscenes+1,),np.int64), "source_code":((nscenes,),np.uint8),
                 "source_sample_index":((nscenes,),np.int32)}
        return folder, {key:np.lib.format.open_memmap(folder/f"{key}.npy",mode="w+",dtype=dtype,shape=shape)
                        for key,(shape,dtype) in specs.items()}

    def add_old_split(name, split_root, files, records, s_array, source_code):
        manifest = json_read(split_root / "manifest.json")
        nscenes = len(files); nobjects = sum(len(records[i]["objects"]) for i in range(nscenes))
        folder, a = allocate_split(name,nscenes,nobjects)
        offsets = np.zeros(nscenes+1,np.int64); object_cursor=0
        s_offsets = np.asarray((st if split_root==OLD_TRAIN else sv)["offsets"],np.int64) if split_root in (OLD_TRAIN,OLD_VAL) else None
        test_cache = None
        if split_root in (OLD_ID, OLD_OOD):
            cache_name = "id_test" if split_root == OLD_ID else "shape_ood"
            croot = ROOT / "outputs/v3_object_transolver_40ep_20260930/tensor_cache" / cache_name
            cm = json_read(croot / "manifest.json")
            if cm.get("status") != "complete" or cm.get("scenes") != nscenes:
                raise RuntimeError(f"Missing completed S-derived development cache for {name}")
            feat_mem = np.load(croot / "features.npy", mmap_mode="r", allow_pickle=False)
            test_offsets = np.asarray(cm["offsets"], np.int64)
            old_scale64 = np.repeat(np.asarray(st["degree_complex_rms"], np.float64), np.arange(1, 16, 2))
            if feat_mem.ndim != 2 or feat_mem.shape[1] != 140 or test_offsets.shape != (nscenes + 1,):
                raise RuntimeError(f"Malformed S-derived review cache for {name}")
            if test_offsets[-1] != feat_mem.shape[0]:
                raise RuntimeError(f"Review-cache scene offsets do not align for {name}")
            test_cache = (feat_mem, test_offsets, np.concatenate((old_scale64, old_scale64)))
        for i,path in enumerate(files):
            with np.load(path,allow_pickle=False) as opened: data={k:opened[k] for k in opened.files}
            rec=records[i]; n=len(data["Z"]); olo=object_cursor; ohi=olo+n; offsets[i+1]=ohi
            if i==0 and not np.allclose(data["phi_pinv"],phi,rtol=2e-5,atol=2e-6):
                raise RuntimeError(f"Old/new projection bases differ in {name}")
            if split_root in (OLD_TRAIN,OLD_VAL): slo,shi=s_offsets[i:i+2]; response=s_array[slo:shi]
            else:
                # Existing v3 tensor cache already stores the approved S(Z) derived from this scene's Z.
                feat_mem, test_offsets, old_scale128 = test_cache
                lo,hi=test_offsets[i:i+2]  # cached offsets count nodes, not objects
                if hi-lo != n*NODES:
                    raise RuntimeError(f"Review-cache object count differs at {name}/{i}")
                old_response=(feat_mem[lo:hi,:128].reshape(n,128,128).astype(np.float64)*old_scale128[None,None,:])
                response=(old_response[...,:64]+1j*old_response[...,64:]).astype(np.complex64)
            response=np.asarray(response,dtype=np.complex64)
            response_scale=np.repeat(s_scale,np.arange(1,16,2))
            first=np.concatenate(((response/response_scale).real,(response/response_scale).imag),axis=-1).astype(np.float32)
            inc=np.asarray(data["u_inc"],np.complex64)
            inc_pair=np.stack((inc.real,inc.imag),axis=-1).astype(np.float32)
            local=np.asarray(data["interface_points"],np.float32)/RADIUS
            centers=np.asarray(data["centers"],np.float32); relative=(centers-centers.mean(0,keepdims=True))/RADIUS
            direction=np.asarray(data["incident_direction"],np.float32)
            wavelength=(1.0/(2*np.pi/2.0)-1/6)/(1/0.6-1/6)*2-1
            f=np.empty((n,128,140),np.float32); f[...,:128]=first; f[...,128:130]=inc_pair
            f[...,130:133]=local[None]; f[...,133:136]=relative[:,None,:]; f[...,136:139]=direction[None,None,:]; f[...,139]=wavelength
            scatter=np.asarray(data["u_total_gt"]-data["u_inc"],np.complex64)
            labels=np.stack((scatter.real,scatter.imag),axis=-1).astype(np.float32)/scatter_scale
            q=np.asarray(data["q64_gt"],np.complex64); qp=np.stack((q.real,q.imag),axis=-1).astype(np.float32)
            z=np.asarray(data["Z"],np.complex64)
            node_lo,node_hi=olo*128,ohi*128
            a["features"][node_lo:node_hi]=f.reshape(-1,140); a["labels"][node_lo:node_hi]=labels.reshape(-1,2)
            a["incident"][node_lo:node_hi]=inc_pair.reshape(-1,2); a["q64"][node_lo:node_hi]=qp.reshape(-1,2)
            a["z_real"][node_lo:node_hi]=z.real.reshape(-1,64)
            a["z_imag"][node_lo:node_hi]=z.imag.reshape(-1,64)
            a["occupancy"][olo:ohi]=make_old_occupancy(data)
            rotations=np.asarray(data["rotations"],np.float32)
            for j,obj in enumerate(rec["objects"]):
                gkey=(obj["library_id"],obj["shape"],tuple(float(x) for x in obj["size"]))
                gid=old_geometry_index[gkey]; a["geometry_index"][olo+j]=gid; a["rotations"][olo+j]=rotations[j]
                cloud=np.asarray(libraries["points"][gid],np.float32)@rotations[j].T
                a["nearest_indices"][olo+j]=cKDTree(cloud).query(interface,k=16)[1].astype(np.uint16)
            a["source_code"][i]=source_code; a["source_sample_index"][i]=i
            object_cursor=ohi
            if (i+1)%1000==0: print(json.dumps({"phase":"pack","split":name,"done":i+1,"total":nscenes}),flush=True)
        a["object_offsets"][:]=offsets
        for x in a.values(): x.flush()
        json_write(folder/"records.json",[{"source":SOURCE_NAMES[source_code],"sample_index":i,
                                           "objects":records[i]["objects"],
                                           "ood_group":records[i].get("ood_group",records[i].get("group"))}
                                          for i in range(nscenes)])
        return {"scenes":nscenes,"objects":nobjects,"source":SOURCE_NAMES[source_code],
                "source_manifest_sha256":sha(split_root/"manifest.json")}

    def add_new_split(name, split, source_code):
        arr,ga=iter_new(NEW_PACK,split,NEW_GEOMETRY); nscenes=len(arr["offsets"])-1; nobjects=int(arr["offsets"][-1])
        folder,a=allocate_split(name,nscenes,nobjects); offsets=np.asarray(arr["offsets"],np.int64)
        a["object_offsets"][:]=offsets
        # The pack's per-object geometry_ids are not an index table.  The
        # sidecar indices address pointclouds.npz's canonical geometry order.
        with np.load(NEW_GEOMETRY/"pointclouds.npz",allow_pickle=False) as pc:
            geometry_library_ids=[str(x) for x in pc["geometry_ids"]]
        id_map={gid:libraries["new_id_to_index"][gid] for gid in geometry_library_ids}
        for i in range(nscenes):
            lo,hi=map(int,offsets[i:i+2]); n=hi-lo
            if i==0:
                with np.load(NEW_PACK/"shared.npz",allow_pickle=False) as shared:
                    if not np.allclose(shared["phi_pinv"],phi,rtol=2e-5,atol=2e-6): raise RuntimeError("v6 projection basis mismatch")
            response=np.asarray(arr["S"][lo:hi],np.complex64); scaled=response/np.repeat(s_scale,np.arange(1,16,2))[None,None,:]
            first=np.concatenate((scaled.real,scaled.imag),axis=-1).astype(np.float32)
            inc=np.asarray(arr["u_inc"][lo:hi],np.complex64); ip=np.stack((inc.real,inc.imag),axis=-1).astype(np.float32)
            centers=np.asarray(arr["centers"][lo:hi],np.float32); rel=(centers-centers.mean(0,keepdims=True))/RADIUS
            direction=np.asarray(arr["incident_direction"][i],np.float32)
            f=np.empty((n,128,140),np.float32); f[...,:128]=first; f[...,128:130]=ip
            f[...,130:133]=interface[None]/RADIUS; f[...,133:136]=rel[:,None,:]; f[...,136:139]=direction[None,None,:]
            f[...,139]=float((1/(np.pi)-1/6)/(1/0.6-1/6)*2-1)
            scatter=np.asarray(arr["u_total_gt"][lo:hi]-inc,np.complex64)
            labels=np.stack((scatter.real,scatter.imag),axis=-1).astype(np.float32)/scatter_scale
            q=np.asarray(arr["q64_gt"][lo:hi],np.complex64); qp=np.stack((q.real,q.imag),axis=-1).astype(np.float32)
            z=np.asarray(arr["Z"][lo:hi],np.complex64); nl,nh=lo*128,hi*128
            a["features"][nl:nh]=f.reshape(-1,140); a["labels"][nl:nh]=labels.reshape(-1,2)
            a["incident"][nl:nh]=ip.reshape(-1,2); a["q64"][nl:nh]=qp.reshape(-1,2)
            a["z_real"][nl:nh]=z.real.reshape(-1,64); a["z_imag"][nl:nh]=z.imag.reshape(-1,64)
            a["occupancy"][lo:hi]=new_occ(ga["occupancy_bits"][lo:hi])
            ids=np.asarray(arr["geometry_ids"][lo:hi]).astype(str)
            geom_index=np.asarray(ga["geometry_index"][lo:hi],np.int64)
            for j in range(n):
                if geometry_library_ids[int(geom_index[j])] != ids[j]: raise RuntimeError("v6 geometry sidecar object mapping mismatch")
                a["geometry_index"][lo+j]=id_map[ids[j]]
                a["rotations"][lo+j]=arr["rotations"][lo+j]
                a["nearest_indices"][lo+j]=ga["nearest_indices"][lo+j]
            a["source_code"][i]=source_code; a["source_sample_index"][i]=i
            if (i+1)%1000==0: print(json.dumps({"phase":"pack","split":name,"done":i+1,"total":nscenes}),flush=True)
        for x in a.values(): x.flush()
        recs=json_read(NEW_PACK/split/"records.json")
        json_write(folder/"records.json",[{"source":SOURCE_NAMES[source_code],"sample_index":i,**r} for i,r in enumerate(recs)])
        return {"scenes":nscenes,"objects":nobjects,"source":SOURCE_NAMES[source_code],
                "source_manifest_sha256":pack["source_binding"]["source_train_manifest_sha256" if split=="train" else "source_val_manifest_sha256"]}

    split_meta={}
    split_meta["train_v3"]=add_old_split("train_v3",OLD_TRAIN,old_train_files,old_train_records,s_train,0)
    split_meta["train_v6s11"]=add_new_split("train_v6s11","train",1)
    split_meta["val_v3"]=add_old_split("val_v3",OLD_VAL,old_val_files,old_val_records,s_val,0)
    split_meta["val_v6s11"]=add_new_split("val_v6s11","val",1)
    split_meta["id_test"]=add_old_split("id_test",OLD_ID,old_id_files,old_id_records,None,0)
    split_meta["peanut_ood"]=add_old_split("peanut_ood",OLD_OOD,old_ood_files,old_ood_records,None,0)
    source_files={
        "canonical_train_manifest":sha(OLD_TRAIN/"manifest.json"), "canonical_val_manifest":sha(OLD_VAL/"manifest.json"),
        "canonical_id_manifest":sha(OLD_ID/"manifest.json"), "canonical_peanut_manifest":sha(OLD_OOD/"manifest.json"),
        "canonical_train_S_manifest":sha(OLD_S/"train/cache_manifest.json"), "canonical_val_S_manifest":sha(OLD_S/"val/cache_manifest.json"),
        "v6s11_pack_manifest":sha(NEW_PACK/"trainable_manifest.json"), "v6s11_geometry_manifest":sha(NEW_GEOMETRY/"geometry_manifest.json"),
        "v6s11_geometry_receipt":sha(NEW_GEOMETRY/"training_interface_validation.json")}
    output_files={}
    for path in sorted(out.rglob("*")):
        if path.is_file() and path.name not in ("manifest.json",): output_files[str(path.relative_to(out)).replace("\\","/")]=sha(path)
    final={"schema":"v6s11-20k-mixed-all-methods-cache-v1","status":"complete","scientific_status":"exploratory; v6s11 independent BEM validation incomplete",
           "root":str(out),"splits":split_meta,"train_scene_count":20000,"train_object_count":split_meta["train_v3"]["objects"]+split_meta["train_v6s11"]["objects"],
           "validation":{"v3":1000,"v6s11":1000,"selection":"equal-weight mean of source-specific per-scene scattered relative L2 means"},
           "cross_run_review":{"id_test":1000,"peanut_ood":500,"previously inspected":True},
           "normalization":stats,"normalization_sha256":sha(out/"statistics.json"),"source_sha256":source_files,
           "geometry_sidecar_receipt_status":receipt["status"],"geometry_sidecar_scientific_release":False,
           "geometry_pointcloud_count":len(libraries["ids"]),"geometry_pointclouds_sha256":{"points":sha(out/"geometry_points.npy"),"normals":sha(out/"geometry_normals.npy")},
           "files_sha256":output_files,"elapsed_seconds":time.perf_counter()-started}
    json_write(out/"manifest.json",final)
    print(json.dumps({"status":"complete","output":str(out),"splits":split_meta,"elapsed_seconds":final["elapsed_seconds"]},ensure_ascii=False),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output",type=Path,default=DEFAULT_OUT)
    args=p.parse_args(); prepare(args)


if __name__=="__main__": main()
