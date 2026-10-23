"""Inference-only Torus review of the seven completed mixed-20k models."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time
import types

ROOT = Path(__file__).resolve().parents[1]
DEPS = ROOT / "work/torus_eval_deps_20261010"
sys.path[:0] = [str(DEPS), str(ROOT / "scripts")]
TORCH_PACKAGES = Path(r"D:\paper_等效散射方法\local_runtimes\torch280_torus_20261010\packages")
if (TORCH_PACKAGES / "torch/__init__.py").exists():
    sys.path.insert(0, str(TORCH_PACKAGES))
os.environ.setdefault("LOCAL_CACHE", str(DEPS / "cache"))

# Windows has no usable native DGL wheel for this Torch/Python pair. The
# qualified PIBNet path executes only edge-index/scatter. These annotation
# namespaces permit unchanged PhysicsNeMo imports; any attempted DGL graph
# operation fails, rather than silently using another numerical backend.
dgl = types.ModuleType("dgl")
dgl.__path__ = []
dgl.DGLGraph = type("DGLGraph", (), {})
dgl.DGLHeteroGraph = dgl.DGLGraph
dgl.function = types.ModuleType("dgl.function")
sys.modules["dgl"] = dgl
sys.modules["dgl.function"] = dgl.function

import numpy as np
import torch
import train_v6s11_20k_all_methods as training


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def summarize(values):
    values = np.asarray(values, np.float64)
    require(values.size > 0 and np.isfinite(values).all(), "Invalid scene metric vector")
    return {"mean": float(values.mean()), "median": float(np.median(values)),
            "p95": float(np.quantile(values, .95))}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sidecar", type=Path, required=True)
    p.add_argument("--training-cache", type=Path, required=True)
    p.add_argument("--trained-results", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--allow-platform-difference", action="store_true",
                   help="Explicit user opt-in for a supplementary local evaluation; retain failed remote-parity receipts")
    args = p.parse_args()
    sidecar, original, results, out = [getattr(args, k).resolve() for k in
                                      ("sidecar", "training_cache", "trained_results", "output")]
    require(not out.exists(), "Output already exists: retain prior attempt, choose a new output")
    require(torch.cuda.is_available(), "CUDA required for the qualified evaluation")
    require(training.sha(Path(training.__file__)) == read(results / "completion_manifest.json")["training_script_sha256"],
            "Frozen mixed20k trainer changed")
    manifest = read(sidecar / "manifest.json")
    require(manifest["status"] == "complete" and manifest["scenes"] == 500,
            "Incomplete Torus review sidecar")
    require(manifest["training_cache_manifest_sha256"] == training.sha(original / "manifest.json")
            and manifest["training_statistics_sha256"] == training.sha(original / "statistics.json"),
            "Sidecar not bound to the current training cache")
    for name, expected in manifest["files_sha256"].items():
        require(training.sha(sidecar / name) == expected, f"Torus cache checksum mismatch: {name}")
    stats = read(sidecar / "statistics.json")
    with np.load(sidecar / "shared.npz", allow_pickle=False) as z:
        shared = {k: z[k] for k in z.files}
    points = np.load(sidecar / "geometry_points.npy", mmap_mode="r")
    normals = np.load(sidecar / "geometry_normals.npy", mmap_mode="r")
    old_points = np.load(original / "geometry_points.npy", mmap_mode="r")
    old_normals = np.load(original / "geometry_normals.npy", mmap_mode="r")
    completion = read(results / "completion_manifest.json")
    old_review = read(results / "review/review_results.json")
    records = read(sidecar / "topology_ood/records.json")
    group_names = np.asarray([r["ood_composition"] for r in records])
    nobjects = np.asarray([r["count"] for r in records])
    require(np.sum(group_names == "mixed") == 250 and np.sum(group_names == "all") == 250,
            "Subgroup composition changed")

    torch.set_num_threads(2)
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda")
    phi = torch.as_tensor(shared["phi_pinv"], device=device)
    out.mkdir(parents=True)
    started = time.perf_counter()
    reports = {}
    for method in training.METHODS:
        training.seed_all(0)
        mdir = results / method
        run = read(mdir / "run_manifest.json")
        expected = completion["methods"][method]
        checkpoint_path = mdir / "best.pt"
        require(training.sha(checkpoint_path) == expected["checkpoints"]["best.pt"]["sha256"],
                f"Downloaded best checkpoint checksum differs for {method}")
        require(run["training_statistics_sha256"] == manifest["training_statistics_sha256"],
                f"Normalization does not match {method} checkpoint")
        require(run["training_script_sha256"] == training.sha(Path(training.__file__)), "Training source mismatch")
        require(torch.__version__ == run["torch"], "Torus inference requires the training Torch version")
        require(training.sha(training.TIMM_INITIALIZER_PATH) == run["timm_0_9_2_initializer_sha256"],
                "Pinned timm initializer mismatch")
        for name, digest in run["model_source_sha256"].items():
            require(training.sha(ROOT / name) == digest, f"Model/metric source changed: {name}")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        require(checkpoint["run_manifest"] == run and checkpoint["epoch"] == expected["selected_epoch"],
                f"Best checkpoint provenance mismatch for {method}")
        require(len(read(mdir / "history.json")) == 200, "Training did not finish all 200 epochs")
        model = training.make_model(method).to(device)
        model.load_state_dict(checkpoint["model"], strict=True)
        model.eval()
        require(sum(p.numel() for p in model.parameters()) == expected["parameters"], "Model parameter count mismatch")
        del checkpoint

        old_ds = training.get_split(original, "id_test", method, stats, shared, old_points, old_normals)
        batch = training.collate_scenes([old_ds[i] for i in range(16)]).to(device)
        if method == "M2":
            # Official Transolver++ samples Gumbel slice assignments even in
            # eval mode. Preserve its algorithm; qualify replay with the same
            # RNG rather than incorrectly comparing independent random draws.
            rng = torch.cuda.get_rng_state_all()
            with torch.no_grad():
                first = model(batch)
                torch.cuda.set_rng_state_all(rng)
                second = model(batch)
            replay_difference = float((first - second).abs().max())
            require(replay_difference < 1e-6 and torch.isfinite(first).all(), "M2 fixed-RNG replay failed")
            parity = {"kind": "official_stochastic_eval_fixed_rng_replay", "max_prediction_difference": replay_difference,
                      "reason": "Official Gumbel sampling remains active in eval; historic draws are not exact replay targets"}
        else:
            with torch.no_grad():
                _, metrics = training.loss_and_scene_metrics(model(batch), batch, phi, stats, .1)
            differences = {}
            for key in ("scattered_rel_l2", "total_rel_l2", "q64_derived_rel_l2"):
                local = metrics[key].cpu().numpy().astype(np.float64)
                remote = np.asarray(old_review["methods"][method]["id_test"]["per_scene_metrics"][key][:16], np.float64)
                maximum = float(np.max(np.abs(local - remote)))
                differences[key] = maximum
            parity_passed = max(differences.values()) <= 2e-4
            require(parity_passed or args.allow_platform_difference,
                    f"Local/remote review parity failed: {method} {differences}")
            parity = {"kind": "remote_ID_first16_scene_metrics", "max_absolute_differences": differences,
                      "tolerance": 2e-4, "passed": parity_passed,
                      "supplementary_platform_difference_opt_in": args.allow_platform_difference}
        del batch, old_ds
        print(json.dumps({"event": "checkpoint_verified_and_runtime_parity_recorded", "method": method,
                          "selected_epoch": expected["selected_epoch"], "parity": parity}), flush=True)

        # One evaluation pass per method, fixed seed 0; no dropout/optimizer.
        # This explicitly fixes the official M2 Gumbel stream without changing
        # the model or selecting a favourable random draw.
        training.seed_all(0)
        ds = training.get_split(sidecar, "topology_ood", method, stats, shared, points, normals)
        ds.set_epoch(0)
        require(len(ds) == 500, "Wrong Torus scene count")
        loader, _ = training.make_loader(ds, shuffle=False, seed=9000, workers=0)
        torch.cuda.reset_peak_memory_stats()
        method_start = time.perf_counter()
        report = training.eval_split(model, loader, device, phi, stats, retain_scene_metrics=True)
        torch.cuda.synchronize()
        require(report["scenes"] == 500, "Evaluation omitted scenes")
        for key, values in report["per_scene_metrics"].items():
            require(len(values) == 500, f"Incomplete metric vector {key}")
            require(report[key] == summarize(values), f"Aggregate mismatch {key}")
        subgroups = {}
        for name in ("mixed", "all"):
            mask = group_names == name
            subgroups[name] = {"scenes": int(mask.sum()), **{
                key: summarize(np.asarray(values)[mask]) for key, values in report["per_scene_metrics"].items()}}
        count_groups = {}
        for count in (2, 3, 4):
            mask = nobjects == count
            count_groups[str(count)] = {"scenes": int(mask.sum()), **{
                key: summarize(np.asarray(values)[mask]) for key, values in report["per_scene_metrics"].items()}}
        report.update({"status": "complete", "method": method, "selected_epoch": expected["selected_epoch"],
                       "checkpoint_sha256": expected["checkpoints"]["best.pt"]["sha256"],
                       "subgroups": subgroups, "object_count_groups": count_groups,
                       "runtime_parity": parity, "evaluation_seed": 0,
                       "seconds": time.perf_counter() - method_start,
                       "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20})
        reports[method] = report
        training.write_json(out / f"{method}.json", report)
        print(json.dumps({"event": "torus_method_complete", "method": method, "scenes": 500,
                          "scattered_rel_l2": report["scattered_rel_l2"], "total_rel_l2": report["total_rel_l2"],
                          "mixed": subgroups["mixed"]["scattered_rel_l2"], "all": subgroups["all"]["scattered_rel_l2"],
                          "seconds": report["seconds"]}), flush=True)
        del model, ds, loader
        torch.cuda.empty_cache()

    training.write_json(out / "review_results.json", {
        "schema": "mixed20k-seven-method-torus-cross-run-review-v1", "status": "complete",
        "completed_utc": datetime.now(timezone.utc).isoformat(), "methods": reports,
        "source_manifest_sha256": manifest["source_manifest_sha256"],
        "sidecar_manifest_sha256": training.sha(sidecar / "manifest.json"),
        "training_statistics_sha256": manifest["training_statistics_sha256"],
        "evaluation_source_sha256": training.sha(Path(__file__)),
        "frozen_training_source_sha256": training.sha(Path(training.__file__)),
        "runtime": {"torch": torch.__version__, "cuda": torch.version.cuda,
                    "device": torch.cuda.get_device_name(device), "numpy": np.__version__,
                    "physicsnemo": importlib.metadata.version("nvidia-physicsnemo"),
                    "einops": importlib.metadata.version("einops"), "precision": "unchanged float32 high matmul",
                    "dgl": "annotation-only namespaces; no DGL graph operation; qualified edge-index backend"},
        "elapsed_seconds": time.perf_counter() - started, "new_training_or_FEM_BEM_solves": 0,
        "supplementary_local_evaluation": args.allow_platform_difference,
        "remote_equivalence_passed_for_deterministic_methods": all(
            r["runtime_parity"].get("passed", False) for m, r in reports.items() if m != "M2"),
        "interpretation": "Previously viewed Torus-family review; v6s11 includes other genus-1 families, so this is unseen-family transfer, not first exposure to holes",
        "v6s11_scientific_release": False,
        "M2_eval": "one fixed-seed-0 Gumbel pass using the unchanged official stochastic inference"})
    print(json.dumps({"event": "all_torus_reviews_complete", "methods": list(reports), "output": str(out)}), flush=True)


if __name__ == "__main__":
    main()
