"""Matched evaluation of Transolver representations on ID and Peanut Shape-OOD."""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import train_v3_transolver_representations as experiment

from cspgnet.official_pibnet_edge_backend import collate_edge_index_graphs
from cspgnet.v3_learning import loss_and_scene_metrics
from cspgnet.v3_transolver_representations import make_dataset, make_model


DATA_ROOT = ROOT / "datasets/canonical_response_v3/data"
AUDIT_ROOT = ROOT / "outputs/v3_sealed_bem_audit_current_20260927"
OUTPUT = experiment.OUT / "evaluation_viewed_id_and_peanut"
SPLITS = ("id_test", "shape_ood")


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path: Path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summarize(values):
    array = np.asarray(values, dtype=np.float64)
    return {"mean": float(array.mean()), "median": float(np.median(array)),
            "p95": float(np.quantile(array, 0.95))}


def checkpoint_for(arm: str) -> Path:
    if arm == "s-response":
        return experiment.S_RUN / "best.pt"
    return experiment.OUT / f"{arm}_seed0_100ep" / "best.pt"


def validate_checkpoint(arm: str, path: Path, spec_hash: str | None):
    run_path = path.parent / "run_manifest.json"
    if not run_path.exists():
        raise FileNotFoundError(f"missing run manifest for {arm}: {run_path}")
    manifest = read_json(run_path)
    state = torch.load(path, map_location="cpu", weights_only=False)
    if state.get("epoch") != 100:
        raise RuntimeError(f"{arm} best checkpoint is not from the 100-epoch schedule")
    if arm == "s-response":
        old_spec_path = experiment.S_RUN.parent / "trial_spec.json"
        old_spec = read_json(old_spec_path)
        old_hash = sha(old_spec_path)
        if manifest.get("trial_spec_sha256") != old_hash:
            raise RuntimeError("S(Z) checkpoint trial-spec hash mismatch")
        if (old_spec.get("train_manifest_sha256") != experiment.sha(experiment.TRAIN / "manifest.json")
                or old_spec.get("val_manifest_sha256") != experiment.sha(experiment.VAL / "manifest.json")):
            raise RuntimeError("S(Z) checkpoint used different train/validation manifests")
    else:
        if manifest.get("trial_spec_sha256") != spec_hash:
            raise RuntimeError(f"{arm} checkpoint comparison protocol hash mismatch")
        if manifest.get("arm") != arm or manifest.get("seed") != 0:
            raise RuntimeError(f"{arm} checkpoint identity mismatch")
    return manifest, state


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this matched evaluation")
    spec = experiment.get_spec(freeze=True)
    spec_path = experiment.OUT / "trial_spec.json"
    spec_hash = sha(spec_path)
    if OUTPUT.exists() and (OUTPUT / "summary.json").exists():
        raise FileExistsError(f"evaluation already completed: {OUTPUT}")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    stats = read_json(experiment.STATS)
    train_scales = read_json(experiment.S_CACHE_META)["degree_complex_rms"]
    audit_details = {}
    for split in SPLITS:
        root = DATA_ROOT / split
        manifest = read_json(root / "manifest.json")
        if manifest.get("status") != "complete" or not manifest.get("sealed"):
            raise RuntimeError(f"split is incomplete or not sealed: {split}")
        audit = read_json(AUDIT_ROOT / f"{split}.json")
        if audit.get("status") != "passed" or audit.get("data_manifest_sha256") != sha(root / "manifest.json"):
            raise RuntimeError(f"independent BEM audit provenance failure: {split}")
        audit_details[split] = {"scenes": len(manifest["samples"]),
                                "manifest_sha256": sha(root / "manifest.json"),
                                "audit_scenes": len(audit.get("selected_indices", [])),
                                "audit_sha256": sha(AUDIT_ROOT / f"{split}.json")}

    evaluation = {"schema": "v3-transolver-representation-evaluation-v1",
                  "status": "exploratory_post_hoc; these ID/Peanut splits were previously viewed",
                  "protocol_sha256": spec_hash, "splits": audit_details,
                  "arms": {}, "source_sha256": sha(Path(__file__)),
                  "pointcloud_cache_root": str(experiment.POINTCLOUD_CACHE)}
    device = torch.device("cuda")
    total_started = time.perf_counter()
    for arm in experiment.ARMS:
        checkpoint_path = checkpoint_for(arm)
        run_manifest, checkpoint = validate_checkpoint(arm, checkpoint_path, spec_hash)
        model = make_model(arm).to(device)
        model.load_state_dict(checkpoint["model"])
        model.eval()
        parameter_count = sum(parameter.numel() for parameter in model.parameters())
        results = {}
        arm_started = time.perf_counter()
        for split in SPLITS:
            root = DATA_ROOT / split
            dataset = make_dataset(
                arm, root, stats, phase="evaluation", s_cache=experiment.S_CACHE,
                train_s_scales=train_scales, pointcloud_cache=experiment.POINTCLOUD_CACHE,
            )
            if len(dataset) != audit_details[split]["scenes"]:
                raise RuntimeError(f"dataset size changed while evaluating {arm}/{split}")
            loader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=0,
                                collate_fn=collate_edge_index_graphs)
            phi_pinv = dataset.phi_pinv.to(device)
            arrays = {}
            sample_indices = []
            split_started = time.perf_counter()
            with torch.inference_mode():
                for batch in loader:
                    batch = batch.to(device)
                    prediction = model(batch)
                    _, scene_metrics = loss_and_scene_metrics(
                        prediction, batch, phi_pinv, stats, experiment.LAMBDA_Q, "scattered"
                    )
                    for name, values in scene_metrics.items():
                        arrays.setdefault(name, []).extend(values.detach().cpu().tolist())
                    counts = batch.sample_node_counts.detach().cpu().long()
                    starts = torch.cat((torch.zeros(1, dtype=torch.long), counts.cumsum(0)[:-1]))
                    sample_indices.extend(batch.sample_index.detach().cpu()[starts].tolist())
            if len(sample_indices) != len(dataset):
                raise RuntimeError(f"wrong number of scenes evaluated for {arm}/{split}")
            arrays = {name: np.asarray(values, dtype=np.float32) for name, values in arrays.items()}
            path = OUTPUT / f"{arm}_{split}.npz"
            pending = path.with_name(path.name + ".pending")
            with pending.open("wb") as handle:
                np.savez_compressed(handle, sample_indices=np.asarray(sample_indices, dtype=np.int32), **arrays)
            os.replace(pending, path)
            metrics = {name: summarize(values) for name, values in arrays.items()}
            results[split] = {"metrics": metrics, "metrics_file": path.name,
                              "seconds": time.perf_counter() - split_started}
            print(json.dumps({"arm": arm, "split": split,
                              "scattered_rel_l2": metrics["scattered_rel_l2"]}, ensure_ascii=False),
                  flush=True)
        evaluation["arms"][arm] = {
            "checkpoint": str(checkpoint_path), "checkpoint_sha256": sha(checkpoint_path),
            "checkpoint_epoch": checkpoint["epoch"], "run_manifest": run_manifest,
            "parameters": parameter_count, "splits": results,
            "seconds": time.perf_counter() - arm_started,
        }
        experiment.save_json(OUTPUT / f"{arm}.json", evaluation["arms"][arm])
        del model
        torch.cuda.empty_cache()
    evaluation["total_seconds"] = time.perf_counter() - total_started
    experiment.save_json(OUTPUT / "summary.json", evaluation)
    print(json.dumps({"status": "complete", "summary": str(OUTPUT / "summary.json"),
                      "seconds": evaluation["total_seconds"]}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
