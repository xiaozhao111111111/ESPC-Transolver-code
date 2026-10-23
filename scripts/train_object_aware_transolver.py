"""Frozen, restartable 3-arm warm-start comparison; 40 additional epochs each."""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
import traceback
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import train_v3_transolver_comparison as base
sys.path.insert(0, str(ROOT / "work/python_deps_repaired"))
import einops
import numpy as np
import torch
from cspgnet.v3_object_transolver import ARMS, ObjectAwareTransolver
from cspgnet.v3_transolver_tensor_cache import TensorSceneDataset, build_cache, collate_tensors, sha, write_json
from cspgnet.v3_learning import loss_and_scene_metrics

OUT = ROOT / "outputs/v3_object_transolver_40ep_20260930"
CACHE = OUT / "tensor_cache"
PARENT = ROOT / "outputs/v3_transolver_100ep_v2_20260929/transolver_seed0_100ep/best.pt"
TESTS = {"id_test": 1000, "shape_ood": 500, "n8_test": 1000}
SCOPES = ("validation", *TESTS)
SOURCES = [Path(__file__), ROOT / "src/cspgnet/v3_object_transolver.py",
           ROOT / "scripts/supervise_object_transolver.py", ROOT / "scripts/object_transolver_worker.py",
           ROOT / "src/cspgnet/v3_transolver_tensor_cache.py", ROOT / "src/cspgnet/v3_transolver_comparison.py",
           ROOT / "src/cspgnet/v3_learning.py", ROOT / "src/cspgnet/v3_feature_trials.py",
           ROOT / "third_party/Transolver/PDE-Solving-StandardBenchmark/model/Transolver_Irregular_Mesh.py",
           ROOT / "third_party/Transolver/PDE-Solving-StandardBenchmark/model/Physics_Attention.py"]


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def atomic_torch(path, state):
    pending = path.with_name(path.name + ".pending")
    torch.save(state, pending)
    os.replace(pending, path)


def statistics_and_scales():
    statistics = read(base.STATS)
    scales = read(base.OLD_TRIAL / "cache/train/cache_manifest.json")["degree_complex_rms"]
    return statistics, scales


def protocol():
    frozen = read(OUT / "protocol.json")
    for name, digest in frozen["source_sha256"].items():
        if sha(ROOT / name) != digest:
            raise RuntimeError(f"frozen implementation changed: {name}")
    for path, digest in frozen["input_sha256"].items():
        if sha(ROOT / path) != digest:
            raise RuntimeError(f"frozen input changed: {path}")
    if einops.__version__ != "0.8.1" or "python_deps_repaired" not in str(einops.__file__):
        raise RuntimeError("pinned einops dependency differs")
    return frozen


def prepare():
    OUT.mkdir(parents=True, exist_ok=True)
    statistics, scales = statistics_and_scales()
    parent = torch.load(PARENT, map_location="cpu", weights_only=False)
    if parent["epoch"] != 100 or parent["manifest"].get("seed") != 0:
        raise RuntimeError("expected completed seed-0 epoch-100 parent")
    for split, root in (("train", base.TRAIN), ("val", base.VAL)):
        build_cache(root, base.OLD_TRIAL / f"cache/{split}", CACHE / split, statistics, scales)
    write_json(OUT / "preparation.json", {"status": "ready_for_preflight", "parent_epoch": 100,
                                        "parent_sha256": sha(PARENT), "utc": now()})


def loaders():
    train, val = TensorSceneDataset(CACHE / "train"), TensorSceneDataset(CACHE / "val")
    if len(train) != 10000 or len(val) != 1000 or not torch.equal(train.phi_pinv, val.phi_pinv):
        raise RuntimeError("wrong train/validation tensor contract")
    return train, val


def all_batches(dataset, indices=None):
    indices = np.arange(len(dataset)) if indices is None else indices
    for offset in range(0, len(indices), 16):
        yield collate_tensors([dataset[int(index)] for index in indices[offset:offset + 16]])


def append_metrics(collected, metrics, loss, scenes):
    collected.setdefault("losses", []).extend([float(loss.detach())] * scenes)
    for name, tensor in metrics.items():
        collected.setdefault(name, []).extend(tensor.detach().cpu().tolist())


def summarize(collected):
    return {"loss": float(np.mean(collected["losses"])), "scenes": len(collected["losses"]),
            **{name: {"mean": float(np.mean(values)), "median": float(np.median(values)),
                      "p95": float(np.quantile(values, 0.95))}
               for name, values in collected.items() if name != "losses"}}


def evaluate(model, dataset, statistics, phi, status_path=None, scope="validation"):
    model.eval()
    device = next(model.parameters()).device
    collected = {}
    indices = []
    with torch.no_grad():
        for step, graph in enumerate(all_batches(dataset)):
            graph = graph.to(device)
            prediction = model(graph)
            loss, metrics = loss_and_scene_metrics(prediction, graph, phi, statistics, 0.1)
            if not torch.isfinite(loss):
                raise ArithmeticError("nonfinite evaluation")
            append_metrics(collected, metrics, loss, len(graph.sample_node_counts))
            indices.extend(dataset.meta["sample_indices"][step * 16:(step + 1) * 16])
            if status_path is not None and step % 10 == 0:
                write_json(status_path, {"status": "evaluating", "scope": scope, "step": step,
                                          "last_progress_utc": now(), "pid": os.getpid()})
    return summarize(collected), collected, indices


def preflight():
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision("high")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    train, val = loaders()
    statistics, scales = statistics_and_scales()
    parent = torch.load(PARENT, map_location="cpu", weights_only=False)
    legacy_train = base.STransformerDataset(base.TRAIN, statistics, base.OLD_TRIAL / "cache/train", scales)
    legacy_val = base.STransformerDataset(base.VAL, statistics, base.OLD_TRIAL / "cache/val", scales)
    parity = {}
    for name, cached, legacy in (("train", train, legacy_train), ("val", val, legacy_val)):
        errors = []
        for index in (0, 1, 2, 7, 19, 101):
            new = collate_tensors([cached[index]])
            old = legacy[index]
            for field in ("node_features", "labels", "incident", "q64_gt", "z_real", "z_imag", "sample_node_counts", "sample_index"):
                error = float((getattr(new, field) - getattr(old, field)).abs().max())
                errors.append(error)
                if error > 1e-6:
                    raise AssertionError(f"tensor cache mismatch: {name}/{index}/{field} {error}")
        parity[name] = max(errors)
    device = torch.device("cuda")
    graph = collate_tensors([train[index] for index in range(16)]).to(device)
    phi = train.phi_pinv.to(device)
    original = base.GroupedTransolver("transolver").to(device).eval()
    original.load_state_dict(parent["model"], strict=True)
    with torch.no_grad():
        original_prediction = original(graph)
    reports = {}
    initial_branch = None
    for arm in ARMS:
        base.seed_everything(0)
        model = ObjectAwareTransolver(arm).to(device)
        model.load_parent(parent["model"])
        with torch.no_grad():
            model.eval()
            discrepancy = float((model(graph) - original_prediction).abs().max())
        if discrepancy > 2e-5:
            raise AssertionError(f"zero-adapter differs from parent: {arm} {discrepancy}")
        if model.object_branch is not None:
            weights = {key: value.detach().cpu().clone() for key, value in model.object_branch.state_dict().items()}
            if initial_branch is None:
                initial_branch = weights
            elif not all(torch.equal(value, initial_branch[key]) for key, value in weights.items()):
                raise AssertionError("adapter initial parameters differ")
            model.object_branch.alpha.data.fill_(1.0)
            sample_index = next(index for index in range(len(train))
                                if train.meta["offsets"][index + 1] - train.meta["offsets"][index] == 384)
            raw = collate_tensors([train[sample_index]]).node_features.to(device).reshape(1, 384, 140)
            permutation = torch.tensor([2, 0, 1], device=device)
            hidden = model.backbone.preprocess(raw)
            before = model.object_branch(hidden, raw).reshape(1, 3, 128, 256)
            after = model.object_branch(hidden.reshape(1, 3, 128, 256)[:, permutation].reshape(1, 384, 256),
                                        raw.reshape(1, 3, 128, 140)[:, permutation].reshape(1, 384, 140))
            permutation_error = float((after.reshape(1, 3, 128, 256) - before[:, permutation]).abs().max().detach())
            if permutation_error > 2e-5:
                raise AssertionError("object order changes adapter")
            singleton = model.object_branch(hidden[:, :128], raw[:, :128])
            if torch.count_nonzero(singleton):
                raise AssertionError("single object received an inter-object message")
            model.object_branch.alpha.data.zero_()
        else:
            permutation_error = 0.0
        model.train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
        torch.cuda.reset_peak_memory_stats()
        timings = []
        for _ in range(4):
            torch.cuda.synchronize()
            start = time.perf_counter()
            loss, _ = loss_and_scene_metrics(model(graph), graph, phi, statistics, 0.1)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            torch.cuda.synchronize()
            timings.append(time.perf_counter() - start)
        if model.object_branch is not None and float(model.object_branch.alpha.abs().detach()) == 0:
            raise AssertionError("zero gate cannot learn")
        reports[arm] = {"parent_output_max_abs": discrepancy, "object_permutation_max_abs": permutation_error,
                        "parameters": sum(p.numel() for p in model.parameters()),
                        "step_seconds_median": float(np.median(timings[1:])),
                        "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
                        "peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20}
        del model, optimizer
        torch.cuda.empty_cache()
    del original, graph
    torch.cuda.empty_cache()
    if reports["object-joint"]["parameters"] != reports["object-sum"]["parameters"]:
        raise AssertionError("adapter parameter budgets differ")
    estimated_gpu_hours = sum(item["step_seconds_median"] for item in reports.values()) * 625 * 40 / 3600
    report = {"status": "passed", "utc": now(), "tensor_parity": parity, "arms": reports,
              "estimated_training_gpu_hours_excluding_validation_and_contention": estimated_gpu_hours,
              "conservative_three_process_vram_mib": sum(item["peak_reserved_mib"] for item in reports.values()) + 1800,
              "gpu_total_mib": torch.cuda.get_device_properties(0).total_memory / 2**20}
    write_json(OUT / "preflight.json", report)
    inputs = [PARENT, base.TRAIN / "manifest.json", base.VAL / "manifest.json", base.STATS,
              base.OLD_TRIAL / "cache/train/cache_manifest.json", base.OLD_TRIAL / "cache/val/cache_manifest.json",
              CACHE / "train/manifest.json", CACHE / "val/manifest.json"]
    frozen = {"schema": "v3-object-transolver-warmstart-three-arm-v1", "status": "frozen_after_preflight",
              "arms": list(ARMS), "seed": 0, "parent_epoch": 100, "additional_epochs": 40,
              "absolute_epochs": [101, 140], "batch_size": 16, "learning_rate": 1e-4,
              "minimum_learning_rate": 1e-7, "weight_decay": 1e-5, "lambda_q": 0.1,
              "optimizer": "fresh AdamW in all three arms", "scheduler": "fresh cosine T_max=40 in all three arms",
              "gradient_clip": 1.0, "checkpoint_interval_steps": 100, "torch_cpu_threads_per_process": 2,
              "selection": "lowest validation mean per-scene scattered relative L2 among the 40 added epochs",
              "report": "both last-epoch-40 and validation-selected checkpoints; no test-based selection",
              "test_counts": TESTS, "single_seed_exploratory": True,
              "source_sha256": {str(path.relative_to(ROOT)): sha(path) for path in SOURCES},
              "input_sha256": {str(path.relative_to(ROOT)): sha(path) for path in inputs},
              "preflight_sha256": sha(OUT / "preflight.json")}
    path = OUT / "protocol.json"
    if path.exists() and read(path) != frozen:
        raise RuntimeError("protocol already frozen with different contents")
    write_json(path, frozen)
    print(json.dumps(report), flush=True)


def train(arm):
    frozen = protocol()
    base.seed_everything(0)
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision("high")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    device = torch.device("cuda")
    statistics, _ = statistics_and_scales()
    training, validation = loaders()
    phi = training.phi_pinv.to(device)
    model = ObjectAwareTransolver(arm).to(device)
    model.load_parent(torch.load(PARENT, map_location="cpu", weights_only=False)["model"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=40, eta_min=1e-7)
    run = OUT / arm
    run.mkdir(parents=True, exist_ok=True)
    manifest = {"schema": "v3-object-transolver-run-v1", "arm": arm, "seed": 0, "device": "cuda",
                "torch": torch.__version__, "protocol_sha256": sha(OUT / "protocol.json"),
                "parameters": sum(p.numel() for p in model.parameters())}
    if (run / "run_manifest.json").exists() and read(run / "run_manifest.json") != manifest:
        raise RuntimeError("resume run manifest differs")
    write_json(run / "run_manifest.json", manifest)
    history, best, completed, cursor, collected, elapsed_before = [], float("inf"), 0, 0, {}, 0.0
    last = run / "last.pt"
    if last.exists():
        state = torch.load(last, map_location=device, weights_only=False)
        if state["manifest"] != manifest:
            raise RuntimeError("resume checkpoint differs")
        model.load_state_dict(state["model"], strict=True)
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        history, best, completed, cursor = state["history"], state["best_score"], state["epoch"], state["cursor"]
        collected, elapsed_before = state["partial_metrics"], state["elapsed_seconds"]
        random.setstate(state["rng"]["python"])
        np.random.set_state(state["rng"]["numpy"])
        torch.set_rng_state(state["rng"]["torch"].cpu())
        torch.cuda.set_rng_state_all([value.cpu() for value in state["rng"]["cuda"]])
        if len(history) != completed:
            raise RuntimeError("checkpoint epoch history inconsistent")
        if not (0 <= completed <= 40 and 0 <= cursor <= 625):
            raise RuntimeError("checkpoint cursor bounds inconsistent")
        if state.get("absolute_epoch") != 100 + completed or scheduler.last_epoch != completed:
            raise RuntimeError("checkpoint schedule/absolute epoch inconsistent")
        if [row["epoch"] for row in history] != list(range(1, completed + 1)):
            raise RuntimeError("checkpoint history ordering inconsistent")
        if completed == 40 and cursor != 0:
            raise RuntimeError("completed checkpoint still has pending work")
        expected_scenes = min(cursor * 16, 10000)
        if collected and (len(collected.get("losses", [])) != expected_scenes or
                          any(len(values) != expected_scenes for values in collected.values())):
            raise RuntimeError("checkpoint partial metrics inconsistent")
        if cursor and not collected:
            raise RuntimeError("checkpoint cursor has no partial metrics")
        if completed:
            selected = min(history, key=lambda row: row["val"]["scattered_rel_l2"]["mean"])
            if not np.isclose(best, selected["val"]["scattered_rel_l2"]["mean"], rtol=0, atol=1e-12):
                raise RuntimeError("checkpoint best score inconsistent")
            best_path = run / "best.pt"
            best_state = torch.load(best_path, map_location="cpu", weights_only=False) if best_path.exists() else None
            if best_state is None or best_state.get("epoch") != selected["epoch"]:
                if selected["epoch"] == completed and cursor == 0:
                    atomic_torch(best_path, state)
                else:
                    raise RuntimeError("best checkpoint inconsistent; historical model cannot be reconstructed")
            elif best_state["manifest"] != manifest:
                raise RuntimeError("best checkpoint manifest inconsistent")
            write_json(run / "history.json", history)
    started = time.perf_counter()
    status_path = run / "status.json"

    def checkpoint():
        state = {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                 "manifest": manifest, "epoch": completed, "absolute_epoch": 100 + completed, "cursor": cursor,
                 "history": history, "partial_metrics": collected, "best_score": best,
                 "elapsed_seconds": elapsed_before + time.perf_counter() - started,
                 "rng": {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(),
                         "cuda": torch.cuda.get_rng_state_all()}}
        atomic_torch(last, state)
        return state

    if not last.exists():
        checkpoint()
    try:
        if not (run / "initial_validation.json").exists():
            initial, _, _ = evaluate(model, validation, statistics, phi, status_path)
            write_json(run / "initial_validation.json", initial)
        for epoch in range(completed + 1, 41):
            permutation = np.random.default_rng(20260930 + epoch).permutation(len(training))
            model.train()
            steps = (len(training) + 15) // 16
            for step in range(cursor, steps):
                indices = permutation[step * 16:(step + 1) * 16]
                graph = collate_tensors([training[int(index)] for index in indices]).to(device)
                loss, metrics = loss_and_scene_metrics(model(graph), graph, phi, statistics, 0.1)
                if not torch.isfinite(loss):
                    raise ArithmeticError("nonfinite training loss")
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
                optimizer.step()
                append_metrics(collected, metrics, loss, len(indices))
                cursor = step + 1
                if cursor % 25 == 0 or cursor == steps:
                    write_json(status_path, {"status": "training", "arm": arm, "epoch": epoch,
                                              "completed_epochs": completed, "step": cursor, "steps": steps,
                                              "last_progress_utc": now(), "pid": os.getpid()})
                if cursor % 100 == 0:
                    checkpoint()
            val_metrics, _, _ = evaluate(model, validation, statistics, phi, status_path)
            scheduler.step()
            row = {"epoch": epoch, "absolute_epoch": 100 + epoch, "train": summarize(collected), "val": val_metrics,
                   "learning_rate": scheduler.get_last_lr()[0],
                   "elapsed_seconds": elapsed_before + time.perf_counter() - started,
                   "adapter_gate": None if model.object_branch is None else float(model.object_branch.alpha.detach())}
            score = val_metrics["scattered_rel_l2"]["mean"]
            improved = score < best
            best = min(best, score)
            history.append(row)
            completed, cursor, collected = epoch, 0, {}
            state = checkpoint()
            if improved:
                atomic_torch(run / "best.pt", state)
            write_json(run / "history.json", history)
            print(json.dumps(row), flush=True)
        write_json(run / "completion.json", {"status": "complete", "arm": arm, "additional_epochs": completed,
                                              "last_sha256": sha(last), "best_sha256": sha(run / "best.pt"),
                                              "best_epoch": min(history, key=lambda row: row["val"]["scattered_rel_l2"]["mean"])["epoch"],
                                              "protocol_sha256": manifest["protocol_sha256"], "utc": now()})
        write_json(status_path, {"status": "complete", "arm": arm, "completed_epochs": completed,
                                  "last_progress_utc": now(), "pid": os.getpid()})
    except BaseException as error:
        write_json(status_path, {"status": "failed", "arm": arm, "completed_epochs": completed,
                                  "cursor": cursor, "error": str(error), "last_progress_utc": now(), "pid": os.getpid()})
        raise


def prepare_tests():
    protocol()
    for arm in ARMS:
        if read(OUT / arm / "completion.json")["additional_epochs"] != 40:
            raise RuntimeError("all training must finish before test cache preparation")
    statistics, scales = statistics_and_scales()
    for split, count in TESTS.items():
        root = ROOT / f"datasets/canonical_response_v3/data/{split}"
        if len(read(root / "manifest.json")["samples"]) != count:
            raise RuntimeError("test size differs")
        build_cache(root, None, CACHE / split, statistics, scales)
    write_json(OUT / "test_cache_ready.json", {"status": "complete", "utc": now()})


def test(arm):
    protocol()
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision("high")
    run = OUT / arm
    if read(run / "completion.json")["additional_epochs"] != 40:
        raise RuntimeError("training incomplete")
    statistics, _ = statistics_and_scales()
    if not (OUT / "test_cache_ready.json").exists():
        raise RuntimeError("single-writer test cache preparation required")
    datasets = {"validation": TensorSceneDataset(CACHE / "val")}
    for split, count in TESTS.items():
        root = ROOT / f"datasets/canonical_response_v3/data/{split}"
        if len(read(root / "manifest.json")["samples"]) != count:
            raise RuntimeError("test size differs")
        datasets[split] = TensorSceneDataset(CACHE / split)
    device = torch.device("cuda")
    model = ObjectAwareTransolver(arm).to(device)
    report = {"arm": arm, "status": "exploratory_single_seed", "checkpoints": {}}
    for kind in ("last", "best"):
        path = run / f"{kind}.pt"
        state = torch.load(path, map_location=device, weights_only=False)
        model.load_state_dict(state["model"], strict=True)
        item = {"epoch": state["epoch"], "checkpoint_sha256": sha(path), "splits": {}}
        for name, dataset in datasets.items():
            sidecar = run / f"evaluation_{kind}_{name}.json"
            array_path = sidecar.with_suffix(".npz")
            if sidecar.exists():
                saved = read(sidecar)
                if saved["checkpoint_sha256"] != item["checkpoint_sha256"] or not array_path.exists():
                    raise RuntimeError("prior evaluation differs")
                item["splits"][name] = saved["result"]
                continue
            start = time.perf_counter()
            result, arrays, indices = evaluate(model, dataset, statistics, dataset.phi_pinv.to(device), run / "evaluation_status.json", name)
            result["evaluation_seconds"] = time.perf_counter() - start
            np.savez_compressed(array_path, sample_indices=np.asarray(indices), **{key: np.asarray(value) for key, value in arrays.items() if key != "losses"})
            write_json(sidecar, {"checkpoint_sha256": item["checkpoint_sha256"], "result": result,
                                 "tensor_manifest_sha256": sha(dataset.root / "manifest.json")})
            item["splits"][name] = result
        report["checkpoints"][kind] = item
    write_json(run / "evaluation.json", report)
    print(json.dumps({"status": "evaluation_complete", "arm": arm}), flush=True)


def compare():
    protocol()
    results = {arm: read(OUT / arm / "evaluation.json") for arm in ARMS}
    paired = {}
    for kind in ("last", "best"):
        paired[kind] = {}
        for name in SCOPES:
            with np.load(OUT / "baseline" / f"evaluation_{kind}_{name}.npz") as reference:
                indices, baseline = reference["sample_indices"], reference["scattered_rel_l2"]
            paired[kind][name] = {}
            for arm in ARMS[1:]:
                with np.load(OUT / arm / f"evaluation_{kind}_{name}.npz") as candidate:
                    if not np.array_equal(indices, candidate["sample_indices"]):
                        raise RuntimeError("paired scene IDs differ")
                    difference = candidate["scattered_rel_l2"] - baseline
                paired[kind][name][arm] = {"mean_difference": float(difference.mean()),
                    "median_difference": float(np.median(difference)), "fraction_scenes_improved": float(np.mean(difference < 0))}
    write_json(OUT / "comparison.json", {"status": "complete", "single_seed_exploratory": True,
                                         "results": results, "paired_scattered_errors": paired})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--action", choices=("prepare", "preflight", "train", "prepare-tests", "test", "compare"), required=True)
    parser.add_argument("--arm", choices=ARMS)
    args = parser.parse_args()
    if args.action in ("train", "test") and args.arm is None:
        parser.error("--arm required")
    {"prepare": prepare, "preflight": preflight, "train": lambda: train(args.arm), "prepare-tests": prepare_tests,
     "test": lambda: test(args.arm), "compare": compare}[args.action]()


if __name__ == "__main__":
    main()
