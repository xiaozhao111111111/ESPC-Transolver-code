"""One frozen, restartable 40-epoch object-compositional Transolver experiment."""
from __future__ import annotations

import argparse
import io
import json
import os
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import train_object_aware_transolver as common
import einops
import numpy as np
import torch
from cspgnet.v3_compositional_transolver import CompositionalTransolver
from cspgnet.v3_object_transolver import ObjectAwareTransolver
from cspgnet.v3_transolver_tensor_cache import TensorSceneDataset, collate_tensors, sha, write_json
from cspgnet.v3_learning import loss_and_scene_metrics

OUT = ROOT / "outputs/v3_compositional_transolver_40ep_20260930"
RUN = OUT / "run"
CACHE = common.CACHE
PARENT = common.PARENT
EPOCHS = 40
SCOPES = {"validation": ("val", 1000), "id_test": ("id_test", 1000),
          "shape_ood": ("shape_ood", 500), "n8_test": ("n8_test", 1000)}
SOURCES = [Path(__file__), ROOT / "src/cspgnet/v3_compositional_transolver.py",
           ROOT / "scripts/supervise_compositional_transolver.py",
           ROOT / "COMPOSITIONAL_TRANSOLVER_40EP.md", *common.SOURCES]


def datasets():
    training, validation = common.loaders()
    if not all((CACHE / folder / "manifest.json").exists() for folder, _ in SCOPES.values()):
        raise RuntimeError("existing completed tensor caches required; do not regenerate data")
    return training, validation


def configure():
    common.base.seed_everything(0)
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision("high")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")


def verify_protocol():
    frozen = common.read(OUT / "protocol.json")
    for key, digest in frozen["source_sha256"].items():
        if sha(ROOT / key) != digest:
            raise RuntimeError(f"frozen implementation changed: {key}")
    for key, digest in frozen["input_sha256"].items():
        if sha(ROOT / key) != digest:
            raise RuntimeError(f"frozen input changed: {key}")
    if einops.__version__ != "0.8.1" or "python_deps_repaired" not in str(einops.__file__):
        raise RuntimeError("pinned dependency identity differs")
    return frozen


def parent_model(device):
    model = CompositionalTransolver().to(device)
    parent = torch.load(PARENT, map_location="cpu", weights_only=False)
    if parent["epoch"] != 100 or parent["manifest"].get("seed") != 0:
        raise RuntimeError("expected original completed epoch-100 seed-0 parent")
    model.load_parent(parent["model"])
    return model


def preflight():
    if (OUT / "protocol.json").exists():
        verify_protocol()
        print(json.dumps(common.read(OUT / "preflight.json")), flush=True)
        return
    common.protocol()  # Preserve/check every frozen implementation of the prior run.
    configure()
    training, validation = datasets()
    checksums = {}
    for folder in ("train", *dict.fromkeys(folder for folder, _ in SCOPES.values())):
        dataset = TensorSceneDataset(CACHE / folder)
        for name, expected in dataset.meta["files_sha256"].items():
            if sha(dataset.root / f"{name}.npy") != expected:
                raise RuntimeError(f"modified tensor cache: {folder}/{name}")
        checksums[folder] = "all stored array checksums matched"
    device = torch.device("cuda")
    statistics, _ = common.statistics_and_scales()
    phi = training.phi_pinv.to(device)
    model = parent_model(device).eval()
    reference = ObjectAwareTransolver("object-sum")
    count = sum(parameter.numel() for parameter in model.parameters())
    if count != sum(parameter.numel() for parameter in reference.parameters()):
        raise AssertionError("parameter budget differs from object-sum")
    del reference
    sample = next(i for i in range(len(training))
                  if training.meta["offsets"][i+1] - training.meta["offsets"][i] == 384)
    raw = collate_tensors([training[sample]]).node_features.to(device).reshape(1, 384, 140)
    permutation = torch.tensor([2, 0, 1], device=device)
    reordered = raw.reshape(1, 3, 128, 140)[:, permutation].reshape_as(raw)
    shifted = raw.clone()
    shifted[..., 133:136] += torch.tensor([1.7, -2.1, 0.6], device=device)
    # Exercise the relative adapter at nonzero strength, not just a zero gate.
    model.object_branch.alpha.data.fill_(1.0)
    with torch.no_grad():
        before = model._forward_scene_batch(raw).reshape(1, 3, 128, 2)
        after = model._forward_scene_batch(reordered).reshape_as(before)
        permutation_error = float((after - before[:, permutation]).abs().max())
        shift_error = float((model._forward_scene_batch(shifted).reshape_as(before) - before).abs().max())
        embedding = model.local_embedding(raw)
        center_error = float((model.local_embedding(shifted) - embedding).abs().max())
        changed_neighbor = raw.clone()
        changed_neighbor[:, 128:, :128] *= 1.73
        local_error = float((model.local_embedding(changed_neighbor)[0] - embedding[0]).abs().max())
        singleton = model.object_branch(embedding[:1], raw[:, :128])
        if torch.count_nonzero(singleton):
            raise AssertionError("N=1 received an inter-object message")
        synthetic8 = raw[:, :128].repeat(1, 8, 1)
        synthetic8.reshape(1, 8, 128, 140)[..., 133] += torch.arange(8, device=device)[None, :, None] * 3.0
        if not torch.isfinite(model._forward_scene_batch(synthetic8)).all():
            raise AssertionError("N=8 forward is nonfinite")
    if max(permutation_error, shift_error, center_error, local_error) > 2e-5:
        raise AssertionError("permutation/relative-position/local-isolation gate failed")
    calls = []
    hook = model.object_branch.register_forward_hook(lambda *args: calls.append(id(args[0])))
    with torch.no_grad():
        model._forward_scene_batch(raw)
    hook.remove()
    if len(calls) != 3 or len(set(calls)) != 1:
        raise AssertionError("expected three calls to exactly one shared adapter")
    model.object_branch.alpha.data.zero_()
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    graph = collate_tensors([training[i] for i in range(16)]).to(device)
    torch.cuda.reset_peak_memory_stats()

    def update(network, optim):
        loss, _ = loss_and_scene_metrics(network(graph), graph, phi, statistics, 0.1)
        if not torch.isfinite(loss):
            raise ArithmeticError("nonfinite preflight loss")
        optim.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(network.parameters(), 1.0, error_if_nonfinite=True)
        optim.step()
        return float(loss.detach())

    timings = []
    for _ in range(4):
        torch.cuda.synchronize()
        start = time.perf_counter()
        update(model, optimizer)
        torch.cuda.synchronize()
        timings.append(time.perf_counter() - start)
    peak_mib = torch.cuda.max_memory_allocated() / 2**20
    if float(model.object_branch.alpha.abs()) == 0:
        raise AssertionError("shared gate did not learn")
    buffer = io.BytesIO()
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all()}, buffer)
    update(model, optimizer)
    expected = {key: value.detach().clone() for key, value in model.state_dict().items()}
    buffer.seek(0)
    restored = torch.load(buffer, map_location=device, weights_only=False)
    model.load_state_dict(restored["model"], strict=True)
    optimizer.load_state_dict(restored["optimizer"])
    torch.set_rng_state(restored["rng"].cpu())
    torch.cuda.set_rng_state_all([value.cpu() for value in restored["cuda_rng"]])
    update(model, optimizer)
    resume_error = max(float((value - expected[key]).abs().max())
                       for key, value in model.state_dict().items())
    if resume_error > 1e-6:
        raise AssertionError(f"optimizer resume changes updates: {resume_error}")
    four = [i for i in range(len(training))
            if training.meta["offsets"][i+1] - training.meta["offsets"][i] == 512][:16]
    graph = collate_tensors([training[i] for i in four]).to(device)
    torch.cuda.reset_peak_memory_stats()
    worst_timings = []
    for _ in range(3):
        torch.cuda.synchronize()
        start = time.perf_counter()
        update(model, optimizer)
        torch.cuda.synchronize()
        worst_timings.append(time.perf_counter() - start)
    peak_mib = max(peak_mib, torch.cuda.max_memory_allocated() / 2**20)
    torch.cuda.synchronize()
    start = time.perf_counter()
    common.evaluate(model, validation, statistics, phi)
    torch.cuda.synchronize()
    validation_seconds = time.perf_counter() - start
    nominal_seconds = (float(np.median(timings[1:])) * 625 + validation_seconds) * EPOCHS
    report = {"status": "passed", "utc": common.now(), "parameters": count,
              "cache_checksums": checksums, "object_permutation_max_abs": permutation_error,
              "common_center_shift_max_abs": shift_error, "encoder_center_dependence_max_abs": center_error,
              "local_embedding_neighbor_dependence_max_abs": local_error, "shared_adapter_calls": len(calls),
              "N1_self_message": "zero", "N8_synthetic_forward": "finite",
              "optimizer_resume_max_abs": resume_error, "gate_after_smoke": float(model.object_branch.alpha.detach()),
              "step_seconds_median": float(np.median(timings[1:])),
              "N4_batch16_step_seconds_median": float(np.median(worst_timings)),
              "validation_seconds": validation_seconds, "peak_allocated_mib": peak_mib,
              "gpu_total_mib": torch.cuda.get_device_properties(0).total_memory / 2**20,
              "estimated_training_seconds_with_25pct_margin": nominal_seconds * 1.25,
              "new_storage_estimate_mib": 300}
    if peak_mib + 1024 > report["gpu_total_mib"]:
        raise RuntimeError("insufficient conservative GPU headroom")
    OUT.mkdir(parents=True, exist_ok=True)
    write_json(OUT / "preflight.json", report)
    inputs = [PARENT, common.OUT / "protocol.json", common.OUT / "preflight.json",
              common.OUT / "comparison.json", common.base.TRAIN / "manifest.json",
              common.base.VAL / "manifest.json", common.base.STATS,
              *[CACHE / folder / "manifest.json" for folder in ("train", *dict.fromkeys(f for f, _ in SCOPES.values()))]]
    frozen = {"schema": "v3-compositional-transolver-40ep-v1", "seed": 0, "parent_epoch": 100,
              "additional_epochs": EPOCHS, "absolute_epochs": [101, 140], "batch_size": 16,
              "learning_rate": 1e-4, "minimum_learning_rate": 1e-7, "weight_decay": 1e-5,
              "lambda_q": 0.1, "gradient_clip": 1.0, "coupling_after_blocks": [1, 2, 3],
              "shared_adapter": True, "local_attention_nodes": 128, "local_slices_per_head": 32,
              "interaction_slices_per_object_head": 8, "center_features_in_encoder": False,
              "selection": "validation mean scene scattered complex relative L2 only",
              "source_sha256": {str(p.relative_to(ROOT)): sha(p) for p in dict.fromkeys(SOURCES)},
              "input_sha256": {str(p.relative_to(ROOT)): sha(p) for p in inputs},
              "preflight_sha256": sha(OUT / "preflight.json"), "single_seed_exploratory": True,
              "test_counts": {key: count for key, (_, count) in SCOPES.items()}}
    write_json(OUT / "protocol.json", frozen)
    print(json.dumps(report), flush=True)


def train():
    frozen = verify_protocol()
    configure()
    training, validation = datasets()
    statistics, _ = common.statistics_and_scales()
    device = torch.device("cuda")
    phi = training.phi_pinv.to(device)
    model = parent_model(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-7)
    RUN.mkdir(parents=True, exist_ok=True)
    manifest = {"schema": "v3-compositional-transolver-run-v1", "seed": 0,
                "protocol_sha256": sha(OUT / "protocol.json"), "torch": torch.__version__,
                "parameters": sum(p.numel() for p in model.parameters())}
    manifest_path = RUN / "run_manifest.json"
    if manifest_path.exists() and common.read(manifest_path) != manifest:
        raise RuntimeError("resume run manifest differs")
    write_json(manifest_path, manifest)
    history, best, completed, cursor, collected, elapsed_before = [], float("inf"), 0, 0, {}, 0.0
    last = RUN / "last.pt"
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
        if len(history) != completed or scheduler.last_epoch != completed or state["absolute_epoch"] != 100 + completed:
            raise RuntimeError("checkpoint epoch/schedule/history inconsistent")
        if not 0 <= completed <= EPOCHS or not 0 <= cursor <= 625 or (completed == EPOCHS and cursor):
            raise RuntimeError("checkpoint cursor bounds inconsistent")
        if [row["epoch"] for row in history] != list(range(1, completed + 1)):
            raise RuntimeError("checkpoint history ordering inconsistent")
        if any(len(values) != cursor * 16 for values in collected.values()) or (cursor and not collected):
            raise RuntimeError("checkpoint partial scene metrics inconsistent")
        if completed:
            selected = min(history, key=lambda row: row["val"]["scattered_rel_l2"]["mean"])
            if not np.isclose(best, selected["val"]["scattered_rel_l2"]["mean"], atol=1e-12, rtol=0):
                raise RuntimeError("checkpoint best score inconsistent")
            best_path = RUN / "best.pt"
            saved_best = torch.load(best_path, map_location="cpu", weights_only=False) if best_path.exists() else None
            if saved_best is None or saved_best.get("epoch") != selected["epoch"]:
                if selected["epoch"] == completed and cursor == 0:
                    common.atomic_torch(best_path, state)
                else:
                    raise RuntimeError("historical best checkpoint cannot be reconstructed")
            elif saved_best["manifest"] != manifest:
                raise RuntimeError("best checkpoint manifest inconsistent")
            write_json(RUN / "history.json", history)
    started = time.perf_counter()

    def checkpoint():
        state = {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                 "manifest": manifest, "epoch": completed, "absolute_epoch": 100 + completed, "cursor": cursor,
                 "history": history, "partial_metrics": collected, "best_score": best,
                 "elapsed_seconds": elapsed_before + time.perf_counter() - started,
                 "rng": {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(),
                         "cuda": torch.cuda.get_rng_state_all()}}
        common.atomic_torch(last, state)
        return state

    if not last.exists():
        checkpoint()
    try:
        if not (RUN / "initial_validation.json").exists():
            initial, _, _ = common.evaluate(model, validation, statistics, phi, RUN / "status.json")
            write_json(RUN / "initial_validation.json", initial)
        for epoch in range(completed + 1, EPOCHS + 1):
            order = np.random.default_rng(20260930 + epoch).permutation(len(training))
            model.train()
            for step in range(cursor, 625):
                indices = order[step*16:(step+1)*16]
                graph = collate_tensors([training[int(i)] for i in indices]).to(device)
                loss, metrics = loss_and_scene_metrics(model(graph), graph, phi, statistics, 0.1)
                if not torch.isfinite(loss):
                    raise ArithmeticError("nonfinite training loss")
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
                optimizer.step()
                common.append_metrics(collected, metrics, loss, len(indices))
                cursor = step + 1
                if cursor % 25 == 0 or cursor == 625:
                    write_json(RUN / "status.json", {"status": "training", "epoch": epoch, "completed_epochs": completed,
                        "step": cursor, "steps": 625, "last_progress_utc": common.now(), "pid": os.getpid()})
                if cursor % 100 == 0:
                    checkpoint()
            val_metrics, _, _ = common.evaluate(model, validation, statistics, phi, RUN / "status.json")
            scheduler.step()
            row = {"epoch": epoch, "absolute_epoch": 100 + epoch, "train": common.summarize(collected),
                   "val": val_metrics, "learning_rate": scheduler.get_last_lr()[0],
                   "elapsed_seconds": elapsed_before + time.perf_counter() - started,
                   "adapter_gate": float(model.object_branch.alpha.detach())}
            improved = val_metrics["scattered_rel_l2"]["mean"] < best
            best = min(best, val_metrics["scattered_rel_l2"]["mean"])
            history.append(row)
            completed, cursor, collected = epoch, 0, {}
            state = checkpoint()
            if improved:
                common.atomic_torch(RUN / "best.pt", state)
            write_json(RUN / "history.json", history)
            print(json.dumps(row), flush=True)
        write_json(RUN / "completion.json", {"status": "complete", "additional_epochs": completed,
            "last_sha256": sha(last), "best_sha256": sha(RUN / "best.pt"), "utc": common.now(),
            "best_epoch": min(history, key=lambda row: row["val"]["scattered_rel_l2"]["mean"])["epoch"],
            "protocol_sha256": manifest["protocol_sha256"]})
        write_json(RUN / "status.json", {"status": "complete", "completed_epochs": completed,
            "last_progress_utc": common.now(), "pid": os.getpid()})
    except BaseException as error:
        write_json(RUN / "status.json", {"status": "failed", "completed_epochs": completed,
            "cursor": cursor, "error": str(error), "last_progress_utc": common.now(), "pid": os.getpid()})
        raise


def evaluate_and_compare():
    verify_protocol()
    configure()
    completion = common.read(RUN / "completion.json")
    if completion["additional_epochs"] != EPOCHS:
        raise RuntimeError("training incomplete")
    statistics, _ = common.statistics_and_scales()
    device = torch.device("cuda")
    model = CompositionalTransolver().to(device)
    report = {"status": "exploratory_single_seed", "checkpoints": {}}
    paired = {}
    for kind in ("last", "best"):
        path = RUN / f"{kind}.pt"
        if sha(path) != completion[f"{kind}_sha256"]:
            raise RuntimeError("completed checkpoint changed")
        state = torch.load(path, map_location=device, weights_only=False)
        if state["manifest"]["protocol_sha256"] != sha(OUT / "protocol.json"):
            raise RuntimeError("checkpoint protocol differs")
        model.load_state_dict(state["model"], strict=True)
        item = {"epoch": state["epoch"], "absolute_epoch": state["absolute_epoch"],
                "checkpoint_sha256": sha(path), "splits": {}}
        paired[kind] = {}
        for scope, (folder, count) in SCOPES.items():
            dataset = TensorSceneDataset(CACHE / folder)
            if len(dataset) != count:
                raise RuntimeError("evaluation scene count differs")
            sidecar = RUN / f"evaluation_{kind}_{scope}.json"
            arrays_path = sidecar.with_suffix(".npz")
            if sidecar.exists():
                saved = common.read(sidecar)
                if saved["checkpoint_sha256"] != item["checkpoint_sha256"] or not arrays_path.exists():
                    raise RuntimeError("existing evaluation identity differs")
                result = saved["result"]
            else:
                torch.cuda.synchronize()
                start = time.perf_counter()
                result, arrays, indices = common.evaluate(model, dataset, statistics, dataset.phi_pinv.to(device),
                                                          RUN / "evaluation_status.json", scope)
                torch.cuda.synchronize()
                result["evaluation_seconds"] = time.perf_counter() - start
                np.savez_compressed(arrays_path, sample_indices=np.asarray(indices),
                                    **{key: np.asarray(value) for key, value in arrays.items() if key != "losses"})
                write_json(sidecar, {"checkpoint_sha256": item["checkpoint_sha256"], "result": result,
                                    "tensor_manifest_sha256": sha(dataset.root / "manifest.json")})
            item["splits"][scope] = result
            paired[kind][scope] = {}
            with np.load(arrays_path) as candidate:
                ids, candidate_errors = candidate["sample_indices"], candidate["scattered_rel_l2"]
                for arm in ("baseline", "object-joint", "object-sum"):
                    with np.load(common.OUT / arm / f"evaluation_{kind}_{scope}.npz") as reference:
                        if not np.array_equal(ids, reference["sample_indices"]):
                            raise RuntimeError("paired scene IDs differ")
                        difference = candidate_errors - reference["scattered_rel_l2"]
                    paired[kind][scope][arm] = {"mean_difference": float(difference.mean()),
                        "median_difference": float(np.median(difference)),
                        "fraction_scenes_improved": float(np.mean(difference < 0))}
        report["checkpoints"][kind] = item
    write_json(RUN / "evaluation.json", report)
    old = common.read(common.OUT / "comparison.json")
    selected = report["checkpoints"]["best"]["splits"]
    reference = old["results"]["object-sum"]["checkpoints"]["best"]["splits"]
    screening = {"N8_relative_improvement": 1 - selected["n8_test"]["scattered_rel_l2"]["mean"] / reference["n8_test"]["scattered_rel_l2"]["mean"],
                 "ID_relative_change": selected["id_test"]["scattered_rel_l2"]["mean"] / reference["id_test"]["scattered_rel_l2"]["mean"] - 1,
                 "Peanut_relative_change": selected["shape_ood"]["scattered_rel_l2"]["mean"] / reference["shape_ood"]["scattered_rel_l2"]["mean"] - 1}
    screening["passed_descriptive_targets"] = screening["N8_relative_improvement"] >= 0.1 and screening["ID_relative_change"] <= 0.05 and screening["Peanut_relative_change"] <= 0.05
    write_json(OUT / "comparison.json", {"status": "complete", "single_seed_exploratory": True,
        "candidate": report, "historical_equal_training_budget_controls": old["results"],
        "paired_candidate_minus_reference": paired, "screening": screening,
        "limitations": ["three changes combined; not a component ablation", "no multi-seed evidence",
                        "previously inspected exploratory tests", "N8 not independently BEM-audited"]})
    write_json(OUT / "completion.json", {"status": "complete", "protocol_sha256": sha(OUT / "protocol.json"),
        "training_completion_sha256": sha(RUN / "completion.json"), "comparison_sha256": sha(OUT / "comparison.json"),
        "evaluation_sha256": sha(RUN / "evaluation.json"), "utc": common.now()})
    print(json.dumps({"status": "complete", "screening": screening, "selected_metrics": selected}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--action", choices=("preflight", "train", "evaluate", "run"), required=True)
    action = parser.parse_args().action
    if action == "preflight":
        preflight()
    elif action == "train":
        train()
    elif action == "evaluate":
        evaluate_and_compare()
    else:
        train()
        evaluate_and_compare()


if __name__ == "__main__":
    main()
