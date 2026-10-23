"""Execution-only CUDA cache recovery for the immutable local surface Erwin run."""
from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
OUT = ROOT / "outputs/point_surface_only_local_20261012"
ATTEMPT = OUT / "erwin_cache_recovery_20261013"
INTERVAL = 10
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    from point_reference_io_recovery import write_json
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    write_json(path, value)


def read(path):
    from point_reference_io_recovery import read_json
    return read_json(path)


def process_record(pid):
    import psutil
    p = psutil.Process(pid)
    return {"pid": pid, "created_at": p.create_time(), "command": p.cmdline()}


def refuse_other_workers():
    import psutil
    for p in psutil.process_iter(["pid", "name", "cmdline"]):
        if p.pid == os.getpid() or (p.info["name"] or "").lower() not in ("python.exe", "pythonw.exe"):
            continue
        cmd = " ".join(p.info["cmdline"] or [])
        if "train_point_surface_only_local.py" in cmd or (
            Path(__file__).name in cmd and ("--train" in cmd or "--qualify" in cmd)
        ):
            raise RuntimeError(f"A surface worker is already active: {process_record(p.pid)}")


def configure():
    import train_point_surface_only_local as local
    s, field_only = local.configure()
    field_only.verify_transfer()
    expected = s.identity("erwin")
    for name in ("erwin_preflight.json", "erwin_memory_qualification.json"):
        r = read(OUT / name)
        if not r or r.get("status") != "passed" or r.get("identity") != expected:
            raise RuntimeError(f"Original source/data qualification mismatch: {name}")
    if read(OUT / "erwin_memory_qualification.json")["batches"] != 200:
        raise RuntimeError("Original variable-size qualification is incomplete")
    return s


def binding(s):
    return {"adapter_sha256": sha(__file__), "original_identity": s.identity("erwin"),
            "cache_interval_updates": INTERVAL, "allocator": os.environ["PYTORCH_ALLOC_CONF"],
            "change": "release unused CUDA cache after each 10 AdamW updates, every 10 batch transfers (including evaluation), and before RAM telemetry"}


def install_cleanup(s):
    """Release only unused allocator blocks, after the identical AdamW update."""
    original_step = s.torch.optim.AdamW.step
    original_require = s.require_ram
    from point_surface_only_data import SurfaceBatch
    original_to = SurfaceBatch.to
    transfers = 0

    def step(optimizer, *args, **kwargs):
        result = original_step(optimizer, *args, **kwargs)
        count = getattr(optimizer, "_surface_cache_updates", 0) + 1
        optimizer._surface_cache_updates = count
        if count % INTERVAL == 0:
            s.torch.cuda.empty_cache()
        return result

    def require():
        s.torch.cuda.empty_cache()
        return original_require()

    def to(batch, *args, **kwargs):
        nonlocal transfers
        transfers += 1
        if transfers % INTERVAL == 0:
            s.torch.cuda.empty_cache()
        return original_to(batch, *args, **kwargs)

    s.torch.optim.AdamW.step = step
    SurfaceBatch.to = to
    s.require_ram = require


def cpu_copy(value, torch):
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: cpu_copy(v, torch) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(cpu_copy(v, torch) for v in value)
    return copy.deepcopy(value)


def max_difference(a, b, torch):
    if torch.is_tensor(a):
        if not torch.is_tensor(b) or a.shape != b.shape or a.dtype != b.dtype:
            raise RuntimeError("Tensor structure changed")
        if a.dtype.is_floating_point or a.dtype.is_complex:
            return float((a - b.cpu()).abs().max()) if a.numel() else 0.
        if not torch.equal(a, b.cpu()):
            raise RuntimeError("Integer/RNG state changed")
        return 0.
    if isinstance(a, dict):
        if a.keys() != b.keys():
            raise RuntimeError("State keys changed")
        return max((max_difference(a[k], b[k], torch) for k in a), default=0.)
    if isinstance(a, (list, tuple)):
        if len(a) != len(b):
            raise RuntimeError("State length changed")
        return max((max_difference(x, y, torch) for x, y in zip(a, b)), default=0.)
    if a != b:
        raise RuntimeError(f"Scalar state changed: {a!r} / {b!r}")
    return 0.


def exercise(s, batches, reference=None):
    from torch.utils.data import DataLoader
    torch = s.torch
    s.frozen.seed_all()
    model = s.SurfaceReference("erwin").cuda()
    stats, phi, ds, _ = s.frozen.setup_data("erwin")
    generator = torch.Generator().manual_seed(0)
    loader = DataLoader(ds, batch_size=16, shuffle=True, generator=generator, collate_fn=s.collate)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    model.train()
    torch.cuda.reset_peak_memory_stats()
    trace, snapshots = [], []
    errors = {"inputs": 0., "outputs": 0., "loss": 0., "gradients": 0., "model_optimizer_rng": 0.}
    start = time.perf_counter()
    for i, batch in enumerate(loader, 1):
        if i > batches:
            break
        cpu_input = cpu_copy(vars(batch), torch) if i <= 8 else None
        batch = batch.to("cuda")
        optimizer.zero_grad(set_to_none=True)
        prediction = model(batch)
        loss, _ = s.frozen.LOSS(prediction, batch, phi, stats, 0.)
        if not torch.isfinite(loss):
            raise RuntimeError("Nonfinite qualification loss")
        loss.backward()
        if i <= 8:
            item = {"inputs": cpu_input, "outputs": prediction.detach().cpu().clone(),
                    "loss": float(loss.detach()),
                    "gradients": [p.grad.detach().cpu().clone() for p in model.parameters() if p.grad is not None]}
            if reference is None:
                trace.append(item)
            else:
                for k in ("inputs", "outputs", "gradients"):
                    errors[k] = max(errors[k], max_difference(reference["trace"][i-1][k], item[k], torch))
                errors["loss"] = max(errors["loss"], abs(reference["trace"][i-1]["loss"]-item["loss"]))
            del item, cpu_input
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
        if i == 8:
            state = cpu_copy({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                              "torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all(),
                              "loader": generator.get_state()}, torch)
            if reference is not None:
                errors["model_optimizer_rng"] = max_difference(reference["state"], state, torch)
                del state
        del prediction, loss, batch
        if i % 10 == 0:
            torch.cuda.synchronize()
            snapshot = {"step": i, "seconds": time.perf_counter()-start, **s.require_ram()}
            snapshots.append(snapshot)
            print(json.dumps({"event": "cache_recovery_soak", **snapshot}), flush=True)
    seconds = time.perf_counter()-start
    peak = torch.cuda.max_memory_allocated()
    del model, optimizer, loader, ds, stats, phi
    gc.collect(); torch.cuda.empty_cache()
    result = {"seconds": seconds, "batches": batches, "peak_gpu_allocated_bytes": peak,
              "snapshots": snapshots, "errors": errors}
    if reference is None:
        result.update(trace=trace, state=state)
    return result


def qualify(s):
    baseline = exercise(s, 8)
    install_cleanup(s)
    candidate = exercise(s, 200, baseline)
    if max(candidate["errors"].values()) > 2e-5:
        raise RuntimeError(f"Cache recovery numerical parity failed: {candidate['errors']}")
    if len(candidate["snapshots"]) != 20:
        raise RuntimeError("Incomplete 200-batch soak")
    write(ATTEMPT / "qualification.json", {"status": "passed", "binding": binding(s),
          "actual_training_loader_seed": 0, "batch_size": 16, "baseline_batches": 8,
          "candidate": candidate, "passed_at": time.time()})
    print(json.dumps({"event": "cache_recovery_qualified", "errors": candidate["errors"],
                      "seconds": candidate["seconds"]}), flush=True)


def train(s):
    r = read(ATTEMPT / "qualification.json")
    if not r or r.get("status") != "passed" or r.get("binding") != binding(s):
        raise RuntimeError("Missing matching cache recovery qualification")
    install_cleanup(s)
    # Retain the original identity and full frozen train/evaluate/checkpoint loop.
    s.train("erwin")


def supervise():
    from supervise_point_surface_only_local import identity, terminate_owned_worker
    import msvcrt
    import psutil
    ATTEMPT.mkdir(parents=True, exist_ok=True)
    with (ATTEMPT / "supervisor.lock").open("a+b") as lock:
        lock.seek(0); msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        refuse_other_workers()
        if not (ATTEMPT / "previous_erwin_state.json").exists():
            write(ATTEMPT / "previous_erwin_state.json", {
                "status": read(OUT / "runs/erwin/status.json"),
                "manifest": read(OUT / "runs/erwin/run_manifest.json"),
                "last_checkpoint_exists": (OUT / "runs/erwin/last.pt").exists(),
                "preserved_at": time.time()})
        record = process_record(os.getpid())
        record.update(status="running", source_sha256=sha(__file__), started_at=time.time())
        write(ATTEMPT / "supervisor.json", record)
        try:
            phases = ["qualify", "train"]
            if (read(ATTEMPT / "qualification.json") or {}).get("status") == "passed":
                phases = ["train"]  # Worker checks the exact source and original identity again.
            for phase in phases:
                stamp = time.time_ns()
                cmd = [sys.executable, "-u", str(Path(__file__).resolve()), "--"+phase]
                stdout = ATTEMPT / f"{phase}_{stamp}.stdout.log"
                stderr = ATTEMPT / f"{phase}_{stamp}.stderr.log"
                with stdout.open("xb") as out, stderr.open("xb") as err:
                    proc = subprocess.Popen(cmd, cwd=ROOT, stdin=subprocess.DEVNULL,
                        stdout=out, stderr=err, creationflags=subprocess.CREATE_NO_WINDOW)
                worker = identity(proc, cmd)
                worker.update(phase=phase, stdout=str(stdout.relative_to(ROOT)), stderr=str(stderr.relative_to(ROOT)))
                record.update(phase=phase, worker=worker, updated_at=time.time())
                write(ATTEMPT / f"{phase}_process.json", worker)
                write(ATTEMPT / "supervisor.json", record)
                print(json.dumps({"event": "recovery_phase_started", **worker}), flush=True)
                try:
                    while proc.poll() is None:
                        p = psutil.Process(proc.pid)
                        if abs(p.create_time()-worker["created_at"]) > .1 or p.cmdline() != cmd:
                            raise RuntimeError("Worker process identity mismatch")
                        available = psutil.virtual_memory().available
                        write(ATTEMPT / "resources.json", {"phase": phase, "pid": proc.pid,
                              "available_ram_bytes": available, "rss_bytes": p.memory_info().rss,
                              "updated_at": time.time()})
                        if available < 1024**3:
                            raise RuntimeError(f"System RAM reserve exhausted: {available}")
                        time.sleep(5)
                except BaseException:
                    terminate_owned_worker(proc, worker)
                    raise
                code = proc.wait()
                write(ATTEMPT / f"{phase}_exit_{stamp}.json", {**worker, "return_code": code, "ended_at": time.time()})
                if code:
                    raise RuntimeError(f"{phase} worker exited {code}; read its attempt-specific stderr")
            record.update(status="complete", completed_at=time.time())
            write(ATTEMPT / "supervisor.json", record)
        except BaseException as e:
            record.update(status="failed", error=repr(e), traceback=traceback.format_exc(), updated_at=time.time())
            write(ATTEMPT / "supervisor.json", record)
            raise


def main():
    parser = argparse.ArgumentParser()
    phase = parser.add_mutually_exclusive_group(required=True)
    phase.add_argument("--qualify", action="store_true")
    phase.add_argument("--train", action="store_true")
    phase.add_argument("--supervise", action="store_true")
    args = parser.parse_args()
    ATTEMPT.mkdir(parents=True, exist_ok=True)
    if args.supervise:
        supervise(); return
    refuse_other_workers()
    try:
        s = configure()
        s.bind_resident_cache("erwin")
        if args.qualify:
            qualify(s)
        else:
            train(s)
    except BaseException as e:
        write(ATTEMPT / f"failure_{time.time_ns()}.json", {"worker": process_record(os.getpid()),
              "error": repr(e), "traceback": traceback.format_exc()})
        raise


if __name__ == "__main__":
    main()
