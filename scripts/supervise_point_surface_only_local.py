"""Sequential, identity-checked local supervisor for field-only surface PTv3/Erwin."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path(sys.executable).resolve()
WORKER = ROOT / "scripts/train_point_surface_only_local.py"
OUT = ROOT / "outputs/point_surface_only_local_20261012"


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".{os.getpid()}.{time.time_ns()}.pending")
    with temp.open("x", encoding="utf-8", newline="\n") as f:
        json.dump(value, f, indent=2, sort_keys=True)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(temp, path)


def read_json(path: Path):
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return None


def identity(process: subprocess.Popen, command: list[str]) -> dict:
    import psutil
    proc = psutil.Process(process.pid)
    return {"pid": process.pid, "created_at": proc.create_time(), "command": command,
            "stdout": None, "stderr": None}


def in_windows_job() -> bool:
    import ctypes
    from ctypes import wintypes
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.GetCurrentProcess.restype = wintypes.HANDLE
    kernel.IsProcessInJob.argtypes = [wintypes.HANDLE, wintypes.HANDLE, ctypes.POINTER(wintypes.BOOL)]
    kernel.IsProcessInJob.restype = wintypes.BOOL
    value = wintypes.BOOL()
    if not kernel.IsProcessInJob(kernel.GetCurrentProcess(), None, ctypes.byref(value)):
        raise ctypes.WinError(ctypes.get_last_error())
    return bool(value.value)


def terminate_owned_worker(proc: subprocess.Popen, record: dict) -> bool:
    """Stop only the exact child launched by this supervisor."""
    import psutil
    try:
        observed = psutil.Process(proc.pid)
        if (abs(observed.create_time() - record["created_at"]) > 0.1
                or observed.cmdline() != record["command"]):
            return False
        proc.terminate()
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            observed = psutil.Process(proc.pid)
            if (abs(observed.create_time() - record["created_at"]) > 0.1
                    or observed.cmdline() != record["command"]):
                return False
            proc.kill()
            proc.wait(timeout=10)
        return True
    except psutil.NoSuchProcess:
        return True


def run_phase(method: str, phase: str) -> int:
    import psutil
    run_dir = OUT / "supervisor"
    run_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.time_ns()
    stdout_path = run_dir / f"{method}_{phase}_{stamp}.stdout.log"
    stderr_path = run_dir / f"{method}_{phase}_{stamp}.stderr.log"
    command = [str(PYTHON), "-u", str(WORKER), "--method", method,
               "--preflight" if phase == "preflight" else "--train"]
    with stdout_path.open("xb") as stdout, stderr_path.open("xb") as stderr:
        proc = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL,
            stdout=stdout, stderr=stderr, creationflags=subprocess.CREATE_NO_WINDOW,
            env={**os.environ, "PYTORCH_ALLOC_CONF": "expandable_segments:True"})
    record = identity(proc, command)
    record.update({"stdout": str(stdout_path.relative_to(ROOT)),
                   "stderr": str(stderr_path.relative_to(ROOT)), "method": method, "phase": phase})
    write_json(run_dir / f"{method}_{phase}_process.json", record)
    write_json(run_dir / "supervisor.json", {"status": "running", "phase": phase,
        "method": method, "worker": record, "updated_at": time.time()})
    try:
        while proc.poll() is None:
            observed = psutil.Process(proc.pid)
            if abs(observed.create_time() - record["created_at"]) > 0.1 or observed.cmdline() != command:
                raise RuntimeError("Worker PID identity changed; refuse further supervision")
            available = psutil.virtual_memory().available
            write_json(run_dir / "resource_status.json", {"updated_at": time.time(),
                "available_ram_bytes": available, "worker_rss_bytes": observed.memory_info().rss,
                "phase": phase, "method": method, "pid": proc.pid})
            if available < 1024**3:
                raise RuntimeError(f"Available system RAM below 1 GiB reserve ({available} bytes)")
            time.sleep(10)
    except BaseException:
        # A supervisor-side guard/error must not leave its known GPU worker
        # consuming resources without supervision. Never touch a mismatched PID.
        terminate_owned_worker(proc, record)
        raise
    return_code = proc.wait()
    write_json(run_dir / f"{method}_{phase}_exit_{stamp}.json", {
        **record, "return_code": return_code, "ended_at": time.time()})
    if return_code:
        write_json(run_dir / "supervisor.json", {"status": "failed", "phase": phase,
            "method": method, "worker": record, "return_code": return_code,
            "updated_at": time.time()})
        raise RuntimeError(f"{method} {phase} exited {return_code}; inspect {stderr_path}")
    return return_code


def reuse_preflight_if_present(method: str) -> bool:
    """Reuse only local passed receipts; the training worker rechecks full identity."""
    preflight_path = OUT / f"{method}_preflight.json"
    memory_path = OUT / f"{method}_memory_qualification.json"
    preflight = read_json(preflight_path)
    memory = read_json(memory_path)
    if (not preflight or preflight.get("status") != "passed"
            or not memory or memory.get("status") != "passed"
            or memory.get("batches") != 200
            or not isinstance(memory.get("peak_gpu_allocated_bytes"), int)):
        return False
    receipt = {
        "method": method,
        "status": "passed_reused",
        "preflight_receipt": str(preflight_path.relative_to(ROOT)),
        "preflight_sha256": sha(preflight_path),
        "memory_qualification_receipt": str(memory_path.relative_to(ROOT)),
        "memory_qualification_sha256": sha(memory_path),
        "batches": memory["batches"],
        "peak_gpu_allocated_bytes": memory["peak_gpu_allocated_bytes"],
        "updated_at": time.time(),
        "identity_verification": "rechecked by the training worker before any optimizer update",
    }
    write_json(OUT / "supervisor" / f"{method}_preflight_reused.json", receipt)
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--child", action="store_true", help="reserved for detached launchers")
    parser.add_argument("--foreground", action="store_true",
                        help="allow supervised execution inside the current Windows job")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    lock_path = OUT / "supervisor.lock"
    with lock_path.open("a+b") as lock:
        import msvcrt
        lock.seek(0)
        try:
            msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError as exc:
            raise RuntimeError("A local surface supervisor already holds the lock") from exc
        attached_to_job = in_windows_job()
        if attached_to_job and not args.foreground:
            raise RuntimeError("Supervisor is attached to a host job; refuse multi-hour training")
        supervisor = {"pid": os.getpid(), "created_at": __import__("psutil").Process().create_time(),
            "command": [str(PYTHON), "-u", str(Path(__file__).resolve())]
                        + (["--foreground"] if args.foreground else []),
            "source_sha256": sha(Path(__file__)), "worker_sha256": sha(WORKER),
            "status": "running", "in_windows_job": attached_to_job,
            "protocol": str((ROOT / "POINT_SURFACE_ONLY_LOCAL_20261012.md").relative_to(ROOT)),
            "started_at": time.time(), "methods": ["ptv3", "erwin"], "sequential": True}
        write_json(OUT / "supervisor.json", supervisor)
        try:
            for method in ("ptv3", "erwin"):
                if not reuse_preflight_if_present(method):
                    run_phase(method, "preflight")
                run_phase(method, "train")
            supervisor.update({"status": "complete", "completed_at": time.time()})
            write_json(OUT / "supervisor.json", supervisor)
        except BaseException as exc:
            supervisor.update({"status": "failed", "error": repr(exc), "updated_at": time.time()})
            write_json(OUT / "supervisor.json", supervisor)
            raise
        finally:
            lock.seek(0)
            msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)


if __name__ == "__main__":
    main()
