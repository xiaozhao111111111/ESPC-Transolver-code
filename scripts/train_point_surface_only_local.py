"""Run the existing field-only direct-surface protocol in the local Windows environment."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

OUT = ROOT / "outputs/point_surface_only_local_20261012"
SOURCE = ROOT / "outputs/point_surface_only_autodl_20261012_v4"
PROTOCOL = ROOT / "POINT_SURFACE_ONLY_LOCAL_20261012.md"
SUPERVISOR = ROOT / "scripts/supervise_point_surface_only_local.py"


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stage_readonly_receipts() -> None:
    """Copy only the two small, immutable source-bound receipts; never overwrite."""
    OUT.mkdir(parents=True, exist_ok=True)
    for name in ("transfer_inventory.json", "interface_qualification.json"):
        source = SOURCE / name
        target = OUT / name
        if not source.is_file():
            raise FileNotFoundError(f"Missing prepared field-only receipt: {source}")
        if target.exists():
            if sha(target) != sha(source):
                raise RuntimeError(f"Existing local receipt differs; preserve and inspect: {target}")
            continue
        with source.open("rb") as reader, target.open("xb") as writer:
            shutil.copyfileobj(reader, writer, length=1024 * 1024)
        if sha(target) != sha(source):
            raise RuntimeError(f"Copied receipt checksum mismatch: {name}")


def configure():
    stage_readonly_receipts()
    import remote_point_surface_only_v4 as field_only
    port = field_only.v3.v2.v1.port
    from erwin_windows_binding import install as install_windows_balltree

    install_windows_balltree()
    import balltree
    # The shared v4 source-bound adapter records the official extension path
    # under balltree.balltree. For this local run, bind that metadata to the
    # already-qualified Windows C ABI library while preserving its algorithms.
    from erwin_windows_binding import LIBRARY as WINDOWS_BALLTREE_LIBRARY
    import types
    balltree.balltree = types.SimpleNamespace(__file__=str(WINDOWS_BALLTREE_LIBRARY))
    # The remote adapter bootstraps Linux I/O and queries Linux wheel metadata.
    # Keep native Windows sharing semantics and report the local C ABI binding.
    port.install_io = lambda: None
    import importlib.metadata
    original_version = importlib.metadata.version
    def local_distribution_version(name: str) -> str:
        if name == "balltree-erwin":
            return "local-windows-cabi"
        return original_version(name)
    port.importlib = types.SimpleNamespace(
        metadata=types.SimpleNamespace(version=local_distribution_version))

    # Redirect only the source-derived adapter's output root to this new run.
    field_only.OUT = OUT
    surface = field_only.configure()
    surface.OUT = OUT
    surface.frozen.RUN_ROOT = OUT
    from prepare_point_reference_tests import sha as source_sha

    inherited_identity = surface.identity

    def local_identity(method: str):
        value = inherited_identity(method)
        branch = value["minimal_branch"]
        branch.update({
            "adapter_sha256": source_sha(Path(__file__)),
            "supervisor_sha256": source_sha(SUPERVISOR),
            "protocol_sha256": source_sha(PROTOCOL),
            "source_adapter_sha256": source_sha(ROOT / "scripts/remote_point_surface_only_v4.py"),
            "execution_target": "local Windows CUDA",
            "gpu_name": surface.torch.cuda.get_device_name(),
            "loss": "field only; lambda_q=0; no Z/q tensors",
            "windows_balltree_adapter_sha256": source_sha(ROOT / "scripts/erwin_windows_binding.py"),
            "windows_balltree_library_sha256": source_sha(WINDOWS_BALLTREE_LIBRARY),
        })
        value["execution_target"] = "local Windows CUDA"
        value["lambda_q"] = 0.0
        value["physical_Z_loaded"] = False
        value["projected_flux_metric_available"] = False
        return json.loads(json.dumps(value))

    surface.identity = local_identity
    surface.frozen.identity = local_identity
    # Recompile the already-qualified source-derived preflight against the local
    # identity so train-time checkpoint/source binding exactly matches it.
    surface.preflight = field_only.v3.v2.v1.adapted_function(
        surface, "preflight", 2, strip_flux_backup=True)
    surface.frozen.LAMBDA_Q = 0.0
    return surface, field_only


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=("ptv3", "erwin"), required=True)
    phase = parser.add_mutually_exclusive_group(required=True)
    phase.add_argument("--preflight", action="store_true")
    phase.add_argument("--train", action="store_true")
    args = parser.parse_args()
    if not PROTOCOL.is_file() or not SUPERVISOR.is_file():
        raise RuntimeError("Local protocol/supervisor source is incomplete")
    surface, field_only = configure()
    try:
        checked_files = field_only.verify_transfer()
        if args.preflight:
            # Qualification must run under the same resident-mmap policy as
            # formal training; otherwise the Erwin 200-batch soak can let the
            # Windows process working set grow past the host RAM reserve.
            surface.bind_resident_cache(args.method)
            surface.require_ram()
            surface.preflight(args.method)
            field_only.v3.v2.qualify_extra(surface, args.method)
        else:
            receipt = surface.read_json(OUT / f"{args.method}_preflight.json")
            if not receipt or receipt.get("status") != "passed" or receipt.get("identity") != surface.identity(args.method):
                raise RuntimeError("Missing matching local GPU preflight; refuse training")
            memory_receipt = surface.read_json(OUT / f"{args.method}_memory_qualification.json")
            if (not memory_receipt or memory_receipt.get("status") != "passed"
                    or memory_receipt.get("identity") != surface.identity(args.method)
                    or memory_receipt.get("batches") != 200
                    or not isinstance(memory_receipt.get("peak_gpu_allocated_bytes"), int)):
                raise RuntimeError("Missing matching 200-batch GPU memory qualification; refuse training")
            # train_point_surface.train uses the qualified Windows lock and
            # worker-only resident cache policy, then the unchanged frozen loop.
            surface.train(args.method)
        print(json.dumps({"event": "phase_complete", "method": args.method,
                          "phase": "preflight" if args.preflight else "train",
                          "inventory_files_verified": checked_files,
                          "identity": surface.identity(args.method)}, default=str), flush=True)
    except BaseException as error:
        failure = OUT / f"{args.method}_{'preflight' if args.preflight else 'train'}_failure_{time.time_ns()}.json"
        surface.write_json(failure, {"method": args.method, "phase": "preflight" if args.preflight else "train",
            "pid": os.getpid(), "error": repr(error), "traceback": traceback.format_exc(),
            "identity": surface.identity(args.method)})
        raise


if __name__ == "__main__":
    main()
