"""Windows I/O-only recovery; the frozen numerical trainer remains unchanged.

Retry only Windows sharing/access errors, for a bounded time. Permanent errors
remain fatal and pending evidence is retained. Monitoring handles share delete
and close before JSON parsing. This host can still refuse MoveFileEx while a
reader is open, even with share-delete, so bounded retries remain necessary.
"""
from __future__ import annotations
import ctypes
from ctypes import wintypes
import json
import msvcrt
import os
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/point_references_20k_20261010'
ATTEMPT = OUT / 'recovery_io_20261011'
RETRY_SECONDS = 5.0


def replace_retry(pending, destination, *, timeout=RETRY_SECONDS, on_retry=None):
    pending, destination = Path(pending), Path(destination)
    started, retries = time.monotonic(), 0
    while True:
        try:
            os.replace(pending, destination)
            if retries:
                print(json.dumps({'event': 'atomic_replace_recovered', 'path': str(destination),
                                  'retries': retries, 'seconds': time.monotonic() - started}), flush=True)
            return
        except OSError as error:
            elapsed = time.monotonic() - started
            if os.name != 'nt' or getattr(error, 'winerror', None) not in (5, 32, 33) or elapsed >= timeout:
                raise
            retries += 1
            if on_retry is not None:
                on_retry(retries, error)
            time.sleep(min(.025 * 2**min(retries - 1, 4), .25, max(0., timeout - elapsed)))


def write_json(path, value):
    path = Path(path)
    pending = path.with_suffix('.pending')
    # Single identity-checked writer per destination; same serialization as the
    # original helper. Close the file before attempting the atomic replacement.
    pending.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')
    replace_retry(pending, path)


def save_checkpoint(path, payload):
    from point_reference_runtime import torch
    path = Path(path)
    pending = path.with_suffix('.pending')
    # Preserve the original torch.save payload, pending filename and semantics.
    torch.save(payload, pending)
    replace_retry(pending, path)


def open_shared(path):
    """A read-only file object sharing READ, WRITE and DELETE on Windows."""
    if os.name != 'nt':
        return Path(path).open('rb')
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.CreateFileW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD,
                                  wintypes.LPVOID, wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE]
    kernel.CreateFileW.restype = wintypes.HANDLE
    handle = kernel.CreateFileW(str(Path(path).resolve()), 0x80000000, 7, None, 3, 128, None)
    if handle == ctypes.c_void_p(-1).value:
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        descriptor = msvcrt.open_osfhandle(handle, os.O_RDONLY | os.O_BINARY)
    except BaseException:
        kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        kernel.CloseHandle(handle)
        raise
    return os.fdopen(descriptor, 'rb')


def read_json(path):
    try:
        with open_shared(path) as stream:
            payload = stream.read()
    except FileNotFoundError:
        return None
    # Parsing a large history must not keep its Windows handle open.
    return json.loads(payload)


def bound_current_worker():
    """Reapply the previously qualified 4.5-GiB resident-cache policy only here."""
    import psutil
    p = psutil.Process()
    command = ' '.join(p.cmdline())
    if 'resume_train_point_references_io' not in command or "'erwin'" not in command:
        raise RuntimeError('Refuse memory policy for an unrelated process')
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.GetCurrentProcess.restype = wintypes.HANDLE
    kernel.GetProcessWorkingSetSizeEx.argtypes = [wintypes.HANDLE, ctypes.POINTER(ctypes.c_size_t),
                                                ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(wintypes.DWORD)]
    kernel.GetProcessWorkingSetSizeEx.restype = wintypes.BOOL
    kernel.SetProcessWorkingSetSizeEx.argtypes = [wintypes.HANDLE, ctypes.c_size_t, ctypes.c_size_t, wintypes.DWORD]
    kernel.SetProcessWorkingSetSizeEx.restype = wintypes.BOOL
    handle = kernel.GetCurrentProcess()
    minimum, maximum, flags = ctypes.c_size_t(), ctypes.c_size_t(), wintypes.DWORD()
    if not kernel.GetProcessWorkingSetSizeEx(handle, ctypes.byref(minimum), ctypes.byref(maximum), ctypes.byref(flags)):
        raise ctypes.WinError(ctypes.get_last_error())
    cap = int(4.5 * 1024**3)
    before = {'minimum': minimum.value, 'maximum': maximum.value, 'flags': flags.value,
              'rss_bytes': p.memory_info().rss}
    if not kernel.SetProcessWorkingSetSizeEx(handle, 32 * 1024**2, cap, 0x4 | 0x2):
        raise ctypes.WinError(ctypes.get_last_error())
    value = {'policy': 'resident working-set only; not a virtual/tensor/GPU memory cap',
             'pid': p.pid, 'created_at': p.create_time(), 'before': before,
             'resident_max_bytes': cap, 'after_rss_bytes': p.memory_info().rss,
             'updated_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}
    write_json(ATTEMPT / 'working_set_policy.json', value)
    return value
