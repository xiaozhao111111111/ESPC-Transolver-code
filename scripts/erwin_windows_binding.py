"""Windows C ABI for the authors' unchanged ball-tree partition algorithm.

Avoids the failing CPython/Cython/MinGW ABI. The only algorithm-source change
is long -> intptr_t (Windows LLP64). Rotation/coarsening wrapper statements
below are executed verbatim from the downloaded official pyx, with hashes.
"""
from __future__ import annotations
import ctypes
import math
from pathlib import Path
import sys
import types
from point_reference_runtime import ROOT, np, torch

SOURCE = ROOT / 'work/point_reference_packages_20261010/balltree_erwin-0.1.0/balltree'
LIBRARY = ROOT / 'work/point_reference_packages_20261010/balltree_windows_build/balltree_cabi.dll'


def install():
    library = ctypes.CDLL(str(LIBRARY))
    nd = np.ctypeslib.ndpointer
    doubles = nd(dtype=np.float64, flags='C_CONTIGUOUS')
    integers = nd(dtype=np.int64, flags='C_CONTIGUOUS')
    masks = nd(dtype=np.bool_, flags='C_CONTIGUOUS')
    library.bt_build.argtypes = [doubles, integers, integers, masks, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64]
    library.bt_partition.argtypes = [doubles, integers, integers, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64]
    library.bt_build.restype = library.bt_partition.restype = None

    def build_balltree(data, batch_idx):
        xyz = np.ascontiguousarray(data.detach().cpu().double().numpy())
        batch = batch_idx.detach().cpu().numpy()
        offsets = np.r_[0, np.flatnonzero(np.diff(batch)) + 1, len(batch)]
        results, mask_results = [], []
        for lo, hi in zip(offsets[:-1], offsets[1:]):
            n = int(hi - lo)
            depth = math.ceil(math.log2(n)) - 1
            size = 2 ** (depth + 1)
            idx = np.arange(n, dtype=np.int64)
            out = np.empty(size, dtype=np.int64)
            mask = np.empty(size, dtype=np.bool_)
            library.bt_build(xyz[lo:hi], idx, out, mask, n, xyz.shape[1], depth)
            results.append(out + lo)
            mask_results.append(mask)
        return (torch.from_numpy(np.concatenate(results)).to(data.device),
                torch.from_numpy(np.concatenate(mask_results)).to(data.device))

    def partition_balltree(data, batch_idx, target_level):
        xyz = np.ascontiguousarray(data.detach().cpu().double().numpy())
        batch = batch_idx.detach().cpu().numpy()
        offsets = np.r_[0, np.flatnonzero(np.diff(batch)) + 1, len(batch)]
        results = []
        for lo, hi in zip(offsets[:-1], offsets[1:]):
            n = int(hi - lo)
            idx = np.arange(n, dtype=np.int64)
            out = np.empty(n, dtype=np.int64)
            library.bt_partition(xyz[lo:hi], idx, out, n, xyz.shape[1], target_level)
            results.append(out + lo)
        return torch.from_numpy(np.concatenate(results)).to(data.device)

    module = types.ModuleType('balltree')
    module.__file__ = __file__
    module.__dict__.update(math=math, torch=torch, build_balltree=build_balltree,
                           partition_balltree=partition_balltree)
    text = (SOURCE / 'balltree.pyx').read_text(encoding='utf-8')
    text = text[text.index('def generate_rotation_matrix'):]
    exec(compile(text, str(SOURCE / 'balltree.pyx'), 'exec'), module.__dict__)
    sys.modules['balltree'] = module
