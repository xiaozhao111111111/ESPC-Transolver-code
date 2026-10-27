"""Portable C ABI for the authors' ball-tree partition algorithm.

Index types use intptr_t to support Windows LLP64 without changing the algorithm.
"""
from __future__ import annotations
import ctypes
import math
import os
from pathlib import Path
import sys
import types
from shared.point_reference_runtime import ROOT, np, torch

SOURCE = Path(__file__).resolve().parent / 'balltree'
LIBRARY = Path(os.environ.get('ESPC_BALLTREE_LIBRARY', str(
    ROOT / 'runtime/native/balltree_build' / ('balltree_cabi.dll' if os.name == 'nt' else 'balltree_cabi.so'))))


def build():
    """Build the C ABI without Cython or a Python SDK; requires a C++ compiler."""
    import shutil
    import subprocess
    directory = ROOT / 'runtime/native/balltree_build'
    directory.mkdir(parents=True, exist_ok=True)
    header = (SOURCE / 'balltree.h').read_text(encoding='utf-8')
    (directory / 'balltree.h').write_text(
        header.replace('using IndexType = long;', 'using IndexType = intptr_t;'), encoding='utf-8')
    compiler = shutil.which('clang++') or shutil.which('g++')
    source = Path(__file__).resolve().with_name('erwin_cabi.cpp')
    if compiler:
        command = [compiler, '-shared', '-O3', '-std=c++14', '-I'+str(directory),
                   str(source), '-o', str(LIBRARY)]
        if os.name != 'nt':
            command.insert(1, '-fPIC')
    elif os.name == 'nt' and shutil.which('cl'):
        command = ['cl', '/LD', '/O2', '/EHsc', '/std:c++14', '/I'+str(directory),
                   str(source), '/Fe:'+str(LIBRARY)]
    else:
        raise RuntimeError('A C++ compiler is required: use clang++, g++ or an MSVC developer shell')
    LIBRARY.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(command, cwd=directory, check=True)
    return LIBRARY


def install():
    if not LIBRARY.is_file():
        raise FileNotFoundError('Build the ball-tree binding with scripts/train_erwin.py --build-native')
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
