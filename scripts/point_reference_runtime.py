"""Isolated official PTv3/Erwin runtime; never changes global Python packages."""
from __future__ import annotations
import ast
import importlib.util
import math
import os
from pathlib import Path
import sys
import types

ROOT = Path(__file__).resolve().parents[1]
DEPS = ROOT / 'work/point_reference_deps_20261010'
TORCH_RUNTIME = Path('D:/paper_等效散射方法/local_runtimes/torch280_torus_20261010/packages')
os.environ.setdefault('OMP_NUM_THREADS', '4')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
sys.path[:0] = [str(TORCH_RUNTIME), str(DEPS), str(ROOT / 'scripts')]
import numpy as np
import torch


def leaf(path, names, namespace):
    """Execute exact existing definitions without their unused heavy imports."""
    tree = ast.parse(Path(path).read_text(encoding='utf-8'), str(path))
    nodes = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names]
    if {n.name for n in nodes} != set(names):
        raise ImportError(f'Leaf definitions changed: {path}')
    body = ast.Module(body=[ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0)] + nodes, type_ignores=[])
    exec(compile(ast.fix_missing_locations(body), str(path), 'exec'), namespace)
    return types.SimpleNamespace(**namespace)


def official_imports():
    # Use the pinned official DropPath, not the full torchvision-dependent timm.
    drop = ROOT / 'work/subset11_train_deps/timm/layers/drop.py'
    ns = {'torch': torch, 'nn': torch.nn}
    leaf(drop, ['drop_path', 'DropPath'], ns)
    for name in ('timm', 'timm.models', 'timm.models.layers'):
        module = types.ModuleType(name)
        module.__path__ = []
        module.__file__ = str(drop)
        module.DropPath = ns['DropPath']
        sys.modules[name] = module
    # Erwin's official mp_steps=0 configuration never uses this optional API.
    # Fail closed if a future configuration accidentally enables it.
    if 'torch_cluster' not in sys.modules and importlib.util.find_spec('torch_cluster') is None:
        module = types.ModuleType('torch_cluster')
        def unavailable(*args, **kwargs):
            raise RuntimeError('Optional torch_cluster is unavailable; mp_steps must remain zero')
        module.radius_graph = unavailable
        sys.modules['torch_cluster'] = module


def load_official_ptv3():
    official_imports()
    directory = ROOT / 'third_party/PointTransformerV3'
    name = '_official_ptv3_reference'
    if name not in sys.modules:
        package = types.ModuleType(name)
        package.__path__ = [str(directory)]
        sys.modules[name] = package
        spec = importlib.util.spec_from_file_location(name + '.model', directory / 'model.py')
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
    return sys.modules[name + '.model'].PointTransformerV3


def load_official_erwin():
    official_imports()
    from erwin_windows_binding import install
    if 'balltree' not in sys.modules:
        install()
    name = '_official_erwin_reference'
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, ROOT / 'third_party/Erwin/models/erwin.py')
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name].ErwinTransformer


def original_loss():
    ns = {'torch': torch}
    return leaf(ROOT / 'src/cspgnet/v3_learning.py',
                ['_complex', '_reduce', 'projected_flux_difference', 'loss_and_scene_metrics'], ns).loss_and_scene_metrics


def original_response():
    from scipy.special import sph_harm_y, spherical_jn, spherical_yn
    formal = leaf(ROOT / 'src/cspgnet/data_generation/formal_neumann_3d.py',
                  ['spherical_modes', 'spherical_harmonic_matrix', 'equal_area_weights', 'weighted_pseudoinverse'],
                  {'np': np, 'math': math, '_spherical_harmonic': lambda l, m, p, a: sph_harm_y(l, m, p, a)})
    formal.DEFAULT_K = 2.0
    formal.DEFAULT_CANONICAL_RADIUS = 1.5
    tmatrix = leaf(ROOT / 'src/cspgnet/data_generation/tmatrix_neumann_3d.py',
                   ['radial_values', 'transition_matrix'],
                   {'np': np, 'formal': formal, 'spherical_jn': spherical_jn, 'spherical_yn': spherical_yn})
    reduced = leaf(ROOT / 'src/cspgnet/data_generation/reduced_z64_solver.py',
                   ['modal_dtn_from_stored_z'], {'np': np, 'formal': formal})
    response = leaf(ROOT / 'src/cspgnet/data_generation/structural_response.py', ['scattering_response'],
                    {'np': np, 'formal': formal, 'tmatrix': tmatrix,
                     'modal_dtn_from_stored_z': reduced.modal_dtn_from_stored_z})
    return response.scattering_response
