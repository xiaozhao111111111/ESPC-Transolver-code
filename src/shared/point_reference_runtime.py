"""Official point-model imports with minimal optional-dependency adapters."""
import importlib.util
from pathlib import Path
import sys
import types
from shared.paths import ROOT
import numpy as np
import torch

def official_imports():
    from shared.vendor.timm.drop import DropPath
    from shared.vendor.timm.weight_init import trunc_normal_
    for name in ('timm', 'timm.layers', 'timm.models', 'timm.models.layers'):
        module = types.ModuleType(name)
        module.__path__ = []
        module.DropPath, module.trunc_normal_ = DropPath, trunc_normal_
        sys.modules[name] = module
    if 'torch_cluster' not in sys.modules and importlib.util.find_spec('torch_cluster') is None:
        module = types.ModuleType('torch_cluster')
        def unavailable(*args, **kwargs):
            raise RuntimeError('Install torch_cluster to enable nonzero message-passing steps')
        module.radius_graph = unavailable
        sys.modules['torch_cluster'] = module
def load_official_ptv3():
    official_imports()
    directory = ROOT / 'src/PTv3/backbone'
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
    from Erwin.shared.erwin_windows_binding import install
    if 'balltree' not in sys.modules:
        install()
    name = '_official_erwin_reference'
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, ROOT / 'src/Erwin/backbone/models/erwin.py')
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name].ErwinTransformer
def original_loss():
    from shared.learning import loss_and_scene_metrics
    return loss_and_scene_metrics
