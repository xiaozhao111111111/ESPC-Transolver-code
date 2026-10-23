"""User-selected field-only branch; frozen models and train loop unchanged."""
from __future__ import annotations
import ast
import copy
import json
from pathlib import Path
import sys
import remote_point_surface as port

ROOT = port.ROOT
OUT = ROOT/'outputs/point_surface_only_autodl_20261012'


def adapted_function(module, name, expected_loss_calls, strip_flux_backup=False):
    path = Path(module.__file__)
    node = copy.deepcopy(next(n for n in ast.parse(path.read_text()).body
                              if isinstance(n, ast.FunctionDef) and n.name == name))
    changed = 0
    for child in ast.walk(node):
        if isinstance(child, ast.Call) and isinstance(child.func, ast.Attribute) and child.func.attr == 'LOSS':
            if len(child.args) != 5 or not isinstance(child.args[-1], ast.Constant) or child.args[-1].value != .1:
                raise RuntimeError('Qualification loss call changed')
            child.args[-1] = ast.Constant(0.); changed += 1
        if strip_flux_backup and isinstance(child, ast.Tuple):
            if all(isinstance(x, ast.Constant) for x in child.elts) and [x.value for x in child.elts] == ['labels','q64_gt','z_real','z_imag','incident']:
                child.elts = [ast.Constant('labels'), ast.Constant('incident')]
    if changed != expected_loss_calls: raise RuntimeError('Qualification loss call count changed')
    ns = dict(module.__dict__)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])), str(path), 'exec'), ns)
    return ns[name]


def configure():
    port.OUT = OUT
    s = port.configure()
    from point_surface_only_data import SurfaceScenes, collate, loss_and_metrics
    from prepare_point_reference_tests import sha
    s.SurfaceScenes = SurfaceScenes; s.collate = collate
    s.frozen.Scenes = SurfaceScenes; s.frozen.collate = collate
    s.frozen.LAMBDA_Q = 0.; s.frozen.LOSS = loss_and_metrics
    s.OUT = OUT; s.frozen.RUN_ROOT = OUT
    original = s.identity

    def identity(method):
        value = original(method)
        value.update(lambda_q=0., physical_Z_loaded=False, projected_flux_metric_available=False,
            minimal_branch={'adapter_sha256':sha(Path(__file__)),
                'loader_sha256':sha(ROOT/'scripts/point_surface_only_data.py'),
                'supervisor_sha256':sha(ROOT/'scripts/supervise_remote_point_surface_only.py'),
                'protocol_sha256':sha(ROOT/'POINT_SURFACE_ONLY_AUTODL_20261012.md'),
                'inventory_sha256':sha(OUT/'transfer_inventory.json'),
                'local_interface_receipt_sha256':sha(OUT/'interface_qualification.json'),
                'only_fields':['labels','incident'], 'change':'user-selected field-only supervision',
                'comparison_limit':'input, query decoder AND loss differ from old S(Z) runs'})
        return json.loads(json.dumps(value))

    s.identity = identity; s.frozen.identity = identity

    def verify_inputs():
        verify_transfer()
        return s.verify_sidecar()

    s.verify_inputs = verify_inputs; s.frozen.verify_inputs = verify_inputs
    s.preflight = adapted_function(s, 'preflight', 2, strip_flux_backup=True)
    return s


def verify_transfer():
    from prepare_point_reference_tests import sha
    manifest = json.loads((OUT/'transfer_inventory.json').read_text())
    if manifest['status'] != 'prepared' or manifest.get('lambda_q') != 0:
        raise RuntimeError('Wrong field-only inventory')
    for rel, record in manifest['files'].items():
        if Path(rel).name in {'features.npy','q64.npy','z_real.npy','z_imag.npy'}:
            raise RuntimeError('Physical feature/flux tensor in minimal inventory')
        path = ROOT/rel
        if path.stat().st_size != record['size'] or sha(path) != record['sha256']:
            raise RuntimeError('Minimal transfer checksum mismatch: '+rel)
    receipt = json.loads((OUT/'interface_qualification.json').read_text())
    if receipt['status'] != 'passed' or receipt['inventory_sha256'] != sha(OUT/'transfer_inventory.json'):
        raise RuntimeError('Missing matching field-only interface qualification')
    return len(manifest['files'])


def main():
    import argparse
    import os
    import time
    import traceback
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', choices=['ptv3','erwin'], required=True)
    parser.add_argument('--preflight', action='store_true')
    args = parser.parse_args(); OUT.mkdir(parents=True, exist_ok=True)
    s = configure()
    try:
        verify_transfer()
        if args.preflight:
            s.preflight(args.method)
            adapted_function(port, 'qualify_extra', 3)(s, args.method)
        else:
            port.train(s, args.method)
    except BaseException as error:
        s.write_json(OUT/f'{args.method}_failure_{time.time_ns()}.json',
            {'pid':os.getpid(),'method':args.method,'preflight':args.preflight,
             'error':repr(error),'traceback':traceback.format_exc()})
        raise


if __name__ == '__main__': main()
