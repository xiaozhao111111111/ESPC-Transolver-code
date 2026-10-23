"""Isolated v2 field-only adapter; original numerical/model sources stay frozen."""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import time
import traceback

import remote_point_surface_only as v1
from point_surface_only_tree_gate_v2 import qualify_extra

ROOT = v1.ROOT
OUT = ROOT / 'outputs/point_surface_only_autodl_20261012_v2'


def configure():
    v1.OUT = OUT
    surface = v1.configure()
    from prepare_point_reference_tests import sha
    old_identity = surface.identity

    def identity(method):
        value = old_identity(method)
        branch = value['minimal_branch']
        branch.update({
            'adapter_sha256': sha(Path(__file__)),
            'supervisor_sha256': sha(ROOT / 'scripts/supervise_remote_point_surface_only_v2.py'),
            'protocol_sha256': sha(ROOT / 'POINT_SURFACE_ONLY_AUTODL_20261012_v2.md'),
            'inventory_sha256': sha(OUT / 'transfer_inventory.json'),
            'local_interface_receipt_sha256': sha(OUT / 'interface_qualification.json'),
            'tree_gate_sha256': sha(ROOT / 'scripts/point_surface_only_tree_gate_v2.py'),
            'tree_input_policy': 'use source-bound saved per-scene tree arrays; native order diagnostic only',
        })
        return json.loads(json.dumps(value))

    surface.identity = identity
    surface.frozen.identity = identity
    surface.OUT = OUT
    surface.frozen.RUN_ROOT = OUT
    return surface


def verify_transfer():
    v1.OUT = OUT
    return v1.verify_transfer()


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', choices=['ptv3', 'erwin'], required=True)
    parser.add_argument('--preflight', action='store_true')
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    surface = configure()
    try:
        verify_transfer()
        if args.preflight:
            surface.preflight(args.method)
            qualify_extra(surface, args.method)
        else:
            v1.port.train(surface, args.method)
    except BaseException as error:
        surface.write_json(OUT / f'{args.method}_failure_{time.time_ns()}.json', {
            'pid': os.getpid(), 'method': args.method, 'preflight': args.preflight,
            'error': repr(error), 'traceback': traceback.format_exc()})
        raise


if __name__ == '__main__':
    main()
