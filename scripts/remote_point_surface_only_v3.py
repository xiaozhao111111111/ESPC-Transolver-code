"""v3 source-bound field-only adapter with identity-consistent preflight."""
from __future__ import annotations
import json
import os
from pathlib import Path
import time
import traceback

import remote_point_surface_only_v2 as v2

ROOT = v2.ROOT
OUT = ROOT / 'outputs/point_surface_only_autodl_20261012_v3'


def configure():
    v2.OUT = OUT
    surface = v2.configure()
    from prepare_point_reference_tests import sha
    previous_identity = surface.identity

    def identity(method):
        value = previous_identity(method)
        value['minimal_branch'].update({
            'adapter_sha256': sha(Path(__file__)),
            'supervisor_sha256': sha(ROOT / 'scripts/supervise_remote_point_surface_only_v3.py'),
            'protocol_sha256': sha(ROOT / 'POINT_SURFACE_ONLY_AUTODL_20261012_v3.md'),
            'inventory_sha256': sha(OUT / 'transfer_inventory.json'),
            'local_interface_receipt_sha256': sha(OUT / 'interface_qualification.json'),
            'preflight_identity_binding': 'preflight recompiled after v3 identity rebinding',
        })
        return json.loads(json.dumps(value))

    surface.identity = identity
    surface.frozen.identity = identity
    surface.OUT = OUT
    surface.frozen.RUN_ROOT = OUT
    # The v2-derived function held a copied global namespace created before
    # this wrapper rebound identity. Rebuild the same source-derived function
    # now, from the frozen function body and current patched namespace.
    surface.preflight = v2.v1.adapted_function(
        surface, 'preflight', 2, strip_flux_backup=True)
    return surface


def verify_transfer():
    v2.OUT = OUT
    return v2.verify_transfer()


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
            v2.qualify_extra(surface, args.method)
        else:
            v2.v1.port.train(surface, args.method)
    except BaseException as error:
        surface.write_json(OUT / f'{args.method}_failure_{time.time_ns()}.json', {
            'pid': os.getpid(), 'method': args.method, 'preflight': args.preflight,
            'error': repr(error), 'traceback': traceback.format_exc()})
        raise


if __name__ == '__main__':
    main()
