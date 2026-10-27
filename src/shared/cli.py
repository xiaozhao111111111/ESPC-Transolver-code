"""Common CLI plumbing; each public script selects exactly one model method."""
import argparse
import ast
import json
import os
from pathlib import Path
import sys

def main(method):
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description='Train '+method)
    if method in ('ptv3', 'erwin'):
        parser.add_argument('--input', choices=('s-z', 'geo-strong', 'surface'), default='geo-strong')
    parser.add_argument('--data-dir', type=Path, default=root/'data')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--workers', type=int, default=0)
    parser.add_argument('--resume', action='store_true', help='Resume a complete last.pt from --output')
    checks = parser.add_mutually_exclusive_group()
    checks.add_argument('--smoke', action='store_true', help='Forward/backward on one batch from each training source; no saved run')
    checks.add_argument('--check-data', action='store_true', help='Verify packaged data SHA-256 and exit')
    checks.add_argument('--check-sources', action='store_true', help='Parse packaged Python sources and exit')
    if method == 'erwin':
        checks.add_argument('--build-native', action='store_true', help='Build the native ball-tree binding and exit')
    parser.add_argument('--sizes-only', action='store_true', help='Skip hashing with --check-data')
    args = parser.parse_args()
    if min(args.epochs, args.batch_size) < 1 or args.workers < 0:
        parser.error('epochs/batch-size must be positive; workers must be nonnegative')
    if args.sizes_only and not args.check_data:
        parser.error('--sizes-only requires --check-data')
    os.environ['ESPC_DATA_ROOT'] = str(args.data_dir.resolve())
    if args.check_sources:
        sources = [*(root/'src').rglob('*.py'), *(root/'scripts').glob('*.py')]
        for path in sources:
            ast.parse(path.read_text(encoding='utf-8-sig'), filename=str(path))
        print(json.dumps({'python_files': len(sources), 'syntax': 'passed'}))
        return
    from shared.data_integrity import verify_data
    if args.check_data:
        print(json.dumps(verify_data(args.sizes_only)))
        return
    if not getattr(args, 'build_native', False):
        verify_data(sizes_only=True)
    branch = getattr(args, 'input', None)
    args.output = (args.output or root/'outputs'/method/(branch or 'default')).resolve()
    # Optional local wheels. A regular environment works without runtime/.
    runtime = root/'runtime'
    sys.path[:0] = [str(runtime/name) for name in ('torch','scientific_deps','numeric_deps','point_deps','model_deps')
                   if (runtime/name).is_dir()]
    sys.path[:0] = [str(root/'src/shared'), str(root/'src/PIBNet/backbone'), str(root/'src/PIBNet')]
    os.environ.setdefault('OMP_NUM_THREADS', '4')
    os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
    if getattr(args, 'build_native', False):
        from Erwin.shared.erwin_windows_binding import build
        print(build())
        return
    if method in ('ptv3', 'erwin'):
        from shared.training.canonical_reference import run
    else:
        from shared.training.mixed20k import run
    run(args, method)
