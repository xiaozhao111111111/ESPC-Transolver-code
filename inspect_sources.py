"""Check the code-only export with the Python standard library; no model is run."""
import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def source_files():
    return (p for p in ROOT.rglob('*') if p.is_file() and '.git' not in p.relative_to(ROOT).parts)


def main():
    manifest = json.loads((ROOT / 'CODE_MANIFEST.json').read_text(encoding='utf-8'))
    parsed = 0
    for rel, record in manifest['files'].items():
        p = ROOT / rel
        data = p.read_bytes()
        if len(data) != record['size'] or hashlib.sha256(data).hexdigest() != record['sha256']:
            raise RuntimeError('Source hash mismatch: ' + rel)
        if p.suffix == '.py':
            ast.parse(data.decode('utf-8-sig'), str(p))
            parsed += 1
    forbidden = {'.npy', '.npz', '.pt', '.pth', '.pkl', '.pickle', '.dll', '.exe', '.pyd', '.pyc'}
    unexpected = [str(p.relative_to(ROOT)) for p in source_files() if p.suffix.lower() in forbidden]
    if unexpected:
        raise RuntimeError('Non-source data/runtime files remain: ' + repr(unexpected))
    tracked = set(manifest['files']) | {'CODE_MANIFEST.json'}
    untracked = [p.relative_to(ROOT).as_posix() for p in source_files()
                 if p.relative_to(ROOT).as_posix() not in tracked]
    if untracked:
        raise RuntimeError('Untracked export files: ' + repr(untracked))
    print(json.dumps({'status': 'passed', 'scope': manifest['scope'],
        'sha256_verified_files': manifest['file_count'], 'python_sources_parsed': parsed,
        'bytes': manifest['bytes'], 'data_or_runtime_files': 0, 'training_executed': False}, ensure_ascii=False))


if __name__ == '__main__':
    main()
