"""New normalization-bound S(Z) sidecars; existing datasets remain immutable."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import time
from point_reference_runtime import ROOT, np, original_response, leaf

CACHE = ROOT / 'outputs/v6s11_20k_all_methods_20261009/prepared_retry5'
TARGET = ROOT / 'outputs/point_references_20k_20261010/test_sidecar'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(4 * 1024**2), b''):
            h.update(b)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    tmp = path.with_suffix('.pending')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')
    tmp.replace(path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--splits', nargs='+', default=['n8_test', 'shape_n8_ood', 'topology_n8_ood'])
    args = p.parse_args()
    stats = json.loads((CACHE / 'statistics.json').read_text())
    response = original_response()
    # Execute the exact previous feature packer without unrelated FEM/cloud imports.
    packer = leaf(ROOT / 'scripts/prepare_torus_20k_review.py', ['response_features'],
                  {'np': np, 'scattering_response': response}).response_features
    ref = np.load(CACHE / 'id_test/features.npy', mmap_mode='r')
    off = np.load(CACHE / 'id_test/object_offsets.npy') * 128
    parity = []
    for index in [0, 47, 999]:
        with np.load(ROOT / f'datasets/canonical_response_v3/data/id_test/sample_{index:05d}.npz') as z:
            actual = packer(dict(z), stats)
        expected = ref[off[index]:off[index + 1]]
        diff = float(np.linalg.norm(actual.astype(float) - expected) / np.linalg.norm(expected))
        if diff > 2e-6:
            raise RuntimeError(f'ID feature conversion parity failed: {diff}')
        parity.append(diff)
    print(json.dumps({'event': 'input_packer_parity', 'errors': parity}), flush=True)
    TARGET.mkdir(parents=True, exist_ok=True)
    for split in args.splits:
        source = ROOT / 'datasets/canonical_response_v3/data' / split
        manifest = json.loads((source / 'manifest.json').read_text(encoding='utf-8'))
        records = sorted(manifest['samples'], key=lambda r: r['index'])
        count = len(records)
        if count != manifest['num_samples'] or manifest['status'] != 'complete':
            raise RuntimeError('Source split not complete')
        target = TARGET / split
        target.mkdir(exist_ok=True)
        identity = {'source_manifest': sha(source / 'manifest.json'),
                    'statistics': sha(CACHE / 'statistics.json'),
                    'source_packer': sha(ROOT / 'scripts/prepare_torus_20k_review.py'),
                    'response_leaf': sha(ROOT / 'scripts/point_reference_runtime.py')}
        if (target / 'manifest.json').exists():
            done = json.loads((target / 'manifest.json').read_text())
            if done['identity'] != identity or done['status'] != 'complete':
                raise RuntimeError('Existing test-sidecar identity mismatch')
            print(f'{split} already complete', flush=True)
            continue
        offsets = np.r_[0, np.cumsum([len(r['objects']) for r in records])].astype(np.int64)
        nn = int(offsets[-1]) * 128
        specs = {'features': (nn, 140), 'labels': (nn, 2), 'incident': (nn, 2),
                 'q64': (nn, 2), 'z_real': (nn, 64), 'z_imag': (nn, 64)}
        progress_file = target / 'progress.json'
        progress = json.loads(progress_file.read_text()) if progress_file.exists() else {'identity': identity, 'done': 0}
        if progress['identity'] != identity:
            raise RuntimeError('Incomplete sidecar identity mismatch')
        arrays = {k: np.lib.format.open_memmap(target / (k + '.npy'),
                                             mode='r+' if (target / (k + '.npy')).exists() else 'w+',
                                             dtype=np.float32, shape=shape) for k, shape in specs.items()}
        np.save(target / 'object_offsets.npy', offsets)
        start = time.perf_counter()
        for i in range(progress['done'], count):
            r = records[i]
            path = source / f'sample_{int(r["index"]):05d}.npz'
            if sha(path) != r['sha256']:
                raise RuntimeError(f'Raw source checksum mismatch: {path}')
            with np.load(path, allow_pickle=False) as z:
                data = dict(z)
            lo, hi = offsets[i:i+2] * 128
            arrays['features'][lo:hi] = packer(data, stats)
            scatter = data['u_total_gt'] - data['u_inc']
            arrays['labels'][lo:hi] = np.stack([scatter.real, scatter.imag], -1).reshape(-1, 2) / stats['scatter_complex_rms']
            for dest, src in [('incident', 'u_inc'), ('q64', 'q64_gt')]:
                v = data[src]
                arrays[dest][lo:hi] = np.stack([v.real, v.imag], -1).reshape(-1, 2)
            arrays['z_real'][lo:hi] = data['Z'].real.reshape(-1, 64)
            arrays['z_imag'][lo:hi] = data['Z'].imag.reshape(-1, 64)
            q = np.einsum('onm,om->on', data['Z'].astype(np.complex128),
                          np.einsum('mn,on->om', data['phi_pinv'].astype(np.complex128), data['u_total_gt']))
            if np.linalg.norm(q - data['q64_gt']) / np.linalg.norm(q) > 1e-4:
                raise RuntimeError('Stored derived flux identity mismatch')
            if (i + 1) % 25 == 0 or i == count - 1:
                for a in arrays.values():
                    a.flush()
                write_json(progress_file, {'identity': identity, 'done': i + 1, 'total': count})
                print(json.dumps({'event': 'test_pack', 'split': split, 'done': i+1, 'total': count,
                                  'seconds': time.perf_counter()-start}), flush=True)
        for a in arrays.values():
            if not np.isfinite(a).all():
                raise RuntimeError('Nonfinite test sidecar')
        write_json(target / 'records.json', records)
        write_json(target / 'manifest.json', {'status': 'complete', 'scenes': count, 'identity': identity,
                   'scientific_status': 'previously viewed exploratory review; no label changes',
                   'id_parity': parity, 'files_sha256': {str(f.relative_to(target)): sha(f)
                        for f in target.glob('*') if f.name not in ['manifest.json', 'progress.json']}})


if __name__ == '__main__':
    main()
