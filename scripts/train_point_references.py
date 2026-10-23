"""Single-seed, restartable local PTv3/Erwin S(Z) comparison on mixed 20k.

Original datasets, models and six tests are immutable. Test selection is never
used for checkpoints. Test results remain previously-viewed exploratory review.
"""
from __future__ import annotations
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time
from point_reference_runtime import ROOT, np, torch, original_loss
from point_reference_models import PointReference, PT_CONFIG, ERWIN_CONFIG, GRID_SIZE
from point_reference_data import Scenes, collate, FIELDS
from torch.utils.data import ConcatDataset, DataLoader
from prepare_point_reference_tests import sha, write_json, CACHE, TARGET

RUN_ROOT = ROOT / 'outputs/point_references_20k_20261010'
EPOCHS, BATCH_SIZE, LAMBDA_Q = 200, 16, 0.1
LOSS = original_loss()
TESTS = {
    'id_test': CACHE / 'id_test', 'shape_ood': CACHE / 'peanut_ood',
    'topology_ood': ROOT / 'outputs/v6s11_20k_all_methods_20261009/torus_review_20261010/cache/topology_ood',
    **{s: TARGET / s for s in ['n8_test', 'shape_n8_ood', 'topology_n8_ood']},
}


def seed_all():
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)


def save(path, payload):
    tmp = path.with_suffix('.pending')
    torch.save(payload, tmp)
    tmp.replace(path)


def source_identity():
    paths = [ROOT / 'scripts' / name for name in ['point_reference_runtime.py', 'point_reference_models.py',
             'point_reference_data.py', 'train_point_references.py', 'erwin_windows_binding.py', 'erwin_cabi.cpp']]
    paths += [ROOT / 'src/cspgnet/v3_learning.py', ROOT / 'work/subset11_train_deps/timm/layers/drop.py',
              ROOT / 'work/point_reference_packages_20261010/balltree_erwin-0.1.0/balltree/balltree.h',
              ROOT / 'work/point_reference_packages_20261010/balltree_erwin-0.1.0/balltree/balltree.pyx',
              ROOT / 'work/point_reference_packages_20261010/balltree_windows_build/balltree_cabi.dll']
    paths += list((ROOT / 'third_party/PointTransformerV3').rglob('*.py'))
    paths += [ROOT / 'third_party/Erwin/models/erwin.py']
    return {str(p.relative_to(ROOT)).replace('\\', '/'): sha(p) for p in paths}


def verify_cache(folder, manifest_file):
    manifest = json.loads(manifest_file.read_text(encoding='utf-8'))
    if manifest['status'] != 'complete':
        raise RuntimeError(f'Incomplete cache: {manifest_file}')
    for name in list(FIELDS) + ['object_offsets']:
        path = folder / (name + '.npy')
        key = str(path.relative_to(manifest_file.parent)).replace('\\', '/')
        if sha(path) != manifest['files_sha256'][key]:
            raise RuntimeError(f'Cache checksum mismatch: {path}')


def verify_inputs():
    for split in ['train_v3', 'train_v6s11', 'val_v3', 'val_v6s11', 'id_test', 'peanut_ood']:
        verify_cache(CACHE / split, CACHE / 'manifest.json')
    verify_cache(TESTS['topology_ood'], TESTS['topology_ood'].parent / 'manifest.json')
    for s in ['n8_test', 'shape_n8_ood', 'topology_n8_ood']:
        verify_cache(TESTS[s], TESTS[s] / 'manifest.json')
    manifest = json.loads((CACHE / 'manifest.json').read_text())
    if sha(CACHE / 'statistics.json') != manifest['normalization_sha256']:
        raise RuntimeError('Training normalization identity mismatch')
    if sha(CACHE / 'shared.npz') != manifest['files_sha256']['shared.npz']:
        raise RuntimeError('Shared canonical projection mismatch')


def identity(method):
    return {'method': method, 'sources': source_identity(), 'training_manifest_sha256': sha(CACHE / 'manifest.json'),
            'statistics_sha256': sha(CACHE / 'statistics.json'), 'shared_sha256': sha(CACHE / 'shared.npz'),
            'epochs': EPOCHS, 'batch_size': BATCH_SIZE, 'seed': 0,
            'optimizer': 'AdamW', 'lr': 1e-4, 'min_lr': 1e-7, 'weight_decay': 1e-5,
            'lambda_q': LAMBDA_Q, 'gradient_clip': 1.0, 'output_head': 'scattered', 'input': 'S(Z),140 channels',
            'selection': 'equal-weight mean of val_v3 and val_v6s11 scene scattered relative L2',
            'model_config': PT_CONFIG if method == 'ptv3' else ERWIN_CONFIG,
            'grid_size': GRID_SIZE if method == 'ptv3' else None,
            'pool_shuffle_orders': False if method == 'ptv3' else None,
            'torch': torch.__version__, 'gpu': torch.cuda.get_device_name(), 'precision': 'float32,matmul high',
            'training_scientific_status': 'partly exploratory; no upgrade of outstanding BEM/truncation gates',
            'test_interpretation': 'all six tests previously viewed; exploratory review, never checkpoint selection'}


def summary(values):
    values = np.asarray(values, dtype=float)
    return {'mean': float(np.mean(values)), 'median': float(np.median(values)),
            'p95': float(np.quantile(values, .95)), 'std_across_scenes': float(np.std(values)), 'scenes': len(values)}


def evaluate(model, dataset, stats, phi, keep=False):
    model.eval()
    results = {}
    t0 = time.perf_counter()
    with torch.no_grad():
        for batch in DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate):
            batch = batch.to('cuda')
            pred = model(batch)
            _, metrics = LOSS(pred, batch, phi, stats, LAMBDA_Q)
            for key, v in metrics.items():
                results.setdefault(key, []).extend(v.cpu().tolist())
    out = {k: summary(v) for k, v in results.items()}
    out['elapsed_seconds'] = time.perf_counter() - t0
    if keep:
        out['per_scene_metrics'] = results
    return out


def setup_data(method):
    stats = json.loads((CACHE / 'statistics.json').read_text())
    with np.load(CACHE / 'shared.npz') as shared:
        phi = torch.tensor(shared['phi_pinv'], device='cuda', dtype=torch.complex64)
    train = ConcatDataset([Scenes(CACHE / s, method) for s in ['train_v3', 'train_v6s11']])
    val = [Scenes(CACHE / s, method) for s in ['val_v3', 'val_v6s11']]
    return stats, phi, train, val


def preflight(method):
    seed_all()
    model = PointReference(method).cuda()
    stats, phi, train, val = setup_data(method)
    # A full batch, mixed scene sizes and both data sources, retaining batch 16.
    indices = list(range(8)) + list(range(10000, 10008))
    batch = collate([train[i] for i in indices]).to('cuda')
    model.eval()
    with torch.no_grad():
        first = model(batch)
        # Labels and physical Z are auxiliary-loss-only, never forward inputs.
        label, q, zr, zi = batch.labels, batch.q64_gt, batch.z_real, batch.z_imag
        batch.labels = torch.randn_like(label)
        batch.q64_gt = torch.randn_like(q)
        batch.z_real, batch.z_imag = torch.randn_like(zr), torch.randn_like(zi)
        leakage = float((first - model(batch)).abs().max())
        batch.labels, batch.q64_gt, batch.z_real, batch.z_imag = label, q, zr, zi
        original_features = batch.node_features.clone()
        batch.node_features[batch.batch_index != 0] += 0.123
        isolation = float((first[batch.batch_index == 0] - model(batch)[batch.batch_index == 0]).abs().max())
        batch.node_features = original_features
    if leakage != 0 or isolation > 2e-5 or not torch.isfinite(first).all():
        raise RuntimeError(f'Forward interface failed: leakage={leakage}, isolation={isolation}')
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS, eta_min=1e-7)
    model.train()
    timings = []
    torch.cuda.reset_peak_memory_stats()
    for i in range(8):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        opt.zero_grad(set_to_none=True)
        loss, _ = LOSS(model(batch), batch, phi, stats, LAMBDA_Q)
        if not torch.isfinite(loss):
            raise RuntimeError('Nonfinite preflight loss')
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        opt.step()
        torch.cuda.synchronize()
        timings.append(time.perf_counter() - t0)
        print(json.dumps({'event': 'preflight_step', 'method': method, 'step': i+1,
                          'loss': float(loss.detach()), 'seconds': timings[-1]}), flush=True)
    model.eval()
    with torch.no_grad():
        expected = model(batch).detach().cpu()
    checkpoint = RUN_ROOT / f'{method}_preflight_checkpoint.pt'
    save(checkpoint, {'model': model.state_dict(), 'optimizer': opt.state_dict(), 'scheduler': scheduler.state_dict()})
    peak = torch.cuda.max_memory_allocated() / 2**20
    del model, opt, scheduler
    gc.collect()
    torch.cuda.empty_cache()
    restored = PointReference(method).cuda()
    loaded = torch.load(checkpoint, map_location='cuda', weights_only=False)
    restored.load_state_dict(loaded['model'], strict=True)
    restored_opt = torch.optim.AdamW(restored.parameters(), lr=1e-4, weight_decay=1e-5)
    restored_opt.load_state_dict(loaded['optimizer'])
    restored_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(restored_opt, EPOCHS, eta_min=1e-7)
    restored_scheduler.load_state_dict(loaded['scheduler'])
    restored.eval()
    with torch.no_grad():
        restore_error = float((expected - restored(batch).cpu()).abs().max())
    if restore_error > 2e-5 or not restored_opt.state:
        raise RuntimeError('Checkpoint/optimizer restore failed')
    # Restore the true geometry mapping on a larger N=8 scene as well.
    test = Scenes(TESTS['n8_test'], method)
    with torch.no_grad():
        n8 = restored(collate([test[0]]).to('cuda'))
    if n8.shape != (1024, 2) or not torch.isfinite(n8).all():
        raise RuntimeError('N8 canonical-output mapping failed')
    ds = train.datasets[0]
    t0 = time.perf_counter()
    for start in range(0, 160, 16):
        collate([ds[i] for i in range(start, start + 16)])
    load_seconds = (time.perf_counter() - t0) / 10
    result = {'status': 'passed', 'identity': identity(method), 'parameters': sum(p.numel() for p in restored.parameters()),
              'label_and_Z_forward_leakage': leakage, 'scene_isolation_max_abs': isolation,
              'checkpoint_restore_max_abs': restore_error, 'n8_output_shape': list(n8.shape),
              'step_seconds': timings, 'warm_step_seconds_mean': float(np.mean(timings[3:])),
              'batch_load_seconds': load_seconds, 'peak_gpu_allocated_mib': peak,
              'estimated_training_hours_no_validation': float(np.mean(timings[3:])) * 1250 * EPOCHS / 3600,
              'tested_train_scenes': indices, 'scientific_release': False}
    write_json(RUN_ROOT / f'{method}_preflight.json', result)
    print(json.dumps({'event': 'preflight_passed', 'method': method, **{k: v for k, v in result.items() if k != 'identity'}}), flush=True)


def train_method(method):
    run_id = identity(method)
    receipt = json.loads((RUN_ROOT / f'{method}_preflight.json').read_text())
    # Normalize tuple/list encoding exactly as it will be saved in JSON.
    run_id = json.loads(json.dumps(run_id))
    if receipt['status'] != 'passed' or receipt['identity'] != run_id:
        raise RuntimeError('Preflight source/protocol/data mismatch')
    verify_inputs()
    out = RUN_ROOT / 'runs' / method
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'run_manifest.json').exists() and json.loads((out / 'run_manifest.json').read_text()) != run_id:
        raise RuntimeError('Existing run identity mismatch')
    write_json(out / 'run_manifest.json', run_id)
    seed_all()
    model = PointReference(method).cuda()
    stats, phi, ds, vals = setup_data(method)
    gen = torch.Generator().manual_seed(0)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, generator=gen, collate_fn=collate)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS, eta_min=1e-7)
    history, best, completed = [], float('inf'), 0
    if (out / 'last.pt').exists():
        ckpt = torch.load(out / 'last.pt', map_location='cuda', weights_only=False)
        if ckpt['identity'] != run_id:
            raise RuntimeError('Checkpoint provenance mismatch')
        model.load_state_dict(ckpt['model'], strict=True)
        opt.load_state_dict(ckpt['optimizer'])
        schedule.load_state_dict(ckpt['scheduler'])
        history, best, completed = ckpt['history'], ckpt['best'], ckpt['epoch']
        random.setstate(ckpt['random'])
        np.random.set_state(ckpt['numpy'])
        torch.set_rng_state(ckpt['torch'].cpu())
        torch.cuda.set_rng_state_all([s.cpu() for s in ckpt['cuda']])
        gen.set_state(ckpt['loader'].cpu())
    for epoch in range(completed, EPOCHS):
        model.train()
        started = time.perf_counter()
        total_loss, seen = 0., 0
        torch.cuda.reset_peak_memory_stats()
        for step, batch in enumerate(loader, 1):
            batch = batch.to('cuda')
            opt.zero_grad(set_to_none=True)
            loss, _ = LOSS(model(batch), batch, phi, stats, LAMBDA_Q)
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite training loss')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            opt.step()
            n = len(batch.sample_node_counts)
            total_loss += float(loss.detach()) * n
            seen += n
            if step == 1 or step % 100 == 0:
                state = {'method': method, 'status': 'running', 'pid': os.getpid(), 'epoch': epoch + 1,
                         'completed_epochs': epoch, 'batch': step, 'batches': len(loader), 'scenes_seen': seen,
                         'updated_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                         'epoch_elapsed_seconds': time.perf_counter() - started}
                write_json(out / 'status.json', state)
                print(json.dumps({'event': 'batch', **state}), flush=True)
        schedule.step()
        reviews = [evaluate(model, d, stats, phi) for d in vals]
        score = float(np.mean([r['scattered_rel_l2']['mean'] for r in reviews]))
        record = {'epoch': epoch + 1, 'train_loss': total_loss / seen, 'val_v3': reviews[0], 'val_v6s11': reviews[1],
                  'selection_score': score, 'lr': opt.param_groups[0]['lr'], 'seconds': time.perf_counter() - started,
                  'peak_gpu_allocated_mib': torch.cuda.max_memory_allocated() / 2**20}
        history.append(record)
        if score < best:
            best = score
            save(out / 'best.pt', {'identity': run_id, 'model': model.state_dict(), 'epoch': epoch+1, 'score': best})
        save(out / 'last.pt', {'identity': run_id, 'model': model.state_dict(), 'optimizer': opt.state_dict(),
              'scheduler': schedule.state_dict(), 'epoch': epoch+1, 'history': history, 'best': best,
              'random': random.getstate(), 'numpy': np.random.get_state(), 'torch': torch.get_rng_state(),
              'cuda': torch.cuda.get_rng_state_all(), 'loader': gen.get_state()})
        write_json(out / 'history.json', history)
        write_json(out / 'status.json', {'method': method, 'status': 'running', 'pid': os.getpid(),
                                        'completed_epochs': epoch+1, 'best_score': best, 'latest': record})
        print(json.dumps({'event': 'epoch', 'method': method, **record, 'best': best}), flush=True)
    write_json(out / 'status.json', {'method': method, 'status': 'training_complete', 'epoch': EPOCHS, 'best_score': best})
    best_state = torch.load(out / 'best.pt', map_location='cuda', weights_only=False)
    if best_state['identity'] != run_id:
        raise RuntimeError('Best checkpoint identity mismatch')
    model.load_state_dict(best_state['model'], strict=True)
    tests = {}
    for split, folder in TESTS.items():
        result = evaluate(model, Scenes(folder, method), stats, phi, keep=True)
        source = json.loads((ROOT / 'datasets/canonical_response_v3/data' / split / 'manifest.json').read_text())
        ordered = sorted(source['samples'], key=lambda r: r['index'])
        if len(ordered) != result['scattered_rel_l2']['scenes']:
            raise RuntimeError('Test manifest/result count mismatch')
        if any('ood_composition' in r for r in ordered):
            result['subgroups'] = {}
            for group in sorted(set(r['ood_composition'] for r in ordered)):
                mask = [r['ood_composition'] == group for r in ordered]
                result['subgroups'][group] = {k: summary(np.asarray(v)[mask]) for k, v in result['per_scene_metrics'].items()}
        tests[split] = result
        write_json(out / f'{split}_results.json', result)
        print(json.dumps({'event': 'test_complete', 'method': method, 'split': split,
                          'scattered_rel_l2': result['scattered_rel_l2']}), flush=True)
    write_json(out / 'all_tests.json', {'status': 'complete', 'method': method,
            'checkpoint_epoch': best_state['epoch'], 'checkpoint_sha256': sha(out / 'best.pt'),
            'identity': run_id, 'tests': tests})
    write_json(out / 'status.json', {'method': method, 'status': 'complete', 'epoch': EPOCHS,
                                    'best_epoch': best_state['epoch'], 'best_score': best, 'tests_complete': list(tests)})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--method', choices=['ptv3', 'erwin'], required=True)
    parser.add_argument('--preflight', action='store_true')
    args = parser.parse_args()
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision('high')
    if not torch.cuda.is_available():
        raise RuntimeError('Local CUDA unavailable')
    if args.preflight:
        preflight(args.method)
    else:
        train_method(args.method)


if __name__ == '__main__':
    main()
