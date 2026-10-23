"""Source-bound qualification and frozen-loop training for Geo-strong Erwin."""
from __future__ import annotations

import argparse
import copy
from dataclasses import replace
import gc
import json
import os
import pickle
from pathlib import Path
import random
import sys
import time
import traceback

sys.path.insert(0, str(Path(__file__).resolve().parent))
os.environ.setdefault('PYTORCH_ALLOC_CONF', 'expandable_segments:True')
if '--qualify' in sys.argv:
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
from point_reference_runtime import ROOT, np, torch
from point_reference_io_recovery import read_json, write_json, save_checkpoint
from prepare_point_reference_tests import sha, CACHE
import point_geo_strong_erwin as geo
import train_point_references as frozen
from point_surface_only_data import _FIELD_LOSS
from train_point_surface_erwin_cache_recovery import cpu_copy, max_difference

OUT = geo.OUT
PROTOCOL = ROOT/'ERWIN_GEO_STRONG_LOCAL_20261013.md'


def identity(method='erwin'):
    value = frozen.identity_original(method)
    value.update(input='Geo-strong: raw 2048 point+normal cloud, nearest16, 128 canonical nodes/body',
        lambda_q=0.0, physical_Z_loaded=False, projected_flux_metric_available=False,
        geometry_cache_manifest_sha256=sha(geo.SIDE/'manifest.json'),
        point_encoder_source_sha256=sha(ROOT/'src/cspgnet/v3_learning.py'),
        geometry_sources=geo.input_binding(), execution_target='local Windows CUDA',
        new_sources={p.relative_to(ROOT).as_posix(): sha(p) for p in
            (Path(__file__), ROOT/'scripts/point_geo_strong_erwin.py',
             ROOT/'scripts/supervise_geo_strong_erwin_local.py',
             ROOT/'scripts/point_surface_only_data.py',
             ROOT/'scripts/train_point_surface_erwin_cache_recovery.py', PROTOCOL)},
        encoder='unchanged GeoStrongAdapted definitions; PIBNet adapter replaced by canonical Erwin',
        decoder='existing canonical Erwin 32->64->2 head; no surface-to-query decoder',
        execution_cache_cleanup='unused CUDA cache released after every 10 updates/transfers',
        comparison_limit='representation/encoder/decoder differ from direct surface; loss also differs from old S(Z)')
    return json.loads(json.dumps(value))


def require_resources():
    import psutil
    value = {'available_ram_bytes': psutil.virtual_memory().available,
             'gpu_allocated_bytes': torch.cuda.memory_allocated(),
             'gpu_reserved_bytes': torch.cuda.memory_reserved(), 'updated_at': time.time()}
    if value['available_ram_bytes'] < 1024**3 or psutil.disk_usage(str(OUT)).free < 512*1024**2:
        raise RuntimeError('RAM or disk reserve exhausted: '+repr(value))
    return value


def install_cleanup():
    step, move = torch.optim.AdamW.step, geo.GeoBatch.to
    transfers = 0
    def cleaned_step(opt, *args, **kwargs):
        result = step(opt, *args, **kwargs)
        opt._geo_updates = getattr(opt, '_geo_updates', 0)+1
        if opt._geo_updates % 10 == 0:
            torch.cuda.empty_cache()
        return result
    def cleaned_to(batch, *args, **kwargs):
        nonlocal transfers
        transfers += 1
        if transfers % 10 == 0:
            torch.cuda.empty_cache()
        return move(batch, *args, **kwargs)
    torch.optim.AdamW.step = cleaned_step
    geo.GeoBatch.to = cleaned_to


class TrainingScenes(geo.Scenes):
    """Unique diagnostic IDs for the two sources; never changes tensors/order."""
    def __getitem__(self, scene):
        item = super().__getitem__(scene)
        if self.split == 'train_v6s11':
            item['index'] += 10000
        return item


def setup_data(method='erwin'):
    stats = read_json(CACHE/'statistics.json')
    train = torch.utils.data.ConcatDataset([TrainingScenes(geo.SPLITS[s], method)
        for s in ('train_v3', 'train_v6s11')])
    vals = [geo.Scenes(geo.SPLITS[s], method) for s in ('val_v3', 'val_v6s11')]
    return stats, None, train, vals


def configure():
    if not hasattr(frozen, 'identity_original'):
        frozen.identity_original = frozen.identity
    frozen.RUN_ROOT = OUT
    frozen.LAMBDA_Q = 0.0
    frozen.PointReference, frozen.Scenes, frozen.collate = geo.model, geo.Scenes, geo.collate
    frozen.LOSS, frozen.setup_data, frozen.identity = _FIELD_LOSS, setup_data, identity
    frozen.verify_inputs, frozen.save = geo.verify_geometry, save_checkpoint
    def guarded_write(path, value):
        if Path(path).name == 'status.json':
            torch.cuda.empty_cache()
            require_resources()
        write_json(path, value)
    frozen.write_json = guarded_write
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision('high')
    if not torch.cuda.is_available():
        raise RuntimeError('Local CUDA unavailable')


def check_cpu_interface():
    from scipy.spatial import cKDTree
    results = {}
    maximum = 0.
    expected_counts = {'train_v3': 10000, 'train_v6s11': 10000,
        'val_v3': 1000, 'val_v6s11': 1000, 'id_test': 1000, 'n8_test': 1000,
        'shape_ood': 500, 'topology_ood': 500, 'shape_n8_ood': 250, 'topology_n8_ood': 250}
    for split, folder in geo.SPLITS.items():
        ds = geo.Scenes(folder)
        if len(ds) != expected_counts[split]:
            raise RuntimeError('Split scene count changed: '+split)
        checks = []
        for scene in (0, len(ds)//2, len(ds)-1):
            item = ds[scene]
            q = len(item['positions']); objects = q//128
            if item['node_features'].shape != (q, 140) or item['geo_local'].shape != (q, 16, 6) \
                    or item['geo_global'].shape != (objects, 2048, 6):
                raise RuntimeError('Geo-strong tensor interface failed')
            if item['labels'].shape != (q, 2) or item['incident'].shape != (q, 2):
                raise RuntimeError('Field ordering/shape failed')
            if not all(torch.isfinite(item[k]).all() for k in
                       ('node_features', 'positions', 'geo_local', 'geo_global', 'labels', 'incident')):
                raise RuntimeError('Nonfinite geometry or field')
            idx, mask, rotations = item['tree']
            if not torch.equal(torch.sort(idx[mask]).values, torch.arange(q)):
                raise RuntimeError('Canonical tree alignment failed')
            for j, r in enumerate(rotations):
                if len(r) != len(idx)//2**j or not torch.equal(torch.sort(r).values, torch.arange(len(r))):
                    raise RuntimeError('Tree rotation mapping failed')
            cloud = item['geo_global'].numpy()
            if np.max(np.abs(np.linalg.norm(cloud[:, :, 3:], axis=-1)-1)) > 2e-5:
                raise RuntimeError('Saved surface normals are not unit vectors')
            ctx = item['node_features'].numpy()[:, 128:].reshape(objects, 128, 12)
            nearest = item['geo_local'].numpy().reshape(objects, 128, 16, 6)
            for body in range(objects):
                _, k = cKDTree(cloud[body, :, :3]).query(ctx[body, :, 2:5], k=16)
                reference = np.concatenate((cloud[body, k, :3]-ctx[body, :, None, 2:5],
                                            cloud[body, k, 3:]), axis=-1)
                # Within a tied neighbor set order may differ, but distances and
                # each cached representative must independently be valid.
                actual_distance = np.sort(np.linalg.norm(nearest[body, :, :, :3], axis=-1), axis=1)
                expected_distance = np.sort(np.linalg.norm(reference[:, :, :3], axis=-1), axis=1)
                maximum = max(maximum, float(np.max(np.abs(actual_distance-expected_distance))))
            checks.append({'scene': scene, 'objects': objects, 'canonical_nodes': q,
                           'raw_surface_points': objects*2048, 'padded_nodes': len(idx)})
        results[split] = checks
    if maximum > 2e-6:
        raise RuntimeError('Nearest16 geometric correctness failed: '+str(maximum))
    return {'splits': results, 'nearest16_distance_max_abs': maximum,
            'no_physical_tensor_attributes': True}


def update(model, optimizer, batch, stats, keep=False):
    model.train(); optimizer.zero_grad(set_to_none=True)
    pred = model(batch)
    loss, _ = _FIELD_LOSS(pred, batch, None, stats, 0.0)
    if not torch.isfinite(loss):
        raise RuntimeError('Nonfinite qualification loss')
    loss.backward()
    if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
        raise RuntimeError('Nonfinite qualification gradient')
    result = {'prediction': pred.detach().cpu(), 'loss': float(loss.detach())}
    if keep:
        result['gradients'] = [p.grad.detach().cpu().clone() for p in model.parameters() if p.grad is not None]
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
    optimizer.step()
    return result


def complete_state(model, optimizer, schedule, generator):
    return cpu_copy({'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
        'scheduler': schedule.state_dict(), 'torch': torch.get_rng_state(),
        'cuda': torch.cuda.get_rng_state_all(), 'loader': generator.get_state(),
        'python_rng': pickle.dumps(random.getstate()), 'numpy_rng': pickle.dumps(np.random.get_state())}, torch)


def qualification():
    geo.verify_geometry()
    torch.use_deterministic_algorithms(True)
    interface = check_cpu_interface()
    install_cleanup()
    frozen.seed_all()
    stats, _, ds, _ = setup_data()
    model = geo.model().cuda()
    batch = geo.collate([ds[i] for i in list(range(8))+list(range(10000, 10008))]).to('cuda')
    model.eval()
    with torch.no_grad():
        first = model(batch)
        features = batch.node_features.clone()
        local = model.local_mlp(batch.geo_local).amax(1)
        glob = model.global_mlp(batch.geo_global)
        code = torch.cat((glob.amax(1), glob.mean(1)), dim=-1).repeat_interleave(128, dim=0)
        features[:, :128] = model.combiner(torch.cat((local, code), dim=-1))
        encoder_error = float((first-model.backbone(replace(batch, node_features=features))).abs().max())
        label, incident = batch.labels, batch.incident
        batch.labels, batch.incident = torch.randn_like(label), torch.randn_like(incident)
        leakage = float((first-model(batch)).abs().max())
        batch.labels, batch.incident = label, incident
        local0, global0 = batch.geo_local, batch.geo_global
        local_new, global_new = local0.clone(), global0.clone()
        local_new[batch.batch_index != 0] += .123
        body_count = int(batch.sample_node_counts[0])//128
        global_new[body_count:] += .123
        batch.geo_local, batch.geo_global = local_new, global_new
        isolation = float((first[batch.batch_index == 0]-model(batch)[batch.batch_index == 0]).abs().max())
        batch.geo_local, batch.geo_global = local0, global0
    if encoder_error > 2e-5 or leakage != 0 or isolation > 2e-5:
        raise RuntimeError('Forward qualification failed: '+repr((encoder_error, leakage, isolation)))
    del batch, first, features, local, glob, code, local_new, global_new, local0, global0
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 200, eta_min=1e-7)
    gen = torch.Generator().manual_seed(0)
    loader = torch.utils.data.DataLoader(ds, batch_size=16, shuffle=True, generator=gen, collate_fn=geo.collate)
    timings, snapshots, scene_ids = [], [], []
    torch.cuda.reset_peak_memory_stats()
    for step, batch in enumerate(loader, 1):
        if step > 200:
            break
        scene_ids.append(batch.sample_index.tolist())
        batch = batch.to('cuda'); torch.cuda.synchronize(); started = time.perf_counter()
        update(model, optimizer, batch, stats)
        torch.cuda.synchronize(); timings.append(time.perf_counter()-started)
        if step == 8:
            resume_batch = cpu_copy(vars(batch), torch)
            restore_state = cpu_copy({'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                'scheduler': schedule.state_dict(), 'random': random.getstate(), 'numpy': np.random.get_state(),
                'torch': torch.get_rng_state(), 'cuda': torch.cuda.get_rng_state_all(), 'loader': gen.get_state()}, torch)
            checkpoint = OUT/f'qualification_checkpoint_{time.time_ns()}.pt'
            save_checkpoint(checkpoint, restore_state)
            expected_update = update(model, optimizer, batch, stats, keep=True)
            expected_state = complete_state(model, optimizer, schedule, gen)
            del model, optimizer, schedule
            gc.collect(); torch.cuda.empty_cache()
            model = geo.model().cuda()
            restored = torch.load(checkpoint, map_location='cpu', weights_only=False)
            model.load_state_dict(restored['model'])
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
            optimizer.load_state_dict(restored['optimizer'])
            schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 200, eta_min=1e-7)
            schedule.load_state_dict(restored['scheduler'])
            random.setstate(restored['random']); np.random.set_state(restored['numpy'])
            torch.set_rng_state(restored['torch']); torch.cuda.set_rng_state_all(restored['cuda'])
            gen.set_state(restored['loader'])
            actual_update = update(model, optimizer, batch, stats, keep=True)
            restore_error = max_difference(expected_update, actual_update, torch)
            restore_state_error = max_difference(expected_state,
                complete_state(model, optimizer, schedule, gen), torch)
            if max(restore_error, restore_state_error) > 2e-5:
                raise RuntimeError('Next-update checkpoint restoration failed')
            del restored, restore_state, resume_batch, expected_update, expected_state, actual_update
        del batch
        if step % 25 == 0:
            torch.cuda.empty_cache()
            value = {'step': step, **require_resources()}
            snapshots.append(value)
            print(json.dumps({'event': 'geo_strong_qualification', **value}), flush=True)
    # Use the same scene IDs and RNG semantics as a clean seed-0 loader.
    control_gen = torch.Generator().manual_seed(0)
    order = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(torch.arange(len(ds))),
        batch_size=16, shuffle=True, generator=control_gen)
    if [v[0].tolist() for i, v in enumerate(order) if i < 200] != scene_ids:
        raise RuntimeError('Actual seed0 data order changed')
    complete_gen = torch.Generator().manual_seed(0)
    complete = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(torch.arange(len(ds))),
        batch_size=16, shuffle=True, generator=complete_gen)
    list(complete)
    restored_gen = torch.Generator(); restored_gen.set_state(complete_gen.get_state())
    next_order = torch.utils.data.DataLoader(complete.dataset, batch_size=16, shuffle=True, generator=restored_gen)
    if not torch.equal(torch.cat([v[0] for v in complete]), torch.cat([v[0] for v in next_order])):
        raise RuntimeError('Full-epoch loader RNG restoration failed')
    n8 = geo.Scenes(geo.SPLITS['n8_test'])
    batch = geo.collate([n8[i] for i in range(16)]).to('cuda')
    model.eval()
    with torch.no_grad():
        predicted = model(batch)
        if predicted.shape != (16384, 2) or not torch.isfinite(predicted).all():
            raise RuntimeError('N8/batch16 output alignment failed')
    update(model, optimizer, batch, stats)
    torch.cuda.synchronize(); resources = require_resources()
    result = {'status': 'passed', 'identity': identity(), 'interface': interface,
        'original_encoder_max_abs': encoder_error, 'field_forward_leakage_max_abs': leakage,
        'scene_isolation_max_abs': isolation, 'checkpoint_next_update_max_abs': restore_error,
        'checkpoint_model_optimizer_max_abs': restore_state_error, 'full_epoch_loader_restore': True,
        'seed0_loader_order_exact': True, 'actual_training_batches': 200, 'batch_size': 16,
        'n8_batch16_output_shape': list(predicted.shape), 'parameters': sum(p.numel() for p in model.parameters()),
        'warm_update_seconds_mean': float(np.mean(timings[20:])),
        'peak_gpu_allocated_bytes': torch.cuda.max_memory_allocated(), 'resource_snapshots': snapshots,
        'n8_resources': resources, 'qualification_deterministic_only': True,
        'scientific_release': False, 'passed_at': time.time()}
    write_json(OUT/'erwin_preflight.json', result)
    print(json.dumps({'event': 'geo_strong_preflight_passed', 'warm_update_seconds_mean': result['warm_update_seconds_mean'],
                      'peak_gpu_allocated_bytes': result['peak_gpu_allocated_bytes']}), flush=True)


def train():
    receipt = read_json(OUT/'erwin_preflight.json')
    if not receipt or receipt['status'] != 'passed' or receipt['identity'] != identity():
        raise RuntimeError('Missing matching Geo-strong qualification')
    install_cleanup()
    frozen.train_method('erwin')


def main():
    parser = argparse.ArgumentParser()
    phase = parser.add_mutually_exclusive_group(required=True)
    phase.add_argument('--qualify', action='store_true')
    phase.add_argument('--train', action='store_true')
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    import msvcrt
    with (OUT/'worker.lock').open('a+b') as lock:
        lock.seek(0); msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        try:
            configure()
            qualification() if args.qualify else train()
        except BaseException as error:
            write_json(OUT/f'failure_{time.time_ns()}.json', {'pid': os.getpid(),
                'phase': 'qualify' if args.qualify else 'train', 'error': repr(error),
                'traceback': traceback.format_exc()})
            raise


if __name__ == '__main__':
    main()
