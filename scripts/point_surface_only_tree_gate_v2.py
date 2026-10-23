"""Versioned, fail-closed qualification for the cached Erwin trees actually used."""
from __future__ import annotations

import json
import time


def _validate_tree(tree, point_count, strides):
    torch = __import__('torch')
    if tree is None or len(tree) != 3:
        raise RuntimeError('Erwin scene is missing its saved tree tuple')
    index, mask, rotations = tree
    padded = 1 << (int(point_count) - 1).bit_length()
    if index.ndim != 1 or mask.ndim != 1 or len(index) != padded or len(mask) != padded:
        raise RuntimeError('Cached tree has an invalid padded shape')
    if mask.dtype != torch.bool or int(mask.sum()) != int(point_count):
        raise RuntimeError('Cached tree mask does not preserve every real surface point')
    if not torch.equal(torch.sort(index[mask].to(torch.int64)).values,
                       torch.arange(point_count, dtype=torch.int64)):
        raise RuntimeError('Cached tree indices are not a permutation of real points')
    if len(rotations) != len(strides) + 1:
        raise RuntimeError('Cached rotation-stage count differs from the model')
    for stage, rotation in enumerate(rotations):
        expected = padded // (2 ** stage)
        if rotation.ndim != 1 or len(rotation) != expected:
            raise RuntimeError(f'Cached rotation stage {stage} has the wrong length')
        values = rotation.to(torch.int64)
        if not torch.equal(torch.sort(values).values, torch.arange(expected, dtype=torch.int64)):
            raise RuntimeError(f'Cached rotation stage {stage} is not a permutation')
    return padded


def _validate_collated_trees(items, tree, strides):
    torch = __import__('torch')
    if tree is None:
        raise RuntimeError('Erwin collator dropped the saved trees')
    indexes, masks, rotations = tree
    point_offset = 0
    rotation_offsets = [0] * len(rotations)
    cursor = 0
    for item in items:
        n = len(item['positions'])
        padded = _validate_tree(item['tree'], n, strides)
        local_index, local_mask, local_rotations = item['tree']
        if not torch.equal(indexes[cursor:cursor+padded], local_index + point_offset):
            raise RuntimeError('Collation changed a cached tree index or mixed scenes')
        if not torch.equal(masks[cursor:cursor+padded], local_mask):
            raise RuntimeError('Collation changed a cached tree mask')
        for stage, rotation in enumerate(local_rotations):
            lo = rotation_offsets[stage]
            hi = lo + len(rotation)
            if not torch.equal(rotations[stage][lo:hi], rotation + lo):
                raise RuntimeError(f'Collation changed rotation stage {stage} or crossed scenes')
            rotation_offsets[stage] = hi
        point_offset += n
        cursor += padded
    if cursor != len(indexes) or any(offset != len(rotations[i]) for i, offset in enumerate(rotation_offsets)):
        raise RuntimeError('Collated tree offsets do not cover the batch exactly')


def _native_diagnostics(s, train):
    """Observe (do not gate on) implementation-defined within-ball order."""
    torch = s.torch
    from balltree import build_balltree_with_rotations
    records = []
    for index in (0, 5000, 9999, 10000, 15000, 19999):
        item = train[index]
        cached = item['tree']
        generated = build_balltree_with_rotations(
            item['positions'], torch.zeros(len(item['positions']), dtype=torch.long),
            s.ERWIN_CONFIG['strides'], s.ERWIN_CONFIG['ball_sizes'], s.ERWIN_CONFIG['rotate'])
        _validate_tree(generated, len(item['positions']), s.ERWIN_CONFIG['strides'])
        records.append({
            'scene_index': index,
            'point_count': len(item['positions']),
            'cached_native_index_exact': bool(torch.equal(cached[0], generated[0])),
            'cached_native_mask_exact': bool(torch.equal(cached[1], generated[1])),
            'cached_native_rotation_exact': [bool(torch.equal(a, b)) for a, b in zip(cached[2], generated[2])],
            'interpretation': 'diagnostic only: training supplies the saved cache; it does not call the native builder'})
    return records


def qualify_extra(s, method):
    """Run the original 200-step soak; validate the exact tree path used at runtime."""
    torch, np = s.torch, s.np
    s.frozen.seed_all()
    model = s.SurfaceReference(method).cuda()
    stats, phi, train, _ = s.frozen.setup_data(method)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    index_set = list(range(8)) + list(range(10000, 10008))
    items = [train[i] for i in index_set]
    cached_order_diagnostics = []
    if method == 'erwin':
        for item in items:
            _validate_tree(item['tree'], len(item['positions']), s.ERWIN_CONFIG['strides'])
        native = _native_diagnostics(s, train)
    else:
        native = []

    first_cpu = s.collate(items)
    if method == 'erwin':
        _validate_collated_trees(items, first_cpu.tree, s.ERWIN_CONFIG['strides'])
    first = first_cpu.to('cuda')

    explicit_tree_calls = {'count': 0}
    if method == 'erwin':
        original_forward = model.core.forward
        def require_saved_tree(*args, **kwargs):
            idx = kwargs.get('tree_idx')
            mask = kwargs.get('tree_mask')
            rotations = kwargs.get('tree_idx_rot')
            if idx is None or mask is None or rotations is None:
                raise RuntimeError('Erwin attempted to rebuild instead of consume the saved tree cache')
            if len(rotations) != len(s.ERWIN_CONFIG['strides']) + 1:
                raise RuntimeError('Erwin did not receive every saved rotation stage')
            explicit_tree_calls['count'] += 1
            return original_forward(*args, **kwargs)
        model.core.forward = require_saved_tree

    model.eval()
    with torch.no_grad():
        expected = model(first).clone()
    torch.cuda.empty_cache()
    with torch.no_grad():
        repeated = model(first)
        cleanup_error = float((expected - repeated).abs().max())
    if cleanup_error > 2e-5 or not torch.isfinite(expected).all():
        raise RuntimeError(f'Cached-input repeatability failed: {cleanup_error}')

    optimizer.zero_grad(set_to_none=True)
    loss_a, _ = s.frozen.LOSS(model(first), first, phi, stats, 0.)
    loss_a.backward()
    grads_a = [p.grad.detach().cpu().clone() for p in model.parameters() if p.grad is not None]
    optimizer.zero_grad(set_to_none=True)
    loss_b, _ = s.frozen.LOSS(model(first), first, phi, stats, 0.)
    loss_b.backward()
    grads_b = [p.grad.detach().cpu() for p in model.parameters() if p.grad is not None]
    if len(grads_a) != len(grads_b):
        raise RuntimeError('Cached-input gradient parameter set changed')
    gradient_error = max((float((a-b).abs().max()) for a, b in zip(grads_a, grads_b)), default=0.)
    if gradient_error > 2e-5:
        raise RuntimeError(f'Cached-input repeat gradient check failed: {gradient_error}')
    del grads_a, grads_b, expected, repeated, loss_a, loss_b, first_cpu, items
    optimizer.zero_grad(set_to_none=True)
    torch.cuda.empty_cache()

    model.train()
    torch.cuda.reset_peak_memory_stats()
    rng = np.random.default_rng(91821)
    snapshots = []
    point_counts = []
    started = time.perf_counter()
    for step in range(200):
        indices = rng.choice(len(train), size=16, replace=False)
        batch_items = [train[int(i)] for i in indices]
        point_counts.extend(len(item['positions']) for item in batch_items)
        cpu_batch = s.collate(batch_items)
        if method == 'erwin':
            for item in batch_items:
                _validate_tree(item['tree'], len(item['positions']), s.ERWIN_CONFIG['strides'])
            _validate_collated_trees(batch_items, cpu_batch.tree, s.ERWIN_CONFIG['strides'])
        batch = cpu_batch.to('cuda')
        optimizer.zero_grad(set_to_none=True)
        loss, _ = s.frozen.LOSS(model(batch), batch, phi, stats, 0.)
        if not torch.isfinite(loss):
            raise RuntimeError('Nonfinite field-only variable-batch soak loss')
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
        if (step + 1) % 25 == 0:
            torch.cuda.synchronize()
            before = s.require_ram()
            torch.cuda.empty_cache()
            snapshot = {'step': step + 1, **before,
                        'reserved_after_cleanup': torch.cuda.memory_reserved()}
            snapshots.append(snapshot)
            print(json.dumps({'event': 'field_only_variable_batch_soak', 'method': method, **snapshot}), flush=True)
        del batch, cpu_batch, batch_items, loss
    result = {
        'status': 'passed', 'identity': s.identity(method), 'batches': 200,
        'batch_size': 16, 'lambda_q': 0,
        'cached_tree_runtime_contract': method == 'erwin',
        'tree_native_order_diagnostics': native,
        'tree_order_rule': 'cached trees are supplied to the unchanged Erwin forward; native index-order mismatch is diagnostic only',
        'explicit_cached_tree_forward_calls': explicit_tree_calls['count'],
        'cached_prediction_repeat_max_abs': cleanup_error,
        'cached_gradient_repeat_max_abs': gradient_error,
        'surface_point_count_range': [int(min(point_counts)), int(max(point_counts))],
        'snapshots': snapshots,
        'peak_gpu_allocated_bytes': torch.cuda.max_memory_allocated(),
        'seconds': time.perf_counter() - started,
    }
    if method == 'erwin' and explicit_tree_calls['count'] < 2:
        raise RuntimeError('Runtime did not exercise explicit cached-tree inputs')
    s.write_json(s.OUT / f'{method}_memory_qualification.json', result)
    print(json.dumps({'event': 'field_only_memory_qualification_passed', 'method': method,
                      'batches': 200, 'peak_gpu_allocated_bytes': result['peak_gpu_allocated_bytes'],
                      'seconds': result['seconds']}), flush=True)
