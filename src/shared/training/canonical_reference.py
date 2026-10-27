"""Restartable PTv3/Erwin training with selectable, explicit input adapters."""
import gc
import json
import os
import random
import time
from shared.point_reference_runtime import np, torch
from shared.paths import ROOT, DATA, CACHE, TESTS
from shared.data_integrity import sha, verify_data
from shared.io import read_json, write_json, save_checkpoint as save
from shared.learning import loss_and_scene_metrics, field_loss_and_scene_metrics
from torch.utils.data import ConcatDataset, DataLoader

EPOCHS, BATCH_SIZE, LAMBDA_Q, SEED, RESUME = 200, 16, .1, 0, False

def seed_all():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

def identity(method):
    return {'method': method, 'input': INPUT, 'epochs': EPOCHS, 'batch_size': BATCH_SIZE,
            'seed': SEED, 'lambda_q': LAMBDA_Q, 'optimizer': 'AdamW',
            'lr': 1e-4, 'min_lr': 1e-7, 'weight_decay': 1e-5, 'gradient_clip': 1.,
            'data_sha256': sha(DATA/'manifest.json'),
            'sources': {p.relative_to(ROOT).as_posix(): sha(p) for p in sorted((ROOT/'src').rglob('*.py'))},
            'selection': 'equal mean of the two validation scattered relative L2 means'}

def verify_inputs():
    verify_data(sizes_only=True)

def setup_data(method):
    stats = read_json(CACHE/'statistics.json')
    phi = None
    if INPUT == 's-z':
        with np.load(CACHE/'shared.npz') as shared:
            phi = torch.tensor(shared['phi_pinv'], device='cuda', dtype=torch.complex64)
    train = ConcatDataset([Scenes(CACHE/s, method) for s in ('train_v3', 'train_v6s11')])
    vals = [Scenes(CACHE/s, method) for s in ('val_v3', 'val_v6s11')]
    return stats, phi, train, vals
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

def train_method(method):
    run_id = identity(method)
    verify_inputs()
    out = RUN_ROOT
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'run_manifest.json').exists() and json.loads((out / 'run_manifest.json').read_text()) != run_id:
        raise RuntimeError('Existing run identity mismatch')
    write_json(out / 'run_manifest.json', run_id)
    seed_all()
    model = (PTv3Reference if method == 'ptv3' else ErwinReference)(method).cuda()
    stats, phi, ds, vals = setup_data(method)
    gen = torch.Generator().manual_seed(SEED)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, generator=gen, collate_fn=collate)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS, eta_min=1e-7)
    history, best, completed = [], float('inf'), 0
    if (out / 'last.pt').exists() and RESUME:
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
            if step % 10 == 0:
                torch.cuda.empty_cache()
            n = len(batch.sample_node_counts)
            total_loss += float(loss.detach()) * n
            seen += n
            if step == 1 or step % 100 == 0:
                state = {'method': method, 'status': 'running', 'pid': os.getpid(), 'epoch': epoch + 1,
                         'completed_epochs': epoch, 'batch': step, 'batches': len(loader), 'scenes_seen': seen,
                         
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
        source = json.loads((DATA / 'test_layouts' / split / 'manifest.json').read_text())
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

def run(args, method):
    global EPOCHS, BATCH_SIZE, LAMBDA_Q, SEED, RESUME, RUN_ROOT, INPUT
    global Scenes, collate, LOSS, PTv3Reference, ErwinReference
    EPOCHS, BATCH_SIZE, SEED, RESUME = args.epochs, args.batch_size, args.seed, args.resume
    RUN_ROOT, INPUT = args.output, args.input
    if INPUT == 'geo-strong':
        if method == 'ptv3':
            import PTv3.Geo_strong.input as adapter
        else:
            import Erwin.Geo_strong.input as adapter
        Scenes, collate = adapter.Scenes, adapter.collate
        PTv3Reference = ErwinReference = adapter.model
    elif INPUT == 'surface':
        from shared.point_surface_only_data import SurfaceScenes, collate as surface_collate
        from PTv3.Surface_point_cloud.model import SurfaceReference as PT
        from Erwin.Surface_point_cloud.model import SurfaceReference as ER
        Scenes, collate, PTv3Reference, ErwinReference = SurfaceScenes, surface_collate, PT, ER
    else:
        from shared.point_reference_data import Scenes as SZScenes, collate as sz_collate
        from PTv3.S_Z.model import PointReference as PT
        from Erwin.S_Z.model import PointReference as ER
        Scenes, collate, PTv3Reference, ErwinReference = SZScenes, sz_collate, PT, ER
    LAMBDA_Q = .1 if INPUT == 's-z' else 0.
    LOSS = loss_and_scene_metrics if INPUT == 's-z' else field_loss_and_scene_metrics
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision('high')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required for training')
    if args.smoke:
        seed_all()
        model = (PTv3Reference if method == 'ptv3' else ErwinReference)(method).cuda()
        stats, phi, ds, _ = setup_data(method)
        for source in ds.datasets:
            batch = collate([source[i] for i in range(BATCH_SIZE)]).to('cuda')
            model.train()
            model.zero_grad(set_to_none=True)
            prediction = model(batch)
            loss, _ = LOSS(prediction, batch, phi, stats, LAMBDA_Q)
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite smoke loss')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            print(json.dumps({'method': method, 'input': INPUT, 'prediction_shape': list(prediction.shape),
                              'loss': float(loss.detach())}), flush=True)
            del batch, prediction, loss
            torch.cuda.empty_cache()
        return
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    if (RUN_ROOT/'last.pt').exists() and not RESUME:
        raise RuntimeError('Checkpoint exists: use --resume or a new --output directory')
    if RESUME and not (RUN_ROOT/'last.pt').is_file():
        raise FileNotFoundError('No complete last.pt to resume')
    train_method(method)
