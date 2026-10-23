"""Independent surface-input comparison, reusing the frozen physical train loop."""
from __future__ import annotations
import argparse
import copy
import gc
import json
import os
from pathlib import Path
import time
import traceback
from point_reference_runtime import ROOT, np, torch
import train_point_references as frozen
from point_surface_data import OUT, SIDE, GRID_SIZE, SurfaceScenes, collate, verify_sidecar
from point_surface_models import SurfaceReference
from point_reference_models import PT_CONFIG, ERWIN_CONFIG
from point_reference_io_recovery import read_json, write_json, save_checkpoint
from prepare_point_reference_tests import sha, CACHE

ORIGINAL_IDENTITY = frozen.identity
ORIGINAL_VERIFY = frozen.verify_inputs
NEW_SOURCES = ['point_surface_data.py', 'point_surface_models.py', 'train_point_surface.py',
               'point_reference_io_recovery.py', 'launch_point_surface.py']


def identity(method):
    base = ORIGINAL_IDENTITY(method)
    base.update(input='actual voxel-sampled surface: 15 known-condition channels',
                surface_manifest_sha256=sha(SIDE/'manifest.json'),
                surface_sources={n: sha(ROOT/'scripts'/n) for n in NEW_SOURCES},
                surface_protocol_sha256=sha(ROOT/'POINT_SURFACE_REFERENCES_20261011.md'),
                test_layout_manifests={s:sha(ROOT/'datasets/canonical_response_v3/data'/s/'manifest.json')
                                       for s in frozen.TESTS},
                model_config={**PT_CONFIG, 'in_channels': 15} if method == 'ptv3'
                             else {**ERWIN_CONFIG, 'c_in': 15},
                grid_size=GRID_SIZE, shared_surface_sampler=True,
                decoder={'query_features': 12, 'neighbors': 64, 'width': 64, 'heads': 4,
                         'relative_xyz': True, 'output': 'original 128*N query scattered field'},
                comparison_limit='representation AND query decoder differ from S(Z) reference')
    return json.loads(json.dumps(base))


def verify_inputs():
    ORIGINAL_VERIFY()
    return verify_sidecar()


def configure():
    # Bind only this new process. None of the frozen files are edited.
    frozen.RUN_ROOT = OUT
    frozen.PointReference = SurfaceReference
    frozen.Scenes = SurfaceScenes
    frozen.collate = collate
    frozen.identity = identity
    frozen.verify_inputs = verify_inputs
    frozen.write_json = write_json
    frozen.save = save_checkpoint


def resource_snapshot():
    import psutil
    free, total = torch.cuda.mem_get_info()
    return {'available_ram_bytes': psutil.virtual_memory().available,
            'worker_rss_bytes': psutil.Process().memory_info().rss,
            'gpu_free_bytes': free, 'gpu_total_bytes': total,
            'gpu_allocated_bytes': torch.cuda.memory_allocated(),
            'gpu_reserved_bytes': torch.cuda.memory_reserved()}


def require_ram():
    snapshot = resource_snapshot()
    if snapshot['available_ram_bytes'] < 1024**3:
        raise RuntimeError(f'System available RAM below 1 GiB reserve: {snapshot}')
    return snapshot


def bind_resident_cache(method):
    """Restrict only this verified worker's resident mmap pages, not tensors/GPU."""
    import ctypes
    from ctypes import wintypes
    import psutil
    process = psutil.Process()
    if 'train_point_surface' not in ' '.join(process.cmdline()):
        raise RuntimeError('Unrelated process: refuse working-set adjustment')
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.GetCurrentProcess.restype = wintypes.HANDLE
    kernel.SetProcessWorkingSetSizeEx.argtypes = [wintypes.HANDLE, ctypes.c_size_t,
                                               ctypes.c_size_t, wintypes.DWORD]
    kernel.SetProcessWorkingSetSizeEx.restype = wintypes.BOOL
    cap = int(4.5*1024**3)
    if not kernel.SetProcessWorkingSetSizeEx(kernel.GetCurrentProcess(), 32*1024**2, cap, 0x4 | 0x2):
        raise ctypes.WinError(ctypes.get_last_error())
    write_json(OUT/f'{method}_working_set_policy.json', {
        'pid': os.getpid(), 'created_at': process.create_time(), 'resident_max_bytes': cap,
        'policy': 'CPU resident mmap cache only; not a virtual, tensor or GPU cap'})


def physical_checks(dataset):
    """Geometry placement/incident phase and common-point identity, independently."""
    max_phase, max_normal, max_coordinate = 0., 0., 0.
    for index in (0, len(dataset)//2, len(dataset)-1):
        item = dataset[index]
        feat, pos, query = (item[k].numpy() for k in ('node_features','positions','query_features'))
        q = query.reshape(-1,128,12)
        direction = q[0,0,8:11].astype(np.float64)
        known0 = complex(q[0,0,0],q[0,0,1])
        origin = item['query_positions'][0].numpy().astype(np.float64)
        expected = known0*np.exp(3j*((pos.astype(np.float64)-origin)@direction))
        actual = feat[:,3]+1j*feat[:,4]
        max_phase = max(max_phase,float(np.max(np.abs(actual-expected))))
        max_normal = max(max_normal,float(np.max(np.abs(np.linalg.norm(feat[:,:3],axis=1)-1))))
        max_coordinate = max(max_coordinate,float(np.max(np.abs(pos-feat[:,5:8]-feat[:,8:11]))))
        if len(np.unique(item['grid'].numpy(),axis=0)) != len(pos):
            raise RuntimeError('Sampled surface grids are not unique')
        other = SurfaceScenes(dataset.folder,'erwin' if dataset.method=='ptv3' else 'ptv3')[index]
        for key in ('node_features','positions','query_features','query_positions','neighbors','grid'):
            if not torch.equal(item[key],other[key]):
                raise RuntimeError(f'Backbones receive different surface/query data: {key}')
    if max_phase > 2e-5 or max_normal > 2e-5 or max_coordinate > 2e-6:
        raise RuntimeError(f'Geometry/plane-wave placement check failed: {max_phase,max_normal,max_coordinate}')
    return {'phase_max_abs':max_phase,'normal_unit_max_abs':max_normal,
            'coordinate_identity_max_abs':max_coordinate,'identical_surface_inputs':True}


def assert_nested_equal(a,b):
    if torch.is_tensor(a):
        # AdamW step counters may be CPU scalars before save and CUDA scalars
        # after map_location. Check exact values, not storage-device identity.
        if not torch.equal(a.detach().cpu(),b.detach().cpu()):
            raise RuntimeError('Optimizer/scheduler tensor restoration differs')
    elif isinstance(a,dict):
        if a.keys()!=b.keys(): raise RuntimeError('Restoration mapping differs')
        for k in a: assert_nested_equal(a[k],b[k])
    elif isinstance(a,(tuple,list)):
        if len(a)!=len(b): raise RuntimeError('Restoration length differs')
        for x,y in zip(a,b): assert_nested_equal(x,y)
    elif a!=b: raise RuntimeError('Restoration scalar differs')


def preflight(method):
    manifest = verify_inputs()
    frozen.seed_all()
    model = SurfaceReference(method).cuda()
    stats, phi, train, val = frozen.setup_data(method)
    checks = [physical_checks(d) for d in [*train.datasets,*val,
               SurfaceScenes(frozen.TESTS['n8_test'],method)]]
    indices = list(range(8))+list(range(10000,10008))
    batch = collate([train[i] for i in indices]).to('cuda')
    require_ram()
    model.eval()
    with torch.no_grad():
        first = model(batch)
        repeated = model(batch)
        deterministic = float((first-repeated).abs().max())
        backups = {k:getattr(batch,k) for k in ('labels','q64_gt','z_real','z_imag','incident')}
        for k,v in backups.items(): setattr(batch,k,torch.randn_like(v))
        leakage = float((first-model(batch)).abs().max())
        for k,v in backups.items(): setattr(batch,k,v)
        original = batch.node_features
        batch.node_features = original.clone()
        batch.node_features[batch.surface_batch_index!=0] += .123
        isolated = model(batch)
        isolation = float((first[batch.batch_index==0]-isolated[batch.batch_index==0]).abs().max())
        batch.node_features = original
        order = torch.randperm(len(batch.query_positions),device='cuda')
        qbackup = {k:getattr(batch,k) for k in ('query_features','query_positions','neighbors')}
        for k,v in qbackup.items(): setattr(batch,k,v[order])
        ordering = float((model(batch)-first[order]).abs().max())
        for k,v in qbackup.items(): setattr(batch,k,v)
    if leakage!=0 or max(deterministic,isolation,ordering)>2e-5 or not torch.isfinite(first).all():
        raise RuntimeError(f'Forward qualification failed: {leakage,deterministic,isolation,ordering}')
    opt = torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt,frozen.EPOCHS,eta_min=1e-7)
    torch.cuda.reset_peak_memory_stats()
    timings=[]; model.train()
    for i in range(8):
        require_ram(); torch.cuda.synchronize(); start=time.perf_counter()
        opt.zero_grad(set_to_none=True)
        loss,_ = frozen.LOSS(model(batch),batch,phi,stats,.1)
        if not torch.isfinite(loss): raise RuntimeError('Nonfinite qualification loss')
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
        opt.step();torch.cuda.synchronize();timings.append(time.perf_counter()-start)
        print(json.dumps({'event':'surface_preflight_step','method':method,'step':i+1,
                          'loss':float(loss.detach()),'seconds':timings[-1]}),flush=True)
    scheduler.step()
    # The largest geometry batches must fit without quietly changing batch16.
    largest=[]
    for offset,dataset in zip((0,10000),train.datasets):
        n=np.diff(dataset.surface_offsets)
        largest.extend(offset+int(i) for i in np.argsort(n,kind='stable')[-8:])
    del first,repeated,isolated
    worst = collate([train[i] for i in largest]).to('cuda')
    opt.zero_grad(set_to_none=True)
    worst_loss,_ = frozen.LOSS(model(worst),worst,phi,stats,.1)
    worst_loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
    opt.step(); del worst,worst_loss
    torch.cuda.synchronize()
    training_peak=torch.cuda.max_memory_allocated()
    model.eval()
    with torch.no_grad(): expected=model(batch).cpu()
    checkpoint=OUT/f'{method}_preflight_checkpoint_{time.time_ns()}.pt'
    save_checkpoint(checkpoint,{'model':model.state_dict(),'optimizer':opt.state_dict(),
                                'scheduler':scheduler.state_dict()})
    optimizer_expected=copy.deepcopy(opt.state_dict());scheduler_expected=copy.deepcopy(scheduler.state_dict())
    del model,opt,scheduler;gc.collect();torch.cuda.empty_cache()
    restored=SurfaceReference(method).cuda()
    loaded=torch.load(checkpoint,map_location='cuda',weights_only=False)
    restored.load_state_dict(loaded['model'],strict=True)
    restored_opt=torch.optim.AdamW(restored.parameters(),lr=1e-4,weight_decay=1e-5)
    restored_opt.load_state_dict(loaded['optimizer'])
    restored_scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(restored_opt,frozen.EPOCHS,eta_min=1e-7)
    restored_scheduler.load_state_dict(loaded['scheduler'])
    assert_nested_equal(optimizer_expected,restored_opt.state_dict())
    assert_nested_equal(scheduler_expected,restored_scheduler.state_dict())
    restored.eval()
    with torch.no_grad(): restore_error=float((expected-restored(batch).cpu()).abs().max())
    if restore_error>2e-5: raise RuntimeError('Checkpoint output restoration failed')
    del loaded,optimizer_expected,scheduler_expected,restored_opt,restored_scheduler,batch
    gc.collect();torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
    test=SurfaceScenes(frozen.TESTS['n8_test'],method)
    n8batch=collate([test[int(i)] for i in np.argsort(np.diff(test.surface_offsets),kind='stable')[-16:]]).to('cuda')
    with torch.no_grad(): n8=restored(n8batch)
    if n8.shape!=(16*1024,2) or not torch.isfinite(n8).all():
        raise RuntimeError('N8 canonical output alignment failed')
    eval_peak=torch.cuda.max_memory_allocated()
    del n8,n8batch;gc.collect();torch.cuda.empty_cache()
    t0=time.perf_counter()
    for start in range(0,160,16): collate([train.datasets[0][i] for i in range(start,start+16)])
    loading=(time.perf_counter()-t0)/10
    result={'status':'passed','identity':identity(method),'scientific_release':False,
            'parameters':sum(p.numel() for p in restored.parameters()),'physical_interface_checks':checks,
            'forward_leakage_max_abs':leakage,'deterministic_max_abs':deterministic,
            'scene_isolation_max_abs':isolation,'query_permutation_max_abs':ordering,
            'checkpoint_restore_max_abs':restore_error,'optimizer_and_scheduler_exact':True,
            'tested_train_scenes':indices,'largest_train_scenes':largest,
            'largest_n8_batch_shape':[16*1024,2], 'surface_counts':manifest['splits'],
            'step_seconds':timings,'warm_step_mean':float(np.mean(timings[3:])),
            'batch_load_seconds':loading,'training_peak_gpu_bytes':training_peak,
            'n8_evaluation_peak_gpu_bytes':eval_peak,'resource_snapshot':require_ram(),
            'estimated_train_hours_excluding_validation':(float(np.mean(timings[3:]))+loading)*1250*200/3600}
    write_json(OUT/f'{method}_preflight.json',result)
    print(json.dumps({'event':'surface_preflight_passed',**{k:v for k,v in result.items() if k not in ('identity','surface_counts')}}),flush=True)


def train(method):
    import msvcrt
    folder=OUT/'runs'/method;folder.mkdir(parents=True,exist_ok=True)
    with (OUT/f'{method}.lock').open('a+b') as lock:
        if lock.tell()==0: lock.write(b'1');lock.flush()
        lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
        if (read_json(folder/'status.json') or {}).get('status')=='complete':
            raise RuntimeError('This completed model is immutable; refuse rerun')
        bind_resident_cache(method);require_ram()
        # Guard snapshots are external telemetry; model/loss/train order unchanged.
        original_write=frozen.write_json
        def guarded_write(path,value):
            if Path(path).name=='status.json':
                snapshot=require_ram()
                write_json(folder/'resources.json',{'utc':time.time(),**snapshot})
            original_write(path,value)
        frozen.write_json=guarded_write
        frozen.train_method(method)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--method',choices=['ptv3','erwin'],required=True)
    parser.add_argument('--preflight',action='store_true');args=parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True);configure()
    torch.set_num_threads(4);torch.set_float32_matmul_precision('high')
    if not torch.cuda.is_available(): raise RuntimeError('Local CUDA unavailable')
    try:
        if args.preflight: preflight(args.method)
        else: train(args.method)
    except BaseException as error:
        failure=OUT/f'{args.method}_failure_{time.time_ns()}.json'
        write_json(failure,{'phase':'preflight' if args.preflight else 'train','method':args.method,
                           'pid':os.getpid(),'error':repr(error),'traceback':traceback.format_exc()})
        raise


if __name__=='__main__': main()
