"""Linux execution-only adapter around frozen surface models and train loop."""
from __future__ import annotations
import argparse
import ast
import contextlib
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess  # Load before compatibility modules; never fake Windows.
import sys
import time
import types

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/point_surface_autodl_20261012'
os.environ.setdefault('PYTORCH_ALLOC_CONF','expandable_segments:True')

def install_io():
    path=ROOT/'scripts/point_reference_io_recovery.py'
    module=types.ModuleType('point_reference_io_recovery')
    module.__file__=str(path)
    module.__dict__.update(Path=Path,os=os,time=time,json=json,RETRY_SECONDS=5.)
    tree=ast.parse(path.read_text())
    names={'replace_retry','write_json','save_checkpoint','open_shared','read_json'}
    nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names]
    if {n.name for n in nodes}!=names: raise RuntimeError('I/O leaf source changed')
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[])),str(path),'exec'),module.__dict__)
    sys.modules[module.__name__]=module

def effective_available():
    import psutil
    available=psutil.virtual_memory().available
    for limit,current in ((Path('/sys/fs/cgroup/memory.max'),Path('/sys/fs/cgroup/memory.current')),
                          (Path('/sys/fs/cgroup/memory/memory.limit_in_bytes'),Path('/sys/fs/cgroup/memory/memory.usage_in_bytes'))):
        if limit.exists() and current.exists():
            value=limit.read_text().strip()
            if value!='max': available=min(available,max(0,int(value)-int(current.read_text())))
    return available

def configure():
    install_io()
    import balltree  # Native Linux binding; unchanged saved trees used in forward.
    from point_reference_runtime import torch
    import train_point_surface as surface
    from prepare_point_reference_tests import sha
    surface.configure()
    surface.OUT=OUT
    surface.frozen.RUN_ROOT=OUT
    original_identity=surface.identity
    def identity(method):
        value=original_identity(method)
        value.update(remote_execution={
            'adapter_sha256':sha(Path(__file__)),
            'supervisor_sha256':sha(ROOT/'scripts/supervise_remote_point_surface.py'),
            'protocol_sha256':sha(ROOT/'POINT_SURFACE_AUTODL_20261012.md'),
            'inventory_sha256':sha(OUT/'transfer_inventory.json'),
            'allocator':os.environ['PYTORCH_ALLOC_CONF'],
            'cache_cleanup':'unused CUDA cache released at telemetry boundaries',
            'native_balltree_sha256':sha(Path(balltree.balltree.__file__)),
            'packages':{n:importlib.metadata.version(n) for n in
                        ('torch','numpy','scipy','spconv-cu126','torch-scatter','einops','psutil','balltree-erwin')}})
        return json.loads(json.dumps(value))
    surface.identity=identity
    surface.frozen.identity=identity
    def require_ram():
        value=surface.resource_snapshot()
        value['available_ram_bytes']=effective_available()
        if value['available_ram_bytes']<1024**3:
            raise RuntimeError(f'Effective available RAM below 1 GiB: {value}')
        return value
    surface.require_ram=require_ram
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision('high')
    if not torch.cuda.is_available(): raise RuntimeError('CUDA runtime unavailable')
    return surface

def verify_transfer():
    from prepare_point_reference_tests import sha
    inventory=json.loads((OUT/'transfer_inventory.json').read_text())
    if inventory['status']!='prepared': raise RuntimeError('Missing transfer binding')
    for rel,record in inventory['files'].items():
        p=ROOT/rel
        if p.stat().st_size!=record['size'] or sha(p)!=record['sha256']:
            raise RuntimeError(f'Remote transfer mismatch: {rel}')
    return len(inventory['files'])

def qualify_extra(s,method):
    """Vary sizes, unlike the repeated fixed batch in the original preflight."""
    torch=s.torch; np=s.np
    s.frozen.seed_all()
    model=s.SurfaceReference(method).cuda()
    stats,phi,train,_=s.frozen.setup_data(method)
    opt=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=1e-5)
    first=s.collate([train[i] for i in list(range(8))+list(range(10000,10008))]).to('cuda')
    model.eval()
    with torch.no_grad(): expected=model(first).clone()
    torch.cuda.empty_cache()
    with torch.no_grad(): cleanup_error=float((expected-model(first)).abs().max())
    if cleanup_error>2e-5: raise RuntimeError('Allocator cleanup changed outputs')
    opt.zero_grad(set_to_none=True)
    loss,_=s.frozen.LOSS(model(first),first,phi,stats,.1)
    loss.backward()
    gradients=[p.grad.clone() for p in model.parameters() if p.grad is not None]
    opt.zero_grad(set_to_none=True);torch.cuda.empty_cache()
    loss2,_=s.frozen.LOSS(model(first),first,phi,stats,.1);loss2.backward()
    actual=[p.grad for p in model.parameters() if p.grad is not None]
    gradient_error=max(float((a-b).abs().max()) for a,b in zip(gradients,actual))
    if gradient_error>2e-5 or len(gradients)!=len(actual):raise RuntimeError('Cleanup gradient parity failed')
    del expected,gradients,actual,first,loss,loss2
    tree_checks=[]
    if method=='erwin':
        from balltree import build_balltree_with_rotations
        for i in (0,10000,19999):
            item=train[i]
            generated=build_balltree_with_rotations(item['positions'],torch.zeros(len(item['positions']),dtype=torch.long),
                s.ERWIN_CONFIG['strides'],s.ERWIN_CONFIG['ball_sizes'],s.ERWIN_CONFIG['rotate'])
            cached=item['tree']
            equal=torch.equal(generated[0],cached[0]) and torch.equal(generated[1],cached[1]) and all(
                torch.equal(a,b) for a,b in zip(generated[2],cached[2]))
            if not equal:raise RuntimeError('Native Linux and saved official trees differ')
            tree_checks.append(i)
    model.train();torch.cuda.reset_peak_memory_stats()
    rng=np.random.default_rng(91821);snapshots=[];start=time.perf_counter()
    for step in range(200):
        indices=rng.choice(len(train),size=16,replace=False)
        batch=s.collate([train[int(i)] for i in indices]).to('cuda')
        opt.zero_grad(set_to_none=True)
        loss,_=s.frozen.LOSS(model(batch),batch,phi,stats,.1)
        if not torch.isfinite(loss):raise RuntimeError('Nonfinite memory-soak loss')
        loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);opt.step()
        if (step+1)%25==0:
            torch.cuda.synchronize();before=s.require_ram();torch.cuda.empty_cache()
            snapshots.append({'step':step+1,**before,'reserved_after_cleanup':torch.cuda.memory_reserved()})
            print(json.dumps({'event':'variable_batch_soak','method':method,**snapshots[-1]}),flush=True)
    result={'status':'passed','identity':s.identity(method),'batches':200,'batch_size':16,
            'cleanup_prediction_max_abs':cleanup_error,'cleanup_gradient_max_abs':gradient_error,
            'native_tree_checked_scenes':tree_checks,'snapshots':snapshots,
            'peak_gpu_allocated_bytes':torch.cuda.max_memory_allocated(),'seconds':time.perf_counter()-start}
    s.write_json(OUT/f'{method}_memory_qualification.json',result)

def train(s,method):
    import fcntl
    OUT.mkdir(parents=True,exist_ok=True)
    with (OUT/f'{method}.lock').open('a+b') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        extra=s.read_json(OUT/f'{method}_memory_qualification.json')
        if not extra or extra['status']!='passed' or extra['identity']!=s.identity(method):
            raise RuntimeError('Missing source-bound variable-batch qualification')
        folder=OUT/'runs'/method;folder.mkdir(parents=True,exist_ok=True)
        if (s.read_json(folder/'status.json') or {}).get('status')=='complete':
            raise RuntimeError('Completed run is immutable')
        original=s.frozen.write_json
        def guarded(path,value):
            if Path(path).name=='status.json':
                s.torch.cuda.empty_cache()
                s.write_json(folder/'resources.json',{'utc':time.time(),**s.require_ram()})
            original(path,value)
        s.frozen.write_json=guarded
        s.frozen.train_method(method)

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--method',choices=['ptv3','erwin'],required=True)
    parser.add_argument('--preflight',action='store_true')
    args=parser.parse_args();OUT.mkdir(parents=True,exist_ok=True)
    s=configure()
    try:
        verify_transfer()
        if args.preflight:
            s.preflight(args.method)
            qualify_extra(s,args.method)
        else:train(s,args.method)
    except BaseException as error:
        import traceback
        s.write_json(OUT/f'{args.method}_failure_{time.time_ns()}.json',{
            'method':args.method,'preflight':args.preflight,'pid':os.getpid(),
            'error':repr(error),'traceback':traceback.format_exc()})
        raise

if __name__=='__main__':main()
