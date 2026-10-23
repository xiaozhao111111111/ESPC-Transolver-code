"""Direct actual-surface tokens; physical arrays remain auxiliary-only.

Compact immutable sidecars reuse completed labels/clouds. No PDE solve occurs.
"""
from __future__ import annotations
import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import time
from point_reference_runtime import ROOT, np, torch, load_official_erwin
from point_reference_data import FIELDS
from point_reference_models import ERWIN_CONFIG
from point_reference_io_recovery import read_json, write_json
from prepare_point_reference_tests import CACHE
from train_point_references import TESTS, verify_inputs as verify_physical_inputs
from prepare_point_reference_tests import sha
from torch.utils.data import Dataset

OUT = ROOT / 'outputs/point_surface_20k_20261011'
SIDE = OUT / 'geometry_v3'
POINTS, K_NEIGHBORS, GRID_SIZE = 2048, 64, 0.02
SPLITS = {**{s: CACHE/s for s in ('train_v3','train_v6s11','val_v3','val_v6s11')}, **TESTS}
TORUS = TESTS['topology_ood'].parent


def old_id(obj):
    return 'old:' + str(obj['library_id']) + ':' + obj['shape'] + ':' + ','.join(format(float(x), '.17g') for x in obj['size'])


def source_binding():
    paths = [CACHE/'manifest.json', CACHE/'statistics.json', CACHE/'shared.npz',
             CACHE/'geometry_library.json', CACHE/'geometry_points.npy', CACHE/'geometry_normals.npy',
             TORUS/'manifest.json', TORUS/'geometry_library.json', TORUS/'geometry_points.npy', TORUS/'geometry_normals.npy']
    for split in SPLITS:
        folder = SPLITS[split]
        paths += [folder/'object_offsets.npy', folder/'records.json']
        if (folder/'manifest.json').exists():
            paths.append(folder/'manifest.json')
        if (folder/'geometry_index.npy').exists():
            paths += [folder/'geometry_index.npy',folder/'rotations.npy']
    return {str(p.relative_to(ROOT)).replace('\\','/'):sha(p) for p in paths}


def library():
    ids, pts, nrm = [], [], []
    for folder in (CACHE,TORUS):
        meta = read_json(folder/'geometry_library.json')
        names = meta['ids'] if isinstance(meta,dict) else [old_id(r) for r in meta]
        p,n = (np.load(folder/f'geometry_{name}.npy',allow_pickle=False) for name in ('points','normals'))
        for name,pp,nn in zip(names,p,n):
            if name in ids:
                j=ids.index(name)
                if not np.array_equal(pts[j],pp) or not np.array_equal(nrm[j],nn):
                    raise RuntimeError('Shared body cloud differs between completed caches')
            else:
                ids.append(name);pts.append(pp);nrm.append(nn)
    p,n=np.stack(pts).astype(np.float32),np.stack(nrm).astype(np.float32)
    if p.shape[1:] != (POINTS,3) or n.shape != p.shape or not np.isfinite(p).all() or not np.allclose(np.linalg.norm(n,axis=-1),1,atol=2e-5):
        raise RuntimeError('Invalid actual surface cloud/normal library')
    return ids,p,n


def placed(points,normals,gids,rotations,context):
    """No S/Z/labels accepted. Context contains only known query conditions."""
    n=len(gids);c=context.reshape(n,128,12)
    r=np.asarray(rotations,np.float32)
    local=np.einsum('opj,okj->opk',points[gids],r)/np.float32(1.5)
    normal=np.einsum('opj,okj->opk',normals[gids],r)
    centers=c[:,0,5:8];direction=c[:,0,8:11]
    pos=local+centers[:,None]
    qpos=c[:,:,2:5]+centers[:,None]
    inc0=c[:,0,0]+1j*c[:,0,1]
    phase=np.sum((local-c[:,0,None,2:5])*direction[:,None,:],axis=-1)*np.float32(3.0)
    inc=inc0[:,None]*np.exp(1j*phase)
    f=np.empty((n,POINTS,15),np.float32)
    f[...,:3]=normal;f[...,3]=inc.real;f[...,4]=inc.imag;f[...,5:8]=local
    f[...,8:11]=centers[:,None,:];f[...,11:14]=direction[:,None,:];f[...,14]=c[:,0,None,11]
    return f.reshape(-1,15),pos.reshape(-1,3),qpos.reshape(-1,3)


def surface_sample(pos):
    grid=np.floor((pos-pos.min(0))/GRID_SIZE).astype(np.int32)
    _,inverse=np.unique(grid,axis=0,return_inverse=True)
    centers=(grid.astype(np.float64)+0.5)*GRID_SIZE+pos.min(0)
    distance=np.sum((pos-centers)**2,axis=1)
    order=np.lexsort((np.arange(len(pos)),distance,inverse))
    first=np.r_[True,inverse[order][1:]!=inverse[order][:-1]]
    chosen=np.sort(order[first])
    if len(np.unique(grid[chosen],axis=0))!=len(chosen):raise RuntimeError('Voxel sampler collision')
    return chosen.astype(np.uint16),grid[chosen]


def prepare():
    SIDE.mkdir(parents=True,exist_ok=True)
    verify_physical_inputs()
    identity={'sources':source_binding(),'preparer':sha(Path(__file__)),
              'points_per_body':POINTS,'query_neighbors':K_NEIGHBORS,'surface_grid_size':GRID_SIZE}
    final=read_json(SIDE/'manifest.json')
    if final:
        if final['status'] != 'complete' or final['identity'] != identity:
            raise RuntimeError('Existing geometry sidecar identity differs; retain it')
        verify_sidecar();print('geometry sidecar already verified complete',flush=True);return
    # A partial writer is never automatically adopted or overwritten.
    if (SIDE/'geometry_points.npy').exists():
        raise RuntimeError('Partial geometry cache exists; inspect retained evidence before recovery')
    ids,points,normals=library();lookup={g:i for i,g in enumerate(ids)}
    np.save(SIDE/'geometry_points.npy',points,allow_pickle=False)
    np.save(SIDE/'geometry_normals.npy',normals,allow_pickle=False)
    write_json(SIDE/'geometry_library.json',{'ids':ids,'actual_surface':True,'source_binding':identity['sources']})
    load_official_erwin()
    from balltree import build_balltree_with_rotations
    from scipy.spatial import cKDTree
    split_counts={};started=time.perf_counter()
    for split,folder in SPLITS.items():
        dest=SIDE/split;dest.mkdir()
        offsets=np.load(folder/'object_offsets.npy',allow_pickle=False)
        feats=np.load(folder/'features.npy',mmap_mode='r',allow_pickle=False)
        nn=int(offsets[-1]);np.save(dest/'object_offsets.npy',offsets,allow_pickle=False)
        gi=np.lib.format.open_memmap(dest/'geometry_index.npy',mode='w+',dtype=np.int32,shape=(nn,))
        rot=np.lib.format.open_memmap(dest/'rotations.npy',mode='w+',dtype=np.float32,shape=(nn,3,3))
        if (folder/'geometry_index.npy').exists():
            libmeta=read_json(folder.parent/'geometry_library.json')
            oldids=libmeta['ids'] if isinstance(libmeta,dict) else [old_id(x) for x in libmeta]
            oldgi=np.load(folder/'geometry_index.npy',allow_pickle=False)
            gi[:]=np.asarray([lookup[oldids[int(j)]] for j in oldgi],np.int32)
            rot[:]=np.load(folder/'rotations.npy',allow_pickle=False)
        else:
            raw=ROOT/'datasets/canonical_response_v3/data'/split
            manifest=read_json(raw/'manifest.json')
            records=sorted(manifest['samples'],key=lambda x:x['index'])
            if len(records)!=len(offsets)-1:
                raise RuntimeError('Source geometry scene ordering/count mismatch')
            for i,rec in enumerate(records):
                lo,hi=map(int,offsets[i:i+2])
                if len(rec['objects']) != hi-lo:raise RuntimeError('Object count mismatch')
                gi[lo:hi]=[lookup[old_id(obj)] for obj in rec['objects']]
                rot[lo:hi]=[obj['rotation'] for obj in rec['objects']]
        if not np.allclose(np.matmul(rot,rot.transpose(0,2,1)),np.eye(3),atol=2e-5) or not np.allclose(np.linalg.det(rot),1,atol=2e-5):
            raise RuntimeError('Not proper stored body rotations')
        knn=np.lib.format.open_memmap(dest/'query_knn.npy',mode='w+',dtype=np.uint16,shape=(nn*128,K_NEIGHBORS))
        # Official balanced trees pad each scene to a power of two. Three
        # objects have 6144 real points but 8192 tree slots; mask retains all.
        padded=np.asarray([1 << (int(n)*POINTS-1).bit_length() for n in np.diff(offsets)],np.int64)
        tree_offsets=np.r_[0,np.cumsum(padded)].astype(np.int64)
        np.save(dest/'tree_offsets.npy',tree_offsets,allow_pickle=False)
        nt=int(tree_offsets[-1])
        # First pass chooses shared real surface representatives. Variable
        # counts require independent surface and padded-tree CSR offsets.
        chosen=[];counts=[]
        context=np.lib.format.open_memmap(dest/'query_context.npy',mode='w+',dtype=np.float32,shape=(nn*128,12))
        for i in range(len(offsets)-1):
            lo,hi=map(int,offsets[i:i+2]);ql,qh=lo*128,hi*128
            ctx=np.array(feats[ql:qh,128:],copy=True);context[ql:qh]=ctx
            _,pos,_=placed(points,normals,np.asarray(gi[lo:hi]),np.asarray(rot[lo:hi]),ctx)
            select,_=surface_sample(pos);chosen.append(select);counts.append(len(select))
        surface_offsets=np.r_[0,np.cumsum(counts)].astype(np.int64)
        padded=np.asarray([1 << (int(n)-1).bit_length() for n in counts],np.int64)
        tree_offsets=np.r_[0,np.cumsum(padded)].astype(np.int64);nt=int(tree_offsets[-1])
        np.save(dest/'surface_offsets.npy',surface_offsets,allow_pickle=False)
        np.save(dest/'tree_offsets.npy',tree_offsets,allow_pickle=False)
        np.save(dest/'surface_indices.npy',np.concatenate(chosen),allow_pickle=False)
        idx=np.lib.format.open_memmap(dest/'tree_idx.npy',mode='w+',dtype=np.uint16,shape=(nt,))
        mask=np.lib.format.open_memmap(dest/'tree_mask.npy',mode='w+',dtype=np.bool_,shape=(nt,))
        rr=[np.lib.format.open_memmap(dest/f'tree_rot{j}.npy',mode='w+',dtype=np.uint16,shape=(nt//(2**j),)) for j in range(4)]
        for i in range(len(offsets)-1):
            lo,hi=map(int,offsets[i:i+2]);qlo,qhi=lo*128,hi*128;plo,phi=lo*POINTS,hi*POINTS
            ctx=np.asarray(context[qlo:qhi])
            _,pos,qp=placed(points,normals,np.asarray(gi[lo:hi]),np.asarray(rot[lo:hi]),ctx)
            selection,grid=surface_sample(pos);pos=pos[selection.astype(np.int64)]
            if not np.array_equal(selection,chosen[i]) or grid.max()>=65536:
                raise RuntimeError(f'Surface voxel collision/depth overflow: {split}/{i}; no point discarded')
            knn[qlo:qhi]=cKDTree(pos).query(qp,k=K_NEIGHBORS,workers=1)[1].astype(np.uint16)
            tpos=torch.from_numpy(pos)
            tid,tm,tr=build_balltree_with_rotations(tpos,torch.zeros(len(pos),dtype=torch.long),
                ERWIN_CONFIG['strides'],ERWIN_CONFIG['ball_sizes'],ERWIN_CONFIG['rotate'])
            tl,th=map(int,tree_offsets[i:i+2])
            if len(tid)!=th-tl or not torch.equal(torch.sort(tid[tm]).values,torch.arange(len(pos))):
                raise RuntimeError('Official tree changed point preservation/padding contract')
            idx[tl:th]=tid.numpy().astype(np.uint16);mask[tl:th]=tm.numpy()
            for j,v in enumerate(tr):
                if len(v)!=(th-tl)//(2**j):raise RuntimeError('Tree rotation stage length mismatch')
                rr[j][tl//(2**j):th//(2**j)]=v.numpy().astype(np.uint16)
            if (i+1)%100==0 or i==len(offsets)-2:
                print(json.dumps({'event':'surface_cache','split':split,'done':i+1,'total':len(offsets)-1,'elapsed':time.perf_counter()-started}),flush=True)
                write_json(SIDE/'progress.json',{'split':split,'done':i+1,'total':len(offsets)-1,'updated_utc':time.time()})
        for arr in [gi,rot,knn,idx,mask,context,*rr]:arr.flush()
        split_counts[split]={'scenes':len(offsets)-1,'objects':nn,'raw_surface_nodes':nn*POINTS,
                            'sampled_surface_nodes':int(surface_offsets[-1]),
                            'scene_surface_min':int(min(counts)),'scene_surface_max':int(max(counts))}
    files={str(p.relative_to(SIDE)).replace('\\','/'):sha(p) for p in SIDE.rglob('*') if p.is_file() and p.name not in ('manifest.json','progress.json')}
    write_json(SIDE/'manifest.json',{'status':'complete','identity':identity,'splits':split_counts,
        'files_sha256':files,'scientific_release':False,'elapsed_seconds':time.perf_counter()-started})
    print(json.dumps({'event':'surface_cache_complete','splits':split_counts}),flush=True)


def verify_sidecar():
    m=read_json(SIDE/'manifest.json')
    if not m or m['status']!='complete' or m['identity']['sources']!=source_binding() or m['identity']['preparer']!=sha(Path(__file__)):
        raise RuntimeError('Missing/stale source-bound surface sidecar')
    for rel,expected in m['files_sha256'].items():
        if sha(SIDE/rel)!=expected:raise RuntimeError(f'Surface checksum mismatch: {rel}')
    return m


class SurfaceScenes(Dataset):
    def __init__(self,folder,method):
        self.folder=Path(folder);self.method=method
        split=next((s for s,p in SPLITS.items() if p.resolve()==self.folder.resolve()),None)
        if split is None:raise RuntimeError('Unexpected physical source')
        self.side=SIDE/split
        self.arrays={name:np.load(self.folder/(name+'.npy'),mmap_mode='r') for name in FIELDS}
        self.offsets=np.load(self.side/'object_offsets.npy')
        self.tree_offsets=np.load(self.side/'tree_offsets.npy')
        self.surface_offsets=np.load(self.side/'surface_offsets.npy')
        self.geo={name:np.load(self.side/(name+'.npy'),mmap_mode='r') for name in ('geometry_index','rotations','query_knn','query_context','surface_indices','tree_idx','tree_mask',*(f'tree_rot{j}' for j in range(4)))}
        self.points=np.load(SIDE/'geometry_points.npy');self.normals=np.load(SIDE/'geometry_normals.npy')

    def __len__(self):return len(self.offsets)-1

    def __getitem__(self,index):
        lo,hi=map(int,self.offsets[index:index+2]);ql,qh=lo*128,hi*128;pl,ph=lo*POINTS,hi*POINTS
        ctx=np.array(self.geo['query_context'][ql:qh],copy=True)
        feat,pos,qp=placed(self.points,self.normals,np.asarray(self.geo['geometry_index'][lo:hi]),np.asarray(self.geo['rotations'][lo:hi]),ctx)
        sl,sh=map(int,self.surface_offsets[index:index+2])
        selected=np.asarray(self.geo['surface_indices'][sl:sh],dtype=np.int64)
        grid=np.floor((pos-pos.min(0))/GRID_SIZE).astype(np.int32)[selected]
        feat,pos=feat[selected],pos[selected]
        vals=[torch.from_numpy(np.array(self.arrays[name][ql:qh],copy=True)) for name in FIELDS[1:]]
        tree=None
        if self.method=='erwin':
            tl,th=map(int,self.tree_offsets[index:index+2])
            tree=(torch.from_numpy(np.array(self.geo['tree_idx'][tl:th],dtype=np.int64)),
                  torch.from_numpy(np.array(self.geo['tree_mask'][tl:th],copy=True)),
                  [torch.from_numpy(np.array(self.geo[f'tree_rot{j}'][tl//(2**j):th//(2**j)],dtype=np.int64)) for j in range(4)])
        return dict(node_features=torch.from_numpy(feat),positions=torch.from_numpy(pos),
                    grid=torch.from_numpy(grid),
                    query_features=torch.from_numpy(ctx),query_positions=torch.from_numpy(qp),
                    neighbors=torch.from_numpy(np.array(self.geo['query_knn'][ql:qh],dtype=np.int64)),
                    values=vals,tree=tree,index=index)


@dataclass
class SurfaceBatch:
    node_features:torch.Tensor
    positions:torch.Tensor
    grid:torch.Tensor
    query_features:torch.Tensor
    query_positions:torch.Tensor
    neighbors:torch.Tensor
    labels:torch.Tensor
    incident:torch.Tensor
    q64_gt:torch.Tensor
    z_real:torch.Tensor
    z_imag:torch.Tensor
    batch_index:torch.Tensor
    sample_node_counts:torch.Tensor
    sample_index:torch.Tensor
    surface_batch_index:torch.Tensor
    tree:tuple|None

    def to(self,device):
        out={}
        for k,v in self.__dict__.items():
            if k=='sample_node_counts':out[k]=v
            elif k=='tree' and v is not None:out[k]=(v[0].to(device),v[1].to(device),[r.to(device) for r in v[2]])
            else:out[k]=v.to(device) if torch.is_tensor(v) else v
        return SurfaceBatch(**out)


def collate(items):
    counts=torch.tensor([len(x['query_positions']) for x in items],dtype=torch.long)
    surface_counts=torch.tensor([len(x['positions']) for x in items],dtype=torch.long)
    values=[torch.cat([x['values'][j] for x in items]) for j in range(5)]
    neighbors=[];off=0;tree=None;idxs=[];masks=[];rots=[[] for _ in range(4)];roff=[0]*4
    for x in items:
        neighbors.append(x['neighbors']+off)
        if x['tree'] is not None:
            idx,mask,rs=x['tree'];idxs.append(idx+off);masks.append(mask)
            for j,r in enumerate(rs):rots[j].append(r+roff[j]);roff[j]+=len(r)
        off+=len(x['positions'])
    if idxs:tree=torch.cat(idxs),torch.cat(masks),[torch.cat(r) for r in rots]
    return SurfaceBatch(*(torch.cat([x[k] for x in items]) for k in ('node_features','positions','grid','query_features','query_positions')),
        torch.cat(neighbors),*values,torch.repeat_interleave(torch.arange(len(items)),counts),counts,
        torch.tensor([x['index'] for x in items]),torch.repeat_interleave(torch.arange(len(items)),surface_counts),tree)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--verify',action='store_true');args=p.parse_args()
    torch.set_num_threads(4)
    if args.verify:verify_sidecar()
    else:prepare()
