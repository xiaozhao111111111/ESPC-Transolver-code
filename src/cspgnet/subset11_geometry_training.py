"""Geometry sidecar loaders; keeps the previously validated Raw-Z/S(Z) path intact."""
from __future__ import annotations
from dataclasses import dataclass,fields,replace
from pathlib import Path
import numpy as np
import torch
from torch import nn
from .subset11_training import SceneBatch,Subset11Dataset,Subset11Transolver,read,sha,verify_pack


def verify_geometry(base,geometry):
    verify_pack(base)
    geometry=Path(geometry); manifest=read(geometry/'geometry_manifest.json')
    if (manifest.get('schema')!='subset11-geometry-sidecar-v1' or manifest.get('status')!='complete'
        or manifest.get('scientific_release') is not False
        or manifest['base_manifest_sha256']!=sha(Path(base)/'trainable_manifest.json')):
        raise RuntimeError('Incomplete, unsupported or mismatched geometry sidecar')
    for name,item in manifest['files'].items():
        if sha(geometry/name)!=item['sha256']: raise RuntimeError('Geometry SHA differs: '+name)
    return manifest


@dataclass
class GeometryBatch(SceneBatch):
    geo_local: torch.Tensor | None=None
    geo_global: torch.Tensor | None=None

    def to(self,device):
        return GeometryBatch(**{f.name:getattr(self,f.name) if f.name=='sample_node_counts' or getattr(self,f.name) is None
            else getattr(self,f.name).to(device,non_blocking=True) for f in fields(self)})

    def pin_memory(self):
        return GeometryBatch(**{f.name:getattr(self,f.name) if getattr(self,f.name) is None
            else getattr(self,f.name).pin_memory() for f in fields(self)})


class FourInputDataset(Subset11Dataset):
    def __init__(self,base,split,representation,geometry):
        if representation not in ('raw-z','s-response','geo-matched','geo-strong'): raise ValueError(representation)
        super().__init__(base,split,representation if representation in ('raw-z','s-response') else 'raw-z')
        self.arm=representation; self.geometry=Path(geometry)
        manifest=read(self.geometry/'geometry_manifest.json')
        if manifest.get('status')!='complete' or manifest['base_manifest_sha256']!=sha(self.root/'trainable_manifest.json'):
            raise RuntimeError('Mismatched geometry sidecar')
        geom_stats=read(self.geometry/'statistics.json')
        if geom_stats['fit_split']!='train' or geom_stats['base_train_manifest_sha256']!=self.statistics['source_train_manifest_sha256']:
            raise RuntimeError('Geometry standardization must fit train only')
        self.statistics={**self.statistics,**geom_stats}
        self.mean=np.asarray(geom_stats['occupancy_channel_mean'],np.float32)
        self.std=np.asarray(geom_stats['occupancy_channel_std'],np.float32)
        with np.load(self.geometry/'pointclouds.npz',allow_pickle=False) as a:
            self.clouds=a['points']; self.normals=a['normals']; self.ids=a['geometry_ids']
        self._geometry_arrays=None

    def __getstate__(self):
        state=super().__getstate__(); state['_geometry_arrays']=None; return state

    def __getitem__(self,index):
        base=super().__getitem__(index)
        result=GeometryBatch(**{f.name:getattr(base,f.name) for f in fields(SceneBatch)})
        if self.arm in ('raw-z','s-response'): return result
        if self._geometry_arrays is None:
            keys=('occupancy_bits','nearest_indices','geometry_index')
            self._geometry_arrays={k:np.load(self.geometry/self.split/(k+'.npy'),mmap_mode='r') for k in keys}
            self._geometry_arrays['rotations']=np.load(self.root/self.split/'rotations.npy',mmap_mode='r')
        lo,hi=int(self.offsets[index]),int(self.offsets[index+1])
        if self.arm=='geo-matched':
            mask=np.unpackbits(self._geometry_arrays['occupancy_bits'][lo:hi],axis=-1,bitorder='little').astype(np.float32)
            result.node_features[:,:128]=torch.from_numpy(((mask-self.mean)/self.std).reshape(-1,128))
        else:
            # No Z in forward inputs; the separate physical Z is loss-only.
            result.node_features[:,:128]=0
            gid=np.asarray(self._geometry_arrays['geometry_index'][lo:hi],np.int64)
            raw=np.asarray(self._geometry_arrays['rotations'][lo:hi],np.float64)
            u,_,v=np.linalg.svd(raw); R=(u@v).astype(np.float32)
            points=np.einsum('opi,oji->opj',self.clouds[gid],R)
            normals=np.einsum('opi,oji->opj',self.normals[gid],R)
            indices=np.asarray(self._geometry_arrays['nearest_indices'][lo:hi],np.int64)
            objects=np.arange(hi-lo)[:,None,None]
            local=np.concatenate(((points[objects,indices]-self.points[None,:,None])/1.5,
                                  normals[objects,indices]),axis=-1)
            glob=np.concatenate((points/1.5,normals),axis=-1)
            result.geo_local=torch.from_numpy(local.reshape(-1,16,6))
            result.geo_global=torch.from_numpy(glob)
        return result


def collate_geometry(scenes):
    result={}
    for f in fields(GeometryBatch):
        values=[getattr(s,f.name) for s in scenes]
        if all(v is None for v in values): result[f.name]=None
        elif any(v is None for v in values): raise ValueError('Mixed geometry arms in batch')
        else: result[f.name]=torch.cat(values)
    result['batch_index']=torch.repeat_interleave(torch.arange(len(scenes)),result['sample_node_counts'])
    return GeometryBatch(**result)


class Subset11GeoStrong(nn.Module):
    """Exactly the existing local/global point encoder plus the same Transolver."""
    def __init__(self):
        super().__init__()
        self.backbone=Subset11Transolver()
        self.local_mlp=nn.Sequential(nn.Linear(6,64),nn.SiLU(),nn.Linear(64,128))
        self.global_mlp=nn.Sequential(nn.Linear(6,64),nn.SiLU(),nn.Linear(64,128))
        self.combiner=nn.Linear(384,128)

    def forward(self,batch):
        if batch.geo_local is None or batch.geo_global is None: raise ValueError('Missing CAD pointcloud')
        local=self.local_mlp(batch.geo_local).amax(dim=1)
        glob=self.global_mlp(batch.geo_global)
        glob=torch.cat((glob.amax(dim=1),glob.mean(dim=1)),dim=-1)
        if int(batch.sample_node_counts.sum())!=len(glob)*128: raise ValueError('Object alignment')
        encoded=self.combiner(torch.cat((local,torch.repeat_interleave(glob,128,dim=0)),dim=-1))
        features=batch.node_features.clone(); features[:,:128]=encoded
        return self.backbone(replace(batch,node_features=features))


def make_model(arm):
    return Subset11GeoStrong() if arm=='geo-strong' else Subset11Transolver()
