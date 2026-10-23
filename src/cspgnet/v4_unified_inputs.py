"""Unified v4 training graphs for the unchanged existing model constructors.

Use the CUDA training runtime, not the FEM runtime's incompatible Torch DLLs.
No old model, trainer, schema, or test file is modified by this adapter.
"""
from __future__ import annotations
import hashlib
import json
import multiprocessing
from pathlib import Path
import numpy as np
import torch
from .structural_dataset import UnifiedSceneIndex,radial_occupancy,sha
from .v3_inputs import sample_to_v3_edge_graph
from .v3_pointcloud import PointCloudCache,attach_geostrong_features
from .data_generation.structural_response import scattering_response


class StructuralPointCloudCache(PointCloudCache):
    def get(self,record):
        if record['shape']!='structural_voxel': return super().get(record)
        path=Path(record['pointcloud_file']); token=record['pointcloud_sha256']
        if token in self.memory: return self.memory[token]
        if sha(path)!=token: raise RuntimeError('structural area pointcloud corrupted')
        with np.load(path,allow_pickle=False) as data:
            points,normals=data['points'],data['normals']
            if str(data['structure_sha256'])!=record['structure_sha256']: raise RuntimeError('cloud belongs to different structure')
        if points.shape!=(2048,3) or normals.shape!=points.shape: raise RuntimeError('pointcloud contract failed')
        self.memory[token]=(points,normals); return points,normals


class V4UnifiedCanonicalDataset(torch.utils.data.Dataset):
    def __init__(self,union_manifest,statistics,representation,pointcloud_cache=None):
        if representation not in ('s-response','raw-z','geo-matched','geo-strong'): raise ValueError(representation)
        self.index=UnifiedSceneIndex(union_manifest); self.files=self.index.files; self.manifest=self.index.manifest
        self.statistics=statistics; self.representation=representation
        if statistics['fit_samples']!=20000 or statistics['union_manifest_sha256']!=sha(union_manifest):
            raise RuntimeError('normalization was not fit on this entire training union')
        self._shared_epoch=multiprocessing.Value('i',0)
        self.pointclouds=StructuralPointCloudCache(pointcloud_cache) if representation=='geo-strong' else None
        if representation=='geo-strong' and pointcloud_cache is None: raise ValueError('geometry cache required')
        first,_=self.index.load(0); self.phi_pinv=torch.from_numpy(first['phi_pinv'].astype(np.complex64))
    def __len__(self): return len(self.index)
    def set_epoch(self,epoch):
        with self._shared_epoch.get_lock(): self._shared_epoch.value=int(epoch)
    def __getitem__(self,i):
        data,record=self.index.load(i)
        material=json.dumps(['canonical-v4-union-remote-edges',sha(self.index.path),i,self._shared_epoch.value]).encode()
        seed=int.from_bytes(hashlib.sha256(material).digest()[:4],'little')
        graph=sample_to_v3_edge_graph(data,representation='zero-z',statistics=self.statistics,
                                     output_head='scattered',edge_seed=seed,sample_index=i)
        if self.representation in ('s-response','raw-z'):
            if self.representation=='raw-z': response=data['Z']; scales=self.statistics['harmonic_degree_complex_rms']
            else: response=data['S'] if 'S' in data else scattering_response(data); scales=self.statistics['s_response_degree_complex_rms']
            response=np.asarray(response,np.complex128)/np.repeat(np.asarray(scales),np.arange(1,16,2))[None,None]
            channels=np.concatenate((response.real,response.imag),axis=-1).astype(np.float32)
            graph.node_features[:,:128]=torch.from_numpy(channels.reshape(-1,128))
        elif self.representation=='geo-matched':
            channels=(radial_occupancy(data)-self.statistics['radial_occupancy_mean'])/self.statistics['radial_occupancy_std']
            graph.node_features[:,:128]=torch.from_numpy(channels.astype(np.float32).reshape(-1,128))
        if self.pointclouds is not None: attach_geostrong_features(graph,data,record,self.pointclouds)
        # Preserve the matched Transolver center channels from all old arms.
        centers=torch.from_numpy(np.asarray(data['centers'],np.float32))
        graph.node_features[:,133:136]=torch.repeat_interleave((centers-centers.mean(0,keepdim=True))/1.5,128,dim=0)
        return graph
