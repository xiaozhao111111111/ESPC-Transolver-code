"""Local/global point-cloud encoding on 128 canonical nodes per object."""
from __future__ import annotations
from dataclasses import dataclass, fields, replace
from pathlib import Path
from shared.point_reference_runtime import ROOT, np, torch
from shared.paths import GEOMETRY as SIDE, SPLITS
from shared.learning import GeoStrongAdapted
import shared.point_surface_data as surface
from Erwin.S_Z.model import PointReference
def placed_geometry(points, normals, gids, rotations, context):
    u, _, vt = np.linalg.svd(np.asarray(rotations, np.float64))
    rotations = (u @ vt).astype(np.float32)
    points = np.einsum('opi,oji->opj', points[gids], rotations)
    normals = np.einsum('opi,oji->opj', normals[gids], rotations)
    queries = np.asarray(context, np.float32).reshape(-1, 128, 12)[:, :, 2:5]*np.float32(1.5)
    return points, normals, queries

@dataclass
class GeoBatch:
    node_features: torch.Tensor
    positions: torch.Tensor
    labels: torch.Tensor
    incident: torch.Tensor
    batch_index: torch.Tensor
    sample_node_counts: torch.Tensor
    sample_index: torch.Tensor
    tree: tuple
    geo_local: torch.Tensor
    geo_global: torch.Tensor

    def to(self, device):
        values = {}
        for f in fields(self):
            v = getattr(self, f.name)
            if f.name == 'sample_node_counts':
                values[f.name] = v
            elif f.name == 'tree':
                values[f.name] = (v[0].to(device), v[1].to(device), [r.to(device) for r in v[2]])
            else:
                values[f.name] = v.to(device)
        return GeoBatch(**values)

class Scenes(torch.utils.data.Dataset):
    def __init__(self, folder, method='erwin'):
        if method != 'erwin':
            raise ValueError('This experiment changes Erwin only')
        self.folder = Path(folder)
        self.split = next((s for s, p in SPLITS.items() if p.resolve() == self.folder.resolve()), None)
        if self.split is None:
            raise ValueError('Unknown source split')
        source, side = surface.SIDE/self.split, SIDE/self.split
        self.arrays = {n: np.load(self.folder/(n+'.npy'), mmap_mode='r') for n in ('labels', 'incident')}
        self.geo = {n: np.load(source/(n+'.npy'), mmap_mode='r') for n in
                    ('object_offsets', 'geometry_index', 'rotations', 'query_context')}
        self.cached = {n: np.load(side/(n+'.npy'), mmap_mode='r') for n in
                       ('tree_offsets', 'nearest16', 'tree_idx', 'tree_mask', *(f'tree_rot{j}' for j in range(4)))}
        self.points = np.load(surface.SIDE/'geometry_points.npy')
        self.normals = np.load(surface.SIDE/'geometry_normals.npy')

    def __len__(self):
        return len(self.geo['object_offsets'])-1

    def __getitem__(self, scene):
        a, b = map(int, self.geo['object_offsets'][scene:scene+2])
        ctx = np.array(self.geo['query_context'][a*128:b*128], copy=True)
        points, normals, queries = placed_geometry(self.points, self.normals,
            np.asarray(self.geo['geometry_index'][a:b]), self.geo['rotations'][a:b], ctx)
        nearest = np.asarray(self.cached['nearest16'][a:b], dtype=np.int64)
        objects = np.arange(b-a)[:, None, None]
        local = np.concatenate(((points[objects, nearest]-queries[:, :, None])/np.float32(1.5),
                                normals[objects, nearest]), axis=-1).reshape(-1, 16, 6)
        glob = np.concatenate((points/np.float32(1.5), normals), axis=-1)
        features = np.concatenate((np.zeros((len(ctx), 128), np.float32), ctx), axis=1)
        lo, hi = map(int, self.cached['tree_offsets'][scene:scene+2])
        tree = (torch.from_numpy(np.array(self.cached['tree_idx'][lo:hi], dtype=np.int64)),
                torch.from_numpy(np.array(self.cached['tree_mask'][lo:hi], copy=True)),
                [torch.from_numpy(np.array(self.cached[f'tree_rot{j}'][lo//2**j:hi//2**j],
                                           dtype=np.int64)) for j in range(4)])
        return {'node_features': torch.from_numpy(features),
                'positions': torch.from_numpy(ctx[:, 2:5]+ctx[:, 5:8]),
                'labels': torch.from_numpy(np.array(self.arrays['labels'][a*128:b*128], copy=True)),
                'incident': torch.from_numpy(np.array(self.arrays['incident'][a*128:b*128], copy=True)),
                'geo_local': torch.from_numpy(np.ascontiguousarray(local)),
                'geo_global': torch.from_numpy(np.ascontiguousarray(glob)), 'tree': tree, 'index': scene}

def collate(items):
    counts = torch.tensor([len(i['positions']) for i in items], dtype=torch.long)
    offset, rotations_offsets = 0, [0]*4
    indexes, masks, rotations = [], [], [[] for _ in range(4)]
    for item in items:
        idx, mask, rot = item['tree']
        indexes.append(idx+offset); masks.append(mask)
        for j, r in enumerate(rot):
            rotations[j].append(r+rotations_offsets[j]); rotations_offsets[j] += len(r)
        offset += len(item['positions'])
    tree = (torch.cat(indexes), torch.cat(masks), [torch.cat(r) for r in rotations])
    return GeoBatch(*(torch.cat([x[name] for x in items]) for name in
        ('node_features', 'positions', 'labels', 'incident')),
        torch.repeat_interleave(torch.arange(len(items)), counts), counts,
        torch.tensor([i['index'] for i in items]), tree,
        torch.cat([i['geo_local'] for i in items]), torch.cat([i['geo_global'] for i in items]))
def model(method='erwin'):
    return GeoStrongAdapted(PointReference(method))
