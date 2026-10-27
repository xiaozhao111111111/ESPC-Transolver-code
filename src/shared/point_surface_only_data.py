"""Real-surface inputs and field labels; no S(Z), Z or flux arrays are loaded."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import shared.point_surface_data as base
from shared.point_surface_data import POINTS, GRID_SIZE, placed
from shared.point_reference_runtime import np, torch
from shared.learning import field_loss_and_scene_metrics as _FIELD_LOSS
from torch.utils.data import Dataset
FIELDS = ('labels', 'incident')
class SurfaceScenes(Dataset):

    def __init__(self, folder, method):
        self.folder = Path(folder)
        self.method = method
        split = next((s for s, p in base.SPLITS.items() if p.resolve() == self.folder.resolve()), None)
        if split is None:
            raise RuntimeError('Unexpected source for field-only loader')
        self.side = base.SIDE / split
        self.arrays = {n: np.load(self.folder / (n + '.npy'), mmap_mode='r') for n in FIELDS}
        self.offsets = np.load(self.side / 'object_offsets.npy')
        self.tree_offsets = np.load(self.side / 'tree_offsets.npy')
        self.surface_offsets = np.load(self.side / 'surface_offsets.npy')
        self.geo = {n: np.load(self.side / (n + '.npy'), mmap_mode='r') for n in ('geometry_index', 'rotations', 'query_knn', 'query_context', 'surface_indices', 'tree_idx', 'tree_mask', *(f'tree_rot{j}' for j in range(4)))}
        self.points = np.load(base.SIDE / 'geometry_points.npy')
        self.normals = np.load(base.SIDE / 'geometry_normals.npy')

    def __len__(self):
        return len(self.offsets) - 1

    def __getitem__(self, index):
        lo, hi = map(int, self.offsets[index:index + 2])
        ql, qh = (lo * 128, hi * 128)
        pl, ph = (lo * POINTS, hi * POINTS)
        ctx = np.array(self.geo['query_context'][ql:qh], copy=True)
        feat, pos, qp = placed(self.points, self.normals, np.asarray(self.geo['geometry_index'][lo:hi]), np.asarray(self.geo['rotations'][lo:hi]), ctx)
        sl, sh = map(int, self.surface_offsets[index:index + 2])
        selected = np.asarray(self.geo['surface_indices'][sl:sh], dtype=np.int64)
        grid = np.floor((pos - pos.min(0)) / GRID_SIZE).astype(np.int32)[selected]
        feat, pos = (feat[selected], pos[selected])
        vals = [torch.from_numpy(np.array(self.arrays[name][ql:qh], copy=True)) for name in FIELDS]
        tree = None
        if self.method == 'erwin':
            tl, th = map(int, self.tree_offsets[index:index + 2])
            tree = (torch.from_numpy(np.array(self.geo['tree_idx'][tl:th], dtype=np.int64)), torch.from_numpy(np.array(self.geo['tree_mask'][tl:th], copy=True)), [torch.from_numpy(np.array(self.geo[f'tree_rot{j}'][tl // 2 ** j:th // 2 ** j], dtype=np.int64)) for j in range(4)])
        return dict(node_features=torch.from_numpy(feat), positions=torch.from_numpy(pos), grid=torch.from_numpy(grid), query_features=torch.from_numpy(ctx), query_positions=torch.from_numpy(qp), neighbors=torch.from_numpy(np.array(self.geo['query_knn'][ql:qh], dtype=np.int64)), values=vals, tree=tree, index=index)

@dataclass
class SurfaceBatch:
    node_features: torch.Tensor
    positions: torch.Tensor
    grid: torch.Tensor
    query_features: torch.Tensor
    query_positions: torch.Tensor
    neighbors: torch.Tensor
    labels: torch.Tensor
    incident: torch.Tensor
    batch_index: torch.Tensor
    sample_node_counts: torch.Tensor
    sample_index: torch.Tensor
    surface_batch_index: torch.Tensor
    tree: tuple | None

    def to(self, device):
        out = {}
        for k, v in self.__dict__.items():
            if k == 'sample_node_counts':
                out[k] = v
            elif k == 'tree' and v is not None:
                out[k] = (v[0].to(device), v[1].to(device), [r.to(device) for r in v[2]])
            else:
                out[k] = v.to(device) if torch.is_tensor(v) else v
        return SurfaceBatch(**out)

def collate(items):
    counts=torch.tensor([len(x['query_positions']) for x in items],dtype=torch.long)
    surface_counts=torch.tensor([len(x['positions']) for x in items],dtype=torch.long)
    values=[torch.cat([x['values'][j] for x in items]) for j in range(2)]
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

loss_and_metrics = _FIELD_LOSS
