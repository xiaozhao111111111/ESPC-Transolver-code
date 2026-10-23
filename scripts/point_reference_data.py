"""Read-only prepared S(Z) datasets with scene-isolated geometry indexes."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from point_reference_runtime import np, torch
from point_reference_models import GRID_SIZE, ERWIN_CONFIG
from torch.utils.data import Dataset

FIELDS = ('features', 'labels', 'incident', 'q64', 'z_real', 'z_imag')


class Scenes(Dataset):
    def __init__(self, folder, method):
        self.folder = Path(folder)
        self.method = method
        self.arrays = {name: np.load(self.folder / (name + '.npy'), mmap_mode='r') for name in FIELDS}
        self.offsets = np.load(self.folder / 'object_offsets.npy', mmap_mode='r') * 128
        self.trees = {}

    def __len__(self):
        return len(self.offsets) - 1

    def __getitem__(self, index):
        lo, hi = map(int, self.offsets[index:index + 2])
        values = [torch.from_numpy(np.array(self.arrays[name][lo:hi], copy=True)) for name in FIELDS]
        f = values[0]
        pos = (f[:, 130:133] + f[:, 133:136]).contiguous()
        grid = torch.floor((pos - pos.min(dim=0).values) / GRID_SIZE).int()
        if self.method == 'ptv3' and len(torch.unique(grid, dim=0)) != len(grid):
            raise RuntimeError(f'Voxel collision: {self.folder.name}/{index}; never drop canonical output nodes')
        tree = None
        if self.method == 'erwin':
            if index not in self.trees:
                from balltree import build_balltree_with_rotations
                cfg = ERWIN_CONFIG
                idx, mask, rotations = build_balltree_with_rotations(
                    pos, torch.zeros(len(pos), dtype=torch.long), cfg['strides'], cfg['ball_sizes'], cfg['rotate'])
                if not torch.equal(torch.sort(idx[mask]).values, torch.arange(len(pos))):
                    raise RuntimeError('Ball tree did not preserve all canonical nodes exactly once')
                padded_batch = torch.zeros(len(idx), dtype=torch.long)
                # Every pooling and attention partition must stay within a scene.
                for stride, size in zip([1] + cfg['strides'], cfg['ball_sizes']):
                    padded_batch = padded_batch[::stride]
                    if len(padded_batch) % size:
                        raise RuntimeError('Ball tree stage is not divisible by the official ball size')
                self.trees[index] = idx, mask, rotations
            tree = self.trees[index]
        return values, pos, grid, tree, index


@dataclass
class Batch:
    node_features: torch.Tensor
    labels: torch.Tensor
    incident: torch.Tensor
    q64_gt: torch.Tensor
    z_real: torch.Tensor
    z_imag: torch.Tensor
    batch_index: torch.Tensor
    sample_node_counts: torch.Tensor
    sample_index: torch.Tensor
    positions: torch.Tensor
    grid: torch.Tensor
    tree: tuple | None

    def to(self, device):
        values = {}
        for k, value in self.__dict__.items():
            if k == 'sample_node_counts':
                values[k] = value
            elif k == 'tree' and value is not None:
                values[k] = (value[0].to(device), value[1].to(device), [v.to(device) for v in value[2]])
            else:
                values[k] = value.to(device) if torch.is_tensor(value) else value
        return Batch(**values)


def collate(items):
    sizes = torch.tensor([len(item[1]) for item in items], dtype=torch.long)
    values = [torch.cat([item[0][i] for item in items]) for i in range(len(FIELDS))]
    tree = None
    if items[0][3] is not None:
        idxs, masks = [], []
        rots = [[] for _ in ERWIN_CONFIG['ball_sizes']]
        node_offset, stage_offsets = 0, [0] * len(rots)
        for values_i, pos, grid, (idx, mask, rotations), scene in items:
            idxs.append(idx + node_offset)
            masks.append(mask)
            node_offset += len(pos)
            for j, rotation in enumerate(rotations):
                rots[j].append(rotation + stage_offsets[j])
                stage_offsets[j] += len(rotation)
        tree = torch.cat(idxs), torch.cat(masks), [torch.cat(r) for r in rots]
    return Batch(*values, torch.repeat_interleave(torch.arange(len(items)), sizes), sizes,
                 torch.tensor([x[4] for x in items]), torch.cat([x[1] for x in items]),
                 torch.cat([x[2] for x in items]), tree)
