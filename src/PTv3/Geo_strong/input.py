"""PTv3 with the shared Geo-strong encoder and canonical query grid."""
from __future__ import annotations
from dataclasses import dataclass, fields
import Erwin.Geo_strong.input as shared
from shared.point_reference_runtime import np, torch
from shared.learning import GeoStrongAdapted
from PTv3.S_Z.model import PointReference
from PTv3.S_Z.config import GRID_SIZE
SIDE, SPLITS = shared.SIDE, shared.SPLITS
def grid_for(positions):
    # Identical to the existing canonical PTv3 quantization. Retain all nodes.
    grid = torch.floor((positions-positions.min(dim=0).values)/GRID_SIZE).int()
    if len(torch.unique(grid, dim=0)) != len(grid):
        raise RuntimeError('PTv3 canonical grid collision; never remove queries')
    return grid

class Scenes(shared.Scenes):
    def __init__(self, folder, method='ptv3'):
        if method != 'ptv3':
            raise ValueError('This independent experiment trains PTv3 only')
        super().__init__(folder, 'erwin')

    def __getitem__(self, scene):
        item = super().__getitem__(scene)
        item['grid'] = grid_for(item['positions'])
        return item

@dataclass
class GeoBatch(shared.GeoBatch):
    grid: torch.Tensor

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

def collate(items):
    base = shared.collate(items)
    return GeoBatch(**vars(base), grid=torch.cat([i['grid'] for i in items]))
def model(method='ptv3'):
    return GeoStrongAdapted(PointReference(method))
