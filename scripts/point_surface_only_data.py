"""Source-derived real surface interface with only incident/field labels."""
from __future__ import annotations
import ast
import copy
from dataclasses import dataclass
from pathlib import Path
import point_surface_data as base
from point_reference_runtime import np, torch, ROOT
from torch.utils.data import Dataset

FIELDS = ('labels', 'incident')


class SurfaceScenes(Dataset):
    def __init__(self, folder, method):
        self.folder = Path(folder); self.method = method
        split = next((s for s, p in base.SPLITS.items()
                      if p.resolve() == self.folder.resolve()), None)
        if split is None: raise RuntimeError('Unexpected source for field-only loader')
        self.side = base.SIDE / split
        self.arrays = {n: np.load(self.folder/(n+'.npy'), mmap_mode='r') for n in FIELDS}
        self.offsets = np.load(self.side/'object_offsets.npy')
        self.tree_offsets = np.load(self.side/'tree_offsets.npy')
        self.surface_offsets = np.load(self.side/'surface_offsets.npy')
        self.geo = {n: np.load(self.side/(n+'.npy'), mmap_mode='r') for n in
                    ('geometry_index', 'rotations', 'query_knn', 'query_context',
                     'surface_indices', 'tree_idx', 'tree_mask',
                     *(f'tree_rot{j}' for j in range(4)))}
        self.points = np.load(base.SIDE/'geometry_points.npy')
        self.normals = np.load(base.SIDE/'geometry_normals.npy')

    def __len__(self): return len(self.offsets)-1


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


def source_definitions():
    tree = ast.parse((ROOT/'scripts/point_surface_data.py').read_text())
    classes = {n.name: n for n in tree.body if isinstance(n, ast.ClassDef)}
    getter = copy.deepcopy(next(n for n in classes['SurfaceScenes'].body
                                if isinstance(n, ast.FunctionDef) and n.name == '__getitem__'))
    # Exactly one change: the values list reads the two declared fields.
    hits = [n for n in ast.walk(getter) if isinstance(n, ast.Subscript)
            and isinstance(n.value, ast.Name) and n.value.id == 'FIELDS']
    if len(hits) != 1 or ast.unparse(hits[0]) != 'FIELDS[1:]':
        raise RuntimeError('Frozen getter contract changed')
    getter = _ReplaceFieldSlice().visit(getter)
    mover = copy.deepcopy(next(n for n in classes['SurfaceBatch'].body
                               if isinstance(n, ast.FunctionDef) and n.name == 'to'))
    collator = copy.deepcopy(next(n for n in tree.body
                                  if isinstance(n, ast.FunctionDef) and n.name == 'collate'))
    hits = [n for n in ast.walk(collator) if isinstance(n, ast.Call)
            and isinstance(n.func, ast.Name) and n.func.id == 'range'
            and len(n.args) == 1 and isinstance(n.args[0], ast.Constant) and n.args[0].value == 5]
    if len(hits) != 1: raise RuntimeError('Frozen collation contract changed')
    hits[0].args[0] = ast.Constant(2)
    ns = dict(np=np, torch=torch, FIELDS=FIELDS, POINTS=base.POINTS,
              GRID_SIZE=base.GRID_SIZE, placed=base.placed, SurfaceBatch=SurfaceBatch)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[getter, mover, collator], type_ignores=[])),
                 str(ROOT/'scripts/point_surface_data.py'), 'exec'), ns)
    SurfaceScenes.__getitem__ = ns['__getitem__']
    SurfaceBatch.to = ns['to']
    return ns['collate']


class _ReplaceFieldSlice(ast.NodeTransformer):
    def visit_Subscript(self, node):
        if isinstance(node.value, ast.Name) and node.value.id == 'FIELDS':
            return ast.copy_location(ast.Name(id='FIELDS', ctx=ast.Load()), node)
        return self.generic_visit(node)


collate = source_definitions()


def field_loss_definition():
    tree = ast.parse((ROOT/'src/cspgnet/v3_learning.py').read_text())
    defs = {n.name: copy.deepcopy(n) for n in tree.body if isinstance(n, ast.FunctionDef)}
    loss = defs['loss_and_scene_metrics']
    removed = []
    body = []
    for statement in loss.body:
        if isinstance(statement, ast.Assign) and len(statement.targets) == 1:
            target = statement.targets[0]
            name = target.id if isinstance(target, ast.Name) else ''
            if name.startswith('q_'):
                removed.append(name); continue
            if name == 'loss':
                statement.value = ast.Call(func=ast.Attribute(value=ast.Name(id='scatter_loss', ctx=ast.Load()),
                    attr='mean', ctx=ast.Load()), args=[], keywords=[])
            if name == 'metrics':
                pairs = [(k, v) for k, v in zip(statement.value.keys, statement.value.values)
                         if not (isinstance(k, ast.Constant) and k.value == 'q64_derived_rel_l2')]
                if len(pairs) != 5: raise RuntimeError('Frozen field metric contract changed')
                statement.value = ast.Dict(keys=[k for k, _ in pairs], values=[v for _, v in pairs])
        body.append(statement)
    if removed != ['q_difference', 'q_numerator', 'q_target', 'q_energy', 'q_loss']:
        raise RuntimeError('Frozen loss contract changed')
    loss.body = body
    ns = {'torch': torch}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[defs['_complex'], defs['_reduce'], loss],
                 type_ignores=[])), str(ROOT/'src/cspgnet/v3_learning.py'), 'exec'), ns)
    return ns['loss_and_scene_metrics']


_FIELD_LOSS = field_loss_definition()


def loss_and_metrics(prediction, batch, phi, statistics, lambda_q, output_head='scattered'):
    if lambda_q != 0: raise RuntimeError('Field-only batch has no flux loss; require lambda_q=0')
    return _FIELD_LOSS(prediction, batch, None, statistics, 0, output_head)
