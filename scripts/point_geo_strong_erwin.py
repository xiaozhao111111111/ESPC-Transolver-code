"""Independent field-only Geo-strong inputs for the unchanged canonical Erwin."""
from __future__ import annotations

import argparse
from dataclasses import dataclass, fields, replace
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
from point_reference_runtime import ROOT, np, torch, leaf, load_official_erwin
from point_reference_models import PointReference, ERWIN_CONFIG
from point_reference_io_recovery import read_json, write_json
import point_surface_data as surface
from prepare_point_reference_tests import sha, CACHE

OUT = ROOT / 'outputs/erwin_geo_strong_local_20261013'
SIDE = OUT / 'geometry'
INVENTORY = ROOT / 'outputs/point_surface_only_local_20261012/transfer_inventory.json'
SPLITS = surface.SPLITS


def input_binding(verify=False):
    inventory = read_json(INVENTORY)
    if not inventory or inventory['status'] != 'prepared' or inventory['lambda_q'] != 0:
        raise RuntimeError('Missing immutable field-only source inventory')
    required = [CACHE/'statistics.json', surface.SIDE/'geometry_points.npy',
                surface.SIDE/'geometry_normals.npy']
    for split, folder in SPLITS.items():
        required += [folder/(name+'.npy') for name in ('labels', 'incident')]
        required += [surface.SIDE/split/(name+'.npy') for name in
                     ('object_offsets', 'geometry_index', 'rotations', 'query_context')]
    records = {}
    for path in required:
        relative = path.relative_to(ROOT).as_posix()
        record = inventory['files'].get(relative)
        if not record:
            raise RuntimeError('Source missing from qualified inventory: '+relative)
        if verify and (path.stat().st_size != record['size'] or sha(path) != record['sha256']):
            raise RuntimeError('Source checksum mismatch: '+relative)
        records[relative] = record
    return {'inventory_sha256': sha(INVENTORY), 'files': records,
            'preparer_sha256': sha(Path(__file__)), 'raw_points_per_body': 2048,
            'canonical_nodes_per_body': 128, 'local_neighbors': 16,
            'cloud_rotation': 'saved proper rotation, SVD polar correction as existing Geo-strong',
            'field_only': True}


def placed_geometry(points, normals, gids, rotations, context):
    u, _, vt = np.linalg.svd(np.asarray(rotations, np.float64))
    rotations = (u @ vt).astype(np.float32)
    points = np.einsum('opi,oji->opj', points[gids], rotations)
    normals = np.einsum('opi,oji->opj', normals[gids], rotations)
    queries = np.asarray(context, np.float32).reshape(-1, 128, 12)[:, :, 2:5]*np.float32(1.5)
    return points, normals, queries


def prepare():
    binding = input_binding(verify=True)
    previous = read_json(SIDE/'manifest.json')
    if previous:
        if previous['status'] != 'complete' or previous['binding'] != binding:
            raise RuntimeError('Existing geometry cache identity differs; preserve it')
        verify_geometry()
        return
    if SIDE.exists() and any(SIDE.iterdir()):
        raise RuntimeError('Partial geometry cache exists; inspect rather than overwrite')
    SIDE.mkdir(parents=True)
    from scipy.spatial import cKDTree
    load_official_erwin()
    from balltree import build_balltree_with_rotations
    points = np.load(surface.SIDE/'geometry_points.npy')
    normals = np.load(surface.SIDE/'geometry_normals.npy')
    if points.shape[1:] != (2048, 3) or normals.shape != points.shape:
        raise RuntimeError('Raw surface geometry shape changed')
    started = time.perf_counter()
    counts = {}
    for split in SPLITS:
        source = surface.SIDE/split
        offsets, gids, rotations, context = [np.load(source/(name+'.npy'), mmap_mode='r')
            for name in ('object_offsets', 'geometry_index', 'rotations', 'query_context')]
        dest = SIDE/split
        dest.mkdir()
        padded = np.array([1 << (int(n)*128-1).bit_length() for n in np.diff(offsets)], np.int64)
        tree_offsets = np.r_[0, np.cumsum(padded)].astype(np.int64)
        np.save(dest/'tree_offsets.npy', tree_offsets, allow_pickle=False)
        count, slots = int(offsets[-1]), int(tree_offsets[-1])
        nearest = np.lib.format.open_memmap(dest/'nearest16.npy', mode='w+', dtype=np.uint16,
                                           shape=(count, 128, 16))
        index = np.lib.format.open_memmap(dest/'tree_idx.npy', mode='w+', dtype=np.uint16, shape=(slots,))
        mask = np.lib.format.open_memmap(dest/'tree_mask.npy', mode='w+', dtype=np.bool_, shape=(slots,))
        rotate = [np.lib.format.open_memmap(dest/f'tree_rot{j}.npy', mode='w+', dtype=np.uint16,
                  shape=(slots//2**j,)) for j in range(4)]
        for scene in range(len(offsets)-1):
            lo, hi = map(int, offsets[scene:scene+2])
            ctx = np.asarray(context[lo*128:hi*128])
            cloud, _, queries = placed_geometry(points, normals, np.asarray(gids[lo:hi]),
                                                 rotations[lo:hi], ctx)
            for body in range(hi-lo):
                distances, neighbors = cKDTree(cloud[body]).query(queries[body], k=16, workers=1)
                if not np.isfinite(distances).all() or neighbors.max() >= 2048:
                    raise RuntimeError('Invalid nearest16 geometry mapping')
                nearest[lo+body] = neighbors.astype(np.uint16)
            positions = ctx[:, 2:5]+ctx[:, 5:8]
            idx, kept, rotations_tree = build_balltree_with_rotations(torch.from_numpy(positions.copy()),
                torch.zeros(len(positions), dtype=torch.long), ERWIN_CONFIG['strides'],
                ERWIN_CONFIG['ball_sizes'], ERWIN_CONFIG['rotate'])
            a, b = map(int, tree_offsets[scene:scene+2])
            if len(idx) != b-a or not torch.equal(torch.sort(idx[kept]).values, torch.arange(len(positions))):
                raise RuntimeError('Canonical tree point preservation failed')
            index[a:b], mask[a:b] = idx.numpy().astype(np.uint16), kept.numpy()
            for j, r in enumerate(rotations_tree):
                rotate[j][a//2**j:b//2**j] = r.numpy().astype(np.uint16)
            if (scene+1) % 500 == 0 or scene == len(offsets)-2:
                value = {'split': split, 'scenes': scene+1, 'total': len(offsets)-1,
                         'elapsed_seconds': time.perf_counter()-started, 'updated_at': time.time()}
                write_json(OUT/'prepare_progress.json', value)
                print(json.dumps({'event': 'geo_strong_cache', **value}), flush=True)
        for array in (nearest, index, mask, *rotate):
            array.flush()
        del nearest, index, mask, rotate
        counts[split] = {'scenes': len(offsets)-1, 'objects': count,
                         'canonical_nodes': count*128, 'padded_tree_nodes': slots}
    file_hashes = {p.relative_to(SIDE).as_posix(): sha(p) for p in SIDE.rglob('*.npy')}
    write_json(SIDE/'manifest.json', {'status': 'complete', 'binding': binding, 'splits': counts,
        'files_sha256': file_hashes, 'elapsed_seconds': time.perf_counter()-started,
        'scientific_release': False})


def verify_geometry():
    value = read_json(SIDE/'manifest.json')
    if not value or value['status'] != 'complete' or value['binding'] != input_binding(verify=True):
        raise RuntimeError('Missing source-bound Geo-strong cache')
    for name, expected in value['files_sha256'].items():
        if sha(SIDE/name) != expected:
            raise RuntimeError('Geo-strong cache hash differs: '+name)
    return value


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
    if method != 'erwin':
        raise ValueError(method)
    # Execute the exact existing Geo-strong MLP/pooling/concatenation statements.
    # Only replace its PIBNet-specific wrapper with the existing canonical Erwin.
    definitions = leaf(ROOT/'src/cspgnet/v3_learning.py', ['GeoStrongAdapted'],
        {'torch': torch, 'nn': torch.nn, 'replace': replace, 'PIBNetEdgeIndex': lambda core: core})
    return definitions.GeoStrongAdapted(PointReference('erwin'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepare', action='store_true', required=True)
    parser.parse_args()
    prepare()
