"""Exact object-block Burton--Miller backend; no FEM or modal truncation.

Each block uses Bempp's original P1 Galerkin kernels in complex128. Self
blocks and their LU factors are rigid-motion invariant and cached on disk.
Cross blocks are assembled sequentially and memory mapped; the global
matrix is never materialized. This is NOT an FMM or a low-rank approximation.
"""
from __future__ import annotations
import gc
import hashlib
import importlib.metadata
import json
from pathlib import Path
import time
import numpy as np
from scipy.linalg import lu_factor, lu_solve
from scipy.sparse.linalg import LinearOperator, gmres
from . import formal_neumann_3d as formal
from .structural_bem import available_physical_bytes


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(4*1024**2), b''):
            h.update(chunk)
    return h.hexdigest()


def array_token(*arrays):
    h = hashlib.sha256()
    for array in arrays:
        array = np.ascontiguousarray(array)
        h.update(str((array.shape, array.dtype.str)).encode())
        h.update(array.tobytes())
    return h.hexdigest()


def numerical_identity():
    return {'backend_sha256': digest(__file__),
            'formal_sha256': digest(formal.__file__),
            'packages': {name: importlib.metadata.version(name) for name in
                         ('bempp-cl', 'numpy', 'scipy', 'numba', 'llvmlite')},
            'k': 2., 'precision': 'double', 'interface': 'numba',
            'quadrature': 'bempp-default', 'formulation': "W+2j*Kprime-1j*M"}


def write_json(path, value):
    path = Path(path)
    pending = path.with_suffix(path.suffix+'.pending')
    pending.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')
    pending.replace(path)


def save_array(path, array):
    path = Path(path)
    pending = path.with_suffix(path.suffix+'.pending')
    with pending.open('wb') as stream:
        np.save(stream, array, allow_pickle=False)
    pending.replace(path)


def memory_admission(required, reserve=1024**3):
    available = available_physical_bytes()
    if available < int(required)+int(reserve):
        raise MemoryError(f'block BEM admission: required={required}, available={available}, reserve={reserve}')
    return available


def space_for(points, faces):
    api = formal._bempp_api('dense')
    grid = api.Grid(np.asarray(points, np.float64).T, np.asarray(faces, np.int64).T)
    space = api.function_space(grid, 'P', 1)
    # Exact full-boundary P1 vertex order, explicitly checked rather than assumed.
    if space.global_dof_count != len(points) or not np.array_equal(space.local2global, grid.elements.T):
        raise ValueError('Unsupported P1 vertex/DOF ordering')
    return space


def combined_block(source, target, *, self_block=False, reserve=1024**3):
    api = formal._bempp_api('dense')
    rows, cols = target.global_dof_count, source.global_dof_count
    # At most two dense operators plus in-place combination; retain a
    # conservative six-block allowance for JIT/kernel allocator workspaces.
    memory_admission(6*16*rows*cols + 128*1024**2, reserve)
    kwargs = {'assembler': 'dense', 'device_interface': 'numba', 'precision': 'double'}
    op = api.operators.boundary.helmholtz.hypersingular(source, target, target, 2., **kwargs)
    result = np.array(op.weak_form().to_dense(), dtype=np.complex128, copy=True)
    del op
    gc.collect()
    op = api.operators.boundary.helmholtz.adjoint_double_layer(source, target, target, 2., **kwargs)
    other = op.weak_form().to_dense()
    # Row-wise scaling avoids allocating a third full dense matrix.
    for row in range(rows):
        result[row] += 2j*other[row]
    del other, op
    if self_block:
        mass = api.operators.boundary.sparse.identity(source, target, target).weak_form().to_sparse().tocoo()
        result[mass.row, mass.col] -= 1j*mass.data
    if not np.isfinite(result).all():
        raise FloatingPointError('Nonfinite Burton--Miller block')
    gc.collect()
    return result


def prepare_self(points, faces, cache_root, *, reserve=1024**3):
    """Immutable, source-bound base-body block and LU; retry partial caches safely."""
    key = array_token(np.asarray(points, np.float64), np.asarray(faces, np.int64))
    root = Path(cache_root)/key
    root.mkdir(parents=True, exist_ok=True)
    expected = {'grid_sha256': key, 'identity': numerical_identity()}
    receipt = root/'receipt.json'
    if receipt.exists():
        saved = json.loads(receipt.read_text(encoding='utf-8'))
        if saved['binding'] != expected:
            raise ValueError('Self BEM cache source identity mismatch')
        for name, sha in saved['files'].items():
            if digest(root/name) != sha:
                raise ValueError('Self BEM cache checksum mismatch')
    else:
        if any(root.iterdir()):
            raise RuntimeError(f'Incomplete self cache retained: {root}; use a fresh evidence version')
        started = time.perf_counter()
        space = space_for(points, faces)
        matrix = combined_block(space, space, self_block=True, reserve=reserve)
        save_array(root/'matrix.npy', matrix)
        memory_admission(3*matrix.nbytes, reserve)
        factor, pivots = lu_factor(matrix, overwrite_a=True, check_finite=False)
        save_array(root/'lu.npy', factor)
        save_array(root/'pivots.npy', pivots)
        write_json(receipt, {'binding': expected, 'seconds': time.perf_counter()-started,
                            'files': {name: digest(root/name) for name in ('matrix.npy', 'lu.npy', 'pivots.npy')}})
        del factor, matrix, space
        gc.collect()
    return {'key': key, 'matrix': np.load(root/'matrix.npy', mmap_mode='r'),
            'lu': np.load(root/'lu.npy', mmap_mode='r'),
            # Windows SciPy's LAPACK wrapper temporarily shifts zero-based
            # pivots in place. Read-only mapped pivot vectors cause a native
            # access violation; keep this tiny integer vector writable.
            'pivots': np.array(np.load(root/'pivots.npy'), dtype=np.int32, copy=True),
            'receipt': str(receipt), 'receipt_sha256': digest(receipt)}


def block_operator(blocks, sizes):
    offsets = np.r_[0, np.cumsum(sizes)]
    def matvec(vector):
        result = np.zeros(offsets[-1], dtype=np.complex128)
        for i in range(len(sizes)):
            for j in range(len(sizes)):
                result[offsets[i]:offsets[i+1]] += blocks[i][j] @ vector[offsets[j]:offsets[j+1]]
        return result
    return LinearOperator((offsets[-1], offsets[-1]), matvec=matvec, dtype=np.complex128)


def solve_meshes(base_meshes, centers, rotations, direction, interface, cache_root,
                 cross_root, *, tolerance=1e-8, reserve=1024**3, benchmark_reference=False):
    """Independent full BEM, original weak residual, block-Jacobi left GMRES."""
    started = time.perf_counter()
    api = formal._bempp_api('dense')
    centers, rotations, direction = map(lambda a: np.asarray(a, np.float64), (centers, rotations, direction))
    if not np.isclose(np.linalg.norm(direction), 1.) or len(base_meshes) != len(centers):
        raise ValueError('Invalid scene')
    for rotation in rotations:
        if not np.allclose(rotation.T@rotation, np.eye(3), atol=1e-12) or not np.isclose(np.linalg.det(rotation), 1.):
            raise ValueError('Rotation must be SO(3)')
    shared = {}; self_blocks = []; spaces = []; world_meshes = []
    for (points, faces), center, rotation in zip(base_meshes, centers, rotations):
        key = array_token(np.asarray(points, np.float64), np.asarray(faces, np.int64))
        if key not in shared:
            shared[key] = prepare_self(points, faces, cache_root, reserve=reserve)
        self_blocks.append(shared[key])
        world = np.asarray(points, np.float64)@rotation.T + center
        world_meshes.append((world, faces))
        spaces.append(space_for(world, faces))
    sizes = [space.global_dof_count for space in spaces]
    offsets = np.r_[0, np.cumsum(sizes)]
    root = Path(cross_root)
    root.mkdir(parents=True, exist_ok=True)
    blocks = [[None]*len(spaces) for _ in spaces]
    cross_seconds = time.perf_counter()
    for i, target in enumerate(spaces):
        for j, source in enumerate(spaces):
            if i == j:
                blocks[i][j] = self_blocks[i]['matrix']
                continue
            path = root/f'block_{i}_{j}.npy'
            if path.exists():
                raise FileExistsError(f'Cross-block evidence already exists: {path}')
            matrix = combined_block(source, target, reserve=reserve)
            save_array(path, matrix)
            del matrix
            gc.collect()
            blocks[i][j] = np.load(path, mmap_mode='r')
    cross_seconds = time.perf_counter()-cross_seconds
    print(json.dumps({'phase': 'cross_assembled', 'vertices': sizes, 'seconds': cross_seconds}), flush=True)

    @api.complex_callable
    def normal_incident(x, n, domain_index, result):
        result[0] = 2j*np.dot(direction, n)*np.exp(2j*np.dot(direction, x))

    rhs = np.concatenate([api.GridFunction(space, fun=normal_incident).projections(space) for space in spaces])
    print(json.dumps({'phase': 'rhs_projected'}), flush=True)
    lhs = block_operator(blocks, sizes)
    def precondition(vector):
        result = np.empty_like(vector)
        for i, cached in enumerate(self_blocks):
            result[offsets[i]:offsets[i+1]] = lu_solve((cached['lu'], cached['pivots']),
                vector[offsets[i]:offsets[i+1]], check_finite=False)
        return result
    preconditioner = LinearOperator(lhs.shape, matvec=precondition, dtype=np.complex128)
    residuals = []
    solve_start = time.perf_counter()
    coefficients, info = gmres(lhs, rhs, M=preconditioner, rtol=tolerance, atol=0., restart=50,
                               maxiter=1000, callback=lambda r: residuals.append(float(r)), callback_type='pr_norm')
    solve_seconds = time.perf_counter()-solve_start
    actual = float(np.linalg.norm(lhs@coefficients-rhs)/np.linalg.norm(rhs))
    print(json.dumps({'phase': 'gmres_completed', 'iterations': len(residuals), 'weak_residual': actual}), flush=True)
    if info != 0 or actual > tolerance:
        raise ArithmeticError(f'Block BEM GMRES failed: info={info}, weak residual={actual}')
    observations = np.concatenate([np.asarray(interface)+center for center in centers]).T
    scattered = np.zeros(observations.shape[1], np.complex128)
    kwargs = {'assembler': 'dense', 'device_interface': 'numba', 'precision': 'double'}
    potential_start = time.perf_counter()
    for i, space in enumerate(spaces):
        density = api.GridFunction(space, coefficients=coefficients[offsets[i]:offsets[i+1]])
        d = api.operators.potential.helmholtz.double_layer(space, observations, 2., **kwargs)
        scattered += d.evaluate(density).ravel()
        del d
        s = api.operators.potential.helmholtz.single_layer(space, observations, 2., **kwargs)
        scattered -= 2j*s.evaluate(density).ravel()
        del s
    total = formal.incident_on_interfaces(centers, interface, direction, 2.)+scattered.reshape(len(centers), -1)
    print(json.dumps({'phase': 'potential_completed'}), flush=True)
    metadata = {'backend': 'exact-object-block-bem-v1', 'identity': numerical_identity(),
                'gmres_info': int(info), 'gmres_iterations': len(residuals),
                'actual_weak_relative_residual': actual, 'self_cache':
                [{k: v for k, v in row.items() if k in ('key', 'receipt', 'receipt_sha256')} for row in self_blocks],
                'vertices_per_body': sizes, 'surface_vertices': int(offsets[-1]),
                'cross_assembly_seconds': cross_seconds, 'gmres_seconds': solve_seconds,
                'potential_seconds': time.perf_counter()-potential_start,
                'seconds': time.perf_counter()-started, 'independent_of_fem': True,
                'no_low_rank_or_multipole_truncation': True,
                'matrix_disk_bytes': sum(blocks[i][j].nbytes for i in range(len(sizes)) for j in range(len(sizes))),
                'cross_files': {p.name: digest(p) for p in root.glob('block_*.npy')}}
    if benchmark_reference:
        points = np.concatenate([mesh[0] for mesh in world_meshes])
        faces = np.concatenate([mesh[1]+offsets[i] for i, mesh in enumerate(world_meshes)])
        global_space = space_for(points, faces)
        reference_start = time.perf_counter()
        reference = combined_block(global_space, global_space, self_block=True, reserve=reserve)
        trial = np.random.default_rng(0).normal(size=offsets[-1])+1j*np.random.default_rng(1).normal(size=offsets[-1])
        metadata['dense_matrix_action_relative_error'] = float(np.linalg.norm(lhs@trial-reference@trial)/np.linalg.norm(reference@trial))
        reference_residuals = []
        reference_coeff, ref_info = gmres(reference, rhs, rtol=tolerance, atol=0.,
            callback=lambda r: reference_residuals.append(float(r)), callback_type='pr_norm')
        metadata.update(dense_reference_info=int(ref_info), dense_reference_iterations=len(reference_residuals),
            dense_reference_seconds=time.perf_counter()-reference_start,
            density_relative_difference=float(np.linalg.norm(reference_coeff-coefficients)/np.linalg.norm(reference_coeff)),
            dense_reference_weak_relative_residual=float(np.linalg.norm(reference@reference_coeff-rhs)/np.linalg.norm(rhs)))
        density = api.GridFunction(global_space, coefficients=reference_coeff)
        d = api.operators.potential.helmholtz.double_layer(global_space, observations, 2., **kwargs)
        s = api.operators.potential.helmholtz.single_layer(global_space, observations, 2., **kwargs)
        reference_scatter = (d.evaluate(density)-2j*s.evaluate(density)).ravel()
        metadata['scattered_field_dense_relative_error'] = float(np.linalg.norm(scattered-reference_scatter)/np.linalg.norm(reference_scatter))
    if not np.isfinite(total).all():
        raise FloatingPointError('Nonfinite block BEM field')
    return total, metadata
