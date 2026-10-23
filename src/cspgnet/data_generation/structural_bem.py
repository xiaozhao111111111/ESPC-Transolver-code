"""Independent exterior Burton--Miller BEM for verified structural solids.

No local FEM/DtN data is used by this solver. Concave/hole-wall normals are
checked by the exact block-union occupancy, never by a radial normal test.
"""
from __future__ import annotations
from dataclasses import dataclass
import ctypes
import os
import time
import numpy as np
from . import formal_neumann_3d as formal


@dataclass(frozen=True)
class OCCSolidSpec:
    """An already-audited local OCC solid for independent BEM meshing."""

    geometry_id: str
    solid: object
    genus: int


class _MemoryStatus(ctypes.Structure):
    _fields_ = [
        ("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
        ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
        ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
        ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
        ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
    ]


def available_physical_bytes() -> int:
    """Return system-wide available physical memory on supported hosts."""
    try:
        import psutil

        return int(psutil.virtual_memory().available)
    except ImportError:
        if os.name == "nt":
            status = _MemoryStatus()
            status.dwLength = ctypes.sizeof(status)
            if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                raise ctypes.WinError()
            return int(status.ullAvailPhys)
        page_size = int(os.sysconf("SC_PAGE_SIZE"))
        available_pages = int(os.sysconf("SC_AVPHYS_PAGES"))
        return page_size * available_pages


def dense_assembly_estimate_bytes(vertex_count: int) -> int:
    """Conservative estimate used before constructing dense BEM operators."""
    if int(vertex_count) < 0:
        raise ValueError("vertex_count must be nonnegative")
    # Eight simultaneous complex128 dense work arrays. This guards admission,
    # but is not a guarantee that allocator/JIT/potential memory is smaller.
    return 8 * 16 * int(vertex_count) ** 2


def _orient_closed_surface(points: np.ndarray, faces: np.ndarray) -> tuple[np.ndarray, int, float]:
    """Orient a closed triangle manifold consistently and choose positive volume."""
    face_count = len(faces)
    directed = np.stack(
        (faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]), axis=1
    )
    canonical = np.sort(directed, axis=2)
    edge_keys, inverse, counts = np.unique(
        canonical.reshape(-1, 2), axis=0, return_inverse=True, return_counts=True
    )
    if np.any(counts != 2):
        bad = np.unique(counts, return_counts=True)
        raise RuntimeError(f"BEM surface is not closed 2-manifold; edge multiplicities={bad}")

    face_ids = np.repeat(np.arange(face_count, dtype=np.int64), 3)
    edge_signs = np.where(
        directed.reshape(-1, 2)[:, 0] < directed.reshape(-1, 2)[:, 1], 1, -1
    ).astype(np.int8)
    order = np.argsort(inverse, kind="stable")
    grouped_faces = face_ids[order].reshape(-1, 2)
    grouped_signs = edge_signs[order].reshape(-1, 2)

    adjacency: list[list[tuple[int, int]]] = [[] for _ in range(face_count)]
    relation = (-grouped_signs[:, 0] * grouped_signs[:, 1]).astype(np.int8)
    for (left, right), sign in zip(grouped_faces, relation):
        adjacency[int(left)].append((int(right), int(sign)))
        adjacency[int(right)].append((int(left), int(sign)))

    orientation = np.zeros(face_count, dtype=np.int8)
    components = 0
    for root in range(face_count):
        if orientation[root] != 0:
            continue
        components += 1
        orientation[root] = 1
        stack = [root]
        while stack:
            face = stack.pop()
            for neighbor, sign in adjacency[face]:
                required = int(orientation[face]) * sign
                if orientation[neighbor] == 0:
                    orientation[neighbor] = required
                    stack.append(neighbor)
                elif orientation[neighbor] != required:
                    raise RuntimeError("BEM surface triangulation is not orientable")

    oriented = np.array(faces, dtype=np.int64, copy=True)
    reverse = orientation < 0
    oriented[reverse, 1], oriented[reverse, 2] = (
        oriented[reverse, 2].copy(), oriented[reverse, 1].copy()
    )
    a, b, c = (points[oriented[:, j]] for j in range(3))
    double_volume = np.einsum("ij,ij->i", a, np.cross(b, c)).sum()
    signed_volume = float(double_volume / 6.0)
    if not np.isfinite(signed_volume) or abs(signed_volume) <= 1.0e-14:
        raise RuntimeError("BEM surface has zero or non-finite oriented volume")
    if signed_volume < 0:
        oriented[:, [1, 2]] = oriented[:, [2, 1]]
        signed_volume = -signed_volume
    return oriented, components, signed_volume


def surface_arrays_for_solid(
    solid,
    center,
    h: float,
    *,
    expected_genus: int,
    geometry_id: str,
):
    """Create an independently oriented BEM mesh from a closed OCC solid.

    Normal orientation is established from a globally consistent closed
    manifold, then selected by positive signed volume. This works for inner
    bore walls as well as outer walls; it does not use radial-from-center
    normals or any FEM/DtN data.
    """
    if not np.isfinite(h) or h <= 0:
        raise ValueError("BEM surface mesh size h must be positive and finite")
    mesh = formal.OCCGeometry(solid).GenerateMesh(maxh=float(h))
    points = np.asarray([tuple(point.p) for point in mesh.Points()], dtype=np.float64)
    faces = np.asarray(
        [[vertex.nr - 1 for vertex in element.vertices] for element in mesh.Elements2D()],
        dtype=np.int64,
    )
    if faces.ndim != 2 or faces.shape[1] != 3 or len(faces) == 0:
        raise RuntimeError("OCC solid did not produce a nonempty triangular BEM boundary")
    if not np.isfinite(points).all() or np.any(faces < 0) or np.any(faces >= len(points)):
        raise RuntimeError("OCC BEM mesh contains invalid vertices or indices")

    used, inverse = np.unique(faces.reshape(-1), return_inverse=True)
    points = points[used]
    faces = inverse.reshape(faces.shape)
    a, b, c = (points[faces[:, j]] for j in range(3))
    double_area = np.linalg.norm(np.cross(b - a, c - a), axis=1)
    if np.any(double_area <= 1.0e-14):
        raise RuntimeError("OCC BEM mesh contains a degenerate triangle")

    faces, components, surface_volume = _orient_closed_surface(points, faces)
    euler = int(len(points) - len(np.unique(np.sort(np.concatenate(
        (faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]), axis=0
    ), axis=1), axis=0)) + len(faces))
    genus = (2 - euler) / 2
    if components != 1 or not float(genus).is_integer():
        raise RuntimeError(
            f"OCC BEM boundary must be one connected orientable closed surface; "
            f"components={components}, Euler={euler}"
        )
    genus = int(genus)
    if genus != int(expected_genus):
        raise RuntimeError(
            f"OCC BEM surface genus mismatch for {geometry_id}: {genus} != {expected_genus}"
        )
    cad_volume = float(solid.mass)
    if not np.isfinite(cad_volume) or cad_volume <= 0:
        raise RuntimeError(f"OCC solid has invalid CAD volume for {geometry_id}")
    relative_volume_error = abs(surface_volume - cad_volume) / cad_volume
    audit = {
        "geometry_id": geometry_id,
        "vertices": int(len(points)),
        "triangles": int(len(faces)),
        "edges": int(len(np.unique(np.sort(np.concatenate(
            (faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]), axis=0
        ), axis=1), axis=0))),
        "connected_components": components,
        "euler_characteristic": euler,
        "genus": genus,
        "mesh_oriented_volume": surface_volume,
        "cad_solid_volume": cad_volume,
        "relative_volume_difference": float(relative_volume_error),
        "verified_outward_normals": True,
        "mesh_h": float(h),
    }
    shifted = points + np.asarray(center, dtype=np.float64)[None, :]
    return shifted, faces, audit


def surface_arrays(spec: "StructuralSpec", center, h: float):
    # Keep the old voxel-family dependency out of the OCC-solid V6 path.
    from .structural_voxels import make_solid, audit_surface

    mesh = formal.OCCGeometry(make_solid(spec)).GenerateMesh(maxh=h)
    points = np.asarray([tuple(p.p) for p in mesh.Points()], dtype=np.float64)
    faces = np.asarray([[v.nr-1 for v in e.vertices] for e in mesh.Elements2D()], dtype=np.int64)
    a, b, c = (points[faces[:, j]] for j in range(3))
    cross = np.cross(b-a, c-a)
    norm = np.linalg.norm(cross, axis=1)
    if np.any(norm <= 1e-14):
        raise RuntimeError("degenerate BEM surface element")
    normal, middle = cross/norm[:, None], (a+b+c)/3
    delta = spec.cell_width*1e-4
    plus = spec.contains(middle+delta*normal, tolerance=1e-12)
    minus = spec.contains(middle-delta*normal, tolerance=1e-12)
    if np.any(plus == minus):
        raise RuntimeError("BEM inside/outside normal check failed")
    reverse = plus & ~minus
    faces[reverse, 1], faces[reverse, 2] = faces[reverse, 2].copy(), faces[reverse, 1].copy()
    used, inverse = np.unique(faces, return_inverse=True)
    points, faces = points[used], inverse.reshape(faces.shape)
    audit = audit_surface(points, faces)
    if not audit['passed'] or audit['genus'] != spec.genus:
        raise RuntimeError(f"BEM topology differs from frozen geometry: {audit}")
    expected = len(spec.cells)*spec.cell_width**3
    if not np.isclose(audit['oriented_volume'], expected, rtol=1e-7):
        raise RuntimeError("BEM surface volume differs from CAD manifest")
    audit.update(verified_outward_normals=True, mesh_h=h)
    return points+np.asarray(center)[None], faces, audit


def solve(objects, centers, direction, interface, h, *, gmres_tolerance=1e-8,
          maximum_dense_bytes=6*1024**3, minimum_available_bytes=1024**3):
    """Accept StructuralSpec or the parent's ObjectSpec; return total field."""
    started = time.perf_counter()
    api = formal._bempp_api('dense')
    all_points, all_faces, domains, audits = [], [], [], []
    offset = 0
    for index, (spec, center) in enumerate(zip(objects, centers), 1):
        if isinstance(spec, OCCSolidSpec):
            points, faces, audit = surface_arrays_for_solid(
                spec.solid, center, h,
                expected_genus=spec.genus,
                geometry_id=spec.geometry_id,
            )
        else:
            from .structural_voxels import StructuralSpec
            if isinstance(spec, StructuralSpec):
                points, faces, audit = surface_arrays(spec, center, h)
            else:
                points, faces = formal._netgen_surface_arrays(spec, center, h)
                used, inverse = np.unique(faces, return_inverse=True)
                points, faces = points[used], inverse.reshape(faces.shape)
                audit = {'shape':spec.shape, 'vertices':len(points), 'triangles':len(faces)}
        all_points.append(points); all_faces.append(faces+offset)
        domains.append(np.full(len(faces), index, dtype=np.uint32))
        audits.append(audit); offset += len(points)
    # Conservative eight simultaneous complex dense matrices; preflight only,
    # not a guarantee that allocator/JIT/potential memory cannot be larger.
    estimate = dense_assembly_estimate_bytes(offset)
    if estimate > maximum_dense_bytes:
        raise MemoryError(f"dense BEM preflight: {offset} vertices, estimate {estimate} bytes exceeds budget")
    available_before_assembly = available_physical_bytes()
    if estimate + int(minimum_available_bytes) > available_before_assembly:
        raise MemoryError(
            "dense BEM preflight would violate the system-memory reserve: "
            f"vertices={offset}, estimate={estimate}, available={available_before_assembly}, "
            f"reserve={int(minimum_available_bytes)}"
        )
    grid = api.Grid(np.concatenate(all_points).T, np.concatenate(all_faces).T, np.concatenate(domains))
    space = api.function_space(grid, 'P', 1)
    kwargs = {'assembler':'dense'}
    identity = api.operators.boundary.sparse.identity(space, space, space)
    w = api.operators.boundary.helmholtz.hypersingular(space, space, space, 2., **kwargs)
    kp = api.operators.boundary.helmholtz.adjoint_double_layer(space, space, space, 2., **kwargs)
    lhs = w+2j*(-.5*identity+kp)
    direction = np.asarray(direction, dtype=np.float64)
    if not np.isclose(np.linalg.norm(direction), 1.):
        raise ValueError('incident direction must have unit length')

    @api.complex_callable
    def normal_incident(x, n, domain_index, result):
        result[0] = 2j*np.dot(direction,n)*np.exp(2j*np.dot(direction,x))

    rhs = api.GridFunction(space, fun=normal_incident)
    print(f'BEM assemble h={h} vertices={offset} triangles={grid.number_of_elements} dense_budget_estimate={estimate}', flush=True)
    density, info, residuals, iterations = api.linalg.gmres(
        lhs, rhs, tol=gmres_tolerance, return_residuals=True, return_iteration_count=True)
    weak_rhs = rhs.projections(space)
    actual_residual = float(np.linalg.norm(lhs.weak_form() @ density.coefficients-weak_rhs)
                            /max(np.linalg.norm(weak_rhs), 1e-30))
    metadata = {'formulation':"burton-miller:D-i*k*S;W+i*k*(-1/2I+K')=d_n(u_inc)",
                'assembler':'dense', 'gmres_info':int(info), 'gmres_iterations':int(iterations),
                'gmres_tolerance':gmres_tolerance,
                'gmres_final_residual':float(residuals[-1]) if residuals else actual_residual,
                'actual_weak_relative_residual':actual_residual, 'mesh_h':h,
                'surface_vertices':offset, 'surface_elements':int(grid.number_of_elements),
                'dense_memory_estimate_bytes':estimate,
                'system_available_before_assembly_bytes':available_before_assembly,
                'system_memory_reserve_bytes':int(minimum_available_bytes),
                'surfaces':audits}
    if info != 0 or actual_residual > gmres_tolerance:
        raise RuntimeError(f'BEM solve failed: {metadata}')
    points = np.concatenate([interface+center[None] for center in centers]).T
    d = api.operators.potential.helmholtz.double_layer(space, points, 2., **kwargs)
    s = api.operators.potential.helmholtz.single_layer(space, points, 2., **kwargs)
    scattered = (d.evaluate(density)-2j*s.evaluate(density)).reshape(len(objects),len(interface))
    total = formal.incident_on_interfaces(np.asarray(centers),interface,direction,2.)+scattered
    if not np.isfinite(total).all():
        raise FloatingPointError('nonfinite BEM evaluation')
    metadata.update(seconds=time.perf_counter()-started, solve_status='converged',
                    independent_of_local_dtn=True)
    return total, metadata
