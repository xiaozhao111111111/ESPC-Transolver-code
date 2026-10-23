"""Formal 3-D Neumann CSP-GNet data generation.

This module is deliberately separate from :mod:`canonical_3d`.  The latter is
kept as a low-fidelity prototype so that existing experiments remain
reproducible.  This implementation provides the versioned, convergence-gated
pipeline used by the formal experiment:

* a fixed spherical interface with 128 Fibonacci points;
* 64 spherical-harmonic Dirichlet probes (``l <= 7``);
* geometry-only projected local Schur/DtN operators, assembled with a
  consistent boundary mass solve and a spherical-harmonic L2 projection;
* exterior sound-hard labels from a Burton--Miller boundary integral equation;
* deterministic, independently sampled train/validation/test scene families.

Large data generation must only be started after a combined graded-FEM and
independent-BEM release report passes. The legacy uniform-mesh
``numerical_gate.py`` cannot authorize generation.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMBA_NUM_THREADS", "1")
os.environ.setdefault("NUMBA_THREADING_LAYER", "workqueue")
# Bempp's Numba cache must be writable.  In restricted Windows environments,
# writing next to the installed package can otherwise retry for minutes.
os.environ.setdefault(
    "NUMBA_CACHE_DIR",
    str(Path(__file__).resolve().parents[3] / "outputs" / "numba_cache"),
)
if sys.platform == "win32":
    _omp_dll_dir = (
        Path(__file__).resolve().parents[3]
        / "work"
        / "openmp_runtime_20260920"
        / "package"
        / "runtimes"
        / "win-x64"
        / "native"
    )
    if _omp_dll_dir.is_dir():
        os.add_dll_directory(str(_omp_dll_dir))

import numpy as np
from scipy.spatial import ConvexHull
try:
    from scipy.special import sph_harm_y

    def _spherical_harmonic(ell, order, polar, azimuth):
        return sph_harm_y(ell, order, polar, azimuth)
except ImportError:  # SciPy < 1.17
    from scipy.special import sph_harm

    def _spherical_harmonic(ell, order, polar, azimuth):
        return sph_harm(order, ell, azimuth, polar)

from netgen.occ import Axis, Axes, Box, Circle, Ellipsoid, Face, OCCGeometry, Pnt, Segment, Sphere, SplineInterpolation, Vec, Wire
from scipy.optimize import brentq
from ngsolve import (
    BilinearForm,
    GridFunction,
    H1,
    LinearForm,
    Mesh,
    MultiVector,
    NodeId,
    VERTEX,
    dx,
    grad,
    ds,
)
from ngsolve.bem import SphericalHarmonicsCF


SCHEMA_VERSION = "cspgnet-neumann-3d-v2-graded"
DEFAULT_K = 2.0
DEFAULT_CANONICAL_RADIUS = 1.5
DEFAULT_INTERFACE_NODES = 128
DEFAULT_PROBE_LMAX = 7
DEFAULT_FLUX_LMAX = 10
E1_CANDIDATE_GRADED_PROFILES = {
    "sphere": "G1",
    "ellipsoid": "G0",
    "box": "G0",
    "peanut": "G0",
    "torus": "G0",
}
DEFAULT_SCHUR_BATCH_SIZE = 16
DEFAULT_GMRES_TOL = 1.0e-8
SHAPE_NAMES = ("sphere", "ellipsoid", "box")
_BEMPP_DENSE_API = None
_BEMPP_DENSE_ONLY_IMPORT = False


def _bempp_api(assembler: str = "dense"):
    """Import Bempp while avoiding JIT compilation of its unused dense-path octree.

    Bempp imports its octree module even when dense assembly is selected.  The
    octree class is only used by the FMM path, but Numba compiles it at import
    time.  Keep the production Numba kernels enabled and defer just that
    unused class compilation for the dense CPU backend.
    """
    global _BEMPP_DENSE_API, _BEMPP_DENSE_ONLY_IMPORT
    if _BEMPP_DENSE_API is not None:
        if assembler == "fmm" and _BEMPP_DENSE_ONLY_IMPORT:
            raise RuntimeError(
                "FMM cannot be used after the dense-only Bempp import; start a fresh process"
            )
        return _BEMPP_DENSE_API
    if assembler == "dense":
        import sys
        import types

        # Bempp imports its Octree class unconditionally, although dense
        # assembly never references it.  Avoid Numba's eager compilation of
        # that unused class while preserving normal JIT kernels for assembly.
        octree_module_name = "bempp_cl.api.utils.octree"
        if octree_module_name not in sys.modules:
            octree_stub = types.ModuleType(octree_module_name)
            octree_stub.Octree = type("DenseOnlyOctreeStub", (), {})
            sys.modules[octree_module_name] = octree_stub
        import bempp_cl.api as bempp_api

        _BEMPP_DENSE_ONLY_IMPORT = True
    else:
        import bempp_cl.api as bempp_api
    _BEMPP_DENSE_API = bempp_api
    return bempp_api


@dataclass(frozen=True)
class ObjectSpec:
    """Actual local geometry used by both the FEM and BEM solvers."""

    shape: str
    size: tuple[float, ...]
    rotation: tuple[tuple[float, float, float], ...]
    axis_permutation: tuple[int, ...] = ()
    smoothing_beta: float | None = None
    meridian_points: int = 65


@dataclass(frozen=True)
class SampleSpec:
    index: int
    split: str
    sample_seed: int
    k: float
    objects: tuple[ObjectSpec, ...]
    centers: tuple[tuple[float, float, float], ...]
    incident_direction: tuple[float, float, float]


def fibonacci_sphere(num_points: int, radius: float) -> np.ndarray:
    indices = np.arange(num_points, dtype=np.float64) + 0.5
    z_coord = 1.0 - 2.0 * indices / num_points
    radial = np.sqrt(np.maximum(0.0, 1.0 - z_coord * z_coord))
    golden = math.pi * (3.0 - math.sqrt(5.0))
    azimuth = golden * indices
    return radius * np.column_stack(
        (radial * np.cos(azimuth), radial * np.sin(azimuth), z_coord)
    )


def spherical_modes(lmax: int) -> list[tuple[int, int]]:
    return [(ell, order) for ell in range(lmax + 1) for order in range(-ell, ell + 1)]


def spherical_harmonic_matrix(
    points: np.ndarray,
    modes: Sequence[tuple[int, int]],
) -> np.ndarray:
    unit = points / np.linalg.norm(points, axis=1, keepdims=True)
    polar = np.arccos(np.clip(unit[:, 2], -1.0, 1.0))
    azimuth = np.arctan2(unit[:, 1], unit[:, 0])
    columns = []
    for ell, order in modes:
        # NGSolve's SphericalHarmonicsCF omits the Condon-Shortley factor for
        # positive m while retaining the standard negative-m functions.
        phase = -1.0 if order > 0 and order % 2 else 1.0
        columns.append(phase * _spherical_harmonic(ell, order, polar, azimuth))
    return np.column_stack(columns).astype(np.complex128, copy=False)


def canonical_surface(
    num_points: int = DEFAULT_INTERFACE_NODES,
    radius: float = DEFAULT_CANONICAL_RADIUS,
) -> tuple[np.ndarray, np.ndarray]:
    points = fibonacci_sphere(num_points, radius)
    hull = ConvexHull(points)
    faces = np.asarray(hull.simplices, dtype=np.int32)
    return points, faces


def equal_area_weights(num_points: int, radius: float) -> np.ndarray:
    return np.full(num_points, 4.0 * math.pi * radius * radius / num_points)


def weighted_pseudoinverse(phi: np.ndarray, weights: np.ndarray) -> np.ndarray:
    gram = phi.conj().T @ (weights[:, None] * phi)
    return np.linalg.solve(gram, phi.conj().T * weights[None, :])


def project_flux(z_response: np.ndarray, boundary_values: np.ndarray, phi_pinv: np.ndarray) -> np.ndarray:
    """Apply the stored rank-64 DtN response to interface values."""
    coefficients = np.einsum("mn,on->om", phi_pinv, boundary_values, optimize=True)
    return np.einsum("onm,om->on", z_response, coefficients, optimize=True)


def random_rotation_matrix(rng: np.random.Generator) -> np.ndarray:
    """Haar-uniform SO(3) rotation via a unit quaternion."""
    u1, u2, u3 = rng.random(3)
    x = math.sqrt(1.0 - u1) * math.sin(2.0 * math.pi * u2)
    y = math.sqrt(1.0 - u1) * math.cos(2.0 * math.pi * u2)
    z = math.sqrt(u1) * math.sin(2.0 * math.pi * u3)
    w = math.sqrt(u1) * math.cos(2.0 * math.pi * u3)
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _axis_angle(rotation: np.ndarray) -> tuple[np.ndarray, float]:
    cosine = float(np.clip((np.trace(rotation) - 1.0) * 0.5, -1.0, 1.0))
    angle = math.acos(cosine)
    if angle < 1.0e-12:
        return np.asarray([1.0, 0.0, 0.0]), 0.0
    if abs(math.pi - angle) < 1.0e-7:
        eigenvalues, eigenvectors = np.linalg.eig(rotation)
        axis = np.real(eigenvectors[:, np.argmin(np.abs(eigenvalues - 1.0))])
    else:
        axis = np.asarray(
            [
                rotation[2, 1] - rotation[1, 2],
                rotation[0, 2] - rotation[2, 0],
                rotation[1, 0] - rotation[0, 1],
            ]
        ) / (2.0 * math.sin(angle))
    axis /= np.linalg.norm(axis)
    return axis, math.degrees(angle)


def _identity_rotation() -> tuple[tuple[float, float, float], ...]:
    return tuple(tuple(float(value) for value in row) for row in np.eye(3))


def random_object(rng: np.random.Generator) -> ObjectSpec:
    shape = str(rng.choice(SHAPE_NAMES))
    if shape == "sphere":
        size = (float(rng.uniform(0.25, 0.55)),)
        rotation = np.eye(3)
        axis_permutation = (0,)
    elif shape == "ellipsoid":
        raw_size = rng.uniform(0.25, 0.55, 3)
        permutation = np.argsort(-raw_size)
        # Store both the permutation and the sorted axes actually passed to
        # OCC.  This makes the manifest an exact description of the geometry.
        size = tuple(float(raw_size[index]) for index in permutation)
        rotation = random_rotation_matrix(rng)
        axis_permutation = tuple(int(index) for index in permutation)
    else:
        while True:
            candidate = rng.uniform(0.22, 0.50, 3)
            if np.linalg.norm(candidate) <= 0.9:
                size = tuple(float(v) for v in candidate)
                break
        rotation = random_rotation_matrix(rng)
        axis_permutation = (0, 1, 2)
    return ObjectSpec(
        shape=shape,
        size=size,
        rotation=tuple(tuple(float(value) for value in row) for row in rotation),
        axis_permutation=axis_permutation,
    )


def make_obstacle(spec: ObjectSpec, center: Sequence[float] = (0.0, 0.0, 0.0)):
    if spec.shape == "sphere":
        shape = Sphere((0.0, 0.0, 0.0), float(spec.size[0]))
    elif spec.shape == "ellipsoid":
        shape = Ellipsoid(Axes((0.0, 0.0, 0.0), (0.0, 0.0, 1.0)), *spec.size)
    elif spec.shape == "box":
        hx, hy, hz = spec.size
        shape = Box((-hx, -hy, -hz), (hx, hy, hz))
    elif spec.shape == "peanut":
        shape = _make_peanut(spec)
    elif spec.shape == "torus":
        if len(spec.size) != 2:
            raise ValueError("torus size must be (major_radius, tube_radius)")
        major, tube = map(float, spec.size)
        if not (0.45 <= major <= 0.60 and 0.14 <= tube <= 0.24):
            raise ValueError("torus radii are outside the registered range")
        circle = Circle(Pnt(major, 0.0, 0.0), Vec(0.0, 1.0, 0.0), tube)
        shape = Face(Wire([circle])).Revolve(Axis((0, 0, 0), (0, 0, 1)), 360)
    else:
        raise ValueError(f"unsupported shape: {spec.shape}")

    if spec.shape != "sphere":
        axis, angle_degrees = _axis_angle(np.asarray(spec.rotation, dtype=np.float64))
        if angle_degrees > 1.0e-10:
            shape = shape.Rotate(Axis((0.0, 0.0, 0.0), tuple(axis)), angle_degrees)
    if np.linalg.norm(center) > 0.0:
        shape = shape.Move(Vec(*(float(value) for value in center)))
    return shape


def peanut_sdf(radius: np.ndarray | float, height: np.ndarray | float,
               r1: float, r2: float, separation: float, beta: float = 30.0):
    """Stable smooth minimum of two axis-aligned sphere signed distances."""
    d1 = np.hypot(radius, np.asarray(height) + separation / 2) - r1
    d2 = np.hypot(radius, np.asarray(height) - separation / 2) - r2
    return -np.logaddexp(-beta * d1, -beta * d2) / beta


def peanut_meridian(spec: ObjectSpec) -> np.ndarray:
    """Sample the zero SDF meridian, including its two axis endpoints."""
    if spec.shape != "peanut" or len(spec.size) != 3:
        raise ValueError("peanut size must be (r1, r2, center_distance)")
    r1, r2, distance = map(float, spec.size)
    beta = 30.0 if spec.smoothing_beta is None else float(spec.smoothing_beta)
    if not (0.30 <= r1 <= 0.45 and 0.30 <= r2 <= 0.45):
        raise ValueError("peanut sphere radii are outside the registered range")
    if not (0.45 <= distance <= min(0.70, 0.8 * (r1 + r2))):
        raise ValueError("peanut center distance is outside the registered range")
    if beta != 30.0 or spec.meridian_points not in (65, 129, 257):
        raise ValueError("peanut beta/meridian resolution differs from the registered protocol")
    lower = brentq(lambda z: peanut_sdf(0.0, z, r1, r2, distance, beta),
                   -distance / 2 - r1 - 0.1, -distance / 2)
    upper = brentq(lambda z: peanut_sdf(0.0, z, r1, r2, distance, beta),
                   distance / 2, distance / 2 + r2 + 0.1)
    heights = np.linspace(lower, upper, spec.meridian_points)
    radii = np.zeros_like(heights)
    for index, height in enumerate(heights[1:-1], start=1):
        if peanut_sdf(0.0, height, r1, r2, distance, beta) >= 0:
            raise ValueError("peanut is disconnected at the symmetry axis")
        radii[index] = brentq(
            lambda radius: peanut_sdf(radius, height, r1, r2, distance, beta),
            0.0, r1 + r2 + distance,
        )
    if float(np.max(np.hypot(radii, heights))) >= 0.9:
        raise ValueError("peanut does not fit inside the registered obstacle radius")
    return np.column_stack((radii, heights))


def _make_peanut(spec: ObjectSpec):
    meridian = peanut_meridian(spec)
    spline = SplineInterpolation([Pnt(float(radius), 0.0, float(height))
                                  for radius, height in meridian])
    closing = Segment(Pnt(0.0, 0.0, float(meridian[-1, 1])),
                      Pnt(0.0, 0.0, float(meridian[0, 1])))
    return Face(Wire([spline, closing])).Revolve(Axis((0, 0, 0), (0, 0, 1)), 360)


def _scene_side_length(object_count: int) -> float:
    return 5.6 * (object_count / 3.0) ** (1.0 / 3.0)


def random_centers(
    rng: np.random.Generator,
    object_count: int,
    minimum_distance: float = 3.6,
    candidates_per_point: int = 8192,
) -> np.ndarray:
    """Best-candidate Poisson placement at approximately constant density."""
    side = _scene_side_length(object_count)
    centers = [rng.uniform(-0.5 * side, 0.5 * side, size=3)]
    while len(centers) < object_count:
        candidates = rng.uniform(-0.5 * side, 0.5 * side, size=(candidates_per_point, 3))
        existing = np.asarray(centers)
        nearest = np.linalg.norm(candidates[:, None, :] - existing[None, :, :], axis=-1).min(axis=1)
        best = int(np.argmax(nearest))
        if nearest[best] < minimum_distance:
            raise RuntimeError(
                f"failed to place {object_count} canonical spheres in side={side:.4f}; "
                f"best remaining separation={nearest[best]:.4f}"
            )
        centers.append(candidates[best])
    centers_array = np.asarray(centers, dtype=np.float64)
    # Remove an arbitrary global translation; only relative placement matters.
    return centers_array - centers_array.mean(axis=0, keepdims=True)


def balanced_object_count(index: int, count_min: int, count_max: int) -> int:
    counts = tuple(range(count_min, count_max + 1))
    return counts[index % len(counts)]


def sample_spec(
    index: int,
    split: str,
    base_seed: int,
    count_min: int,
    count_max: int,
    k: float = DEFAULT_K,
) -> SampleSpec:
    sample_seed = int(base_seed + 1_000_003 * index)
    rng = np.random.default_rng(sample_seed)
    object_count = balanced_object_count(index, count_min, count_max)
    objects = tuple(random_object(rng) for _ in range(object_count))
    centers = random_centers(rng, object_count)
    direction = rng.normal(size=3)
    direction /= np.linalg.norm(direction)
    return SampleSpec(
        index=index,
        split=split,
        sample_seed=sample_seed,
        k=float(k),
        objects=objects,
        centers=tuple(tuple(float(v) for v in row) for row in centers),
        incident_direction=tuple(float(v) for v in direction),
    )


def _harmonic_projection(values: np.ndarray, points: np.ndarray, lmax: int) -> np.ndarray:
    modes = spherical_modes(lmax)
    basis = spherical_harmonic_matrix(points, modes)
    weights = equal_area_weights(len(points), float(np.linalg.norm(points[0])))
    coefficients = weighted_pseudoinverse(basis, weights) @ values
    return basis @ coefficients


def _harmonic_projection_between_meshes(
    source_values: np.ndarray,
    source_points: np.ndarray,
    source_weights: np.ndarray,
    target_points: np.ndarray,
    lmax: int,
) -> np.ndarray:
    modes = spherical_modes(lmax)
    source_basis = spherical_harmonic_matrix(source_points, modes)
    target_basis = spherical_harmonic_matrix(target_points, modes)
    coefficients = weighted_pseudoinverse(source_basis, source_weights) @ source_values
    return target_basis @ coefficients


def _outer_nodes(mesh: Mesh) -> tuple[np.ndarray, np.ndarray]:
    region = mesh.Boundaries("canonical")
    node_numbers = sorted(
        {vertex.nr for element in region.Elements() for vertex in element.vertices}
    )
    coordinates = np.asarray(
        [tuple(mesh[NodeId(VERTEX, node)].point) for node in node_numbers],
        dtype=np.float64,
    )
    return np.asarray(node_numbers, dtype=np.int64), coordinates


def _prepare_local_system(mesh: Mesh, k: float):
    fes = H1(mesh, order=1, complex=True, dirichlet="canonical")
    trial, test = fes.TnT()
    helmholtz = BilinearForm(fes)
    helmholtz += grad(trial) * grad(test) * dx - k * k * trial * test * dx
    helmholtz.Assemble()
    rhs = LinearForm(fes)
    rhs.Assemble()
    inverse = helmholtz.mat.Inverse(freedofs=fes.FreeDofs())

    boundary_mass = BilinearForm(fes)
    boundary_mass += trial * test * ds("canonical")
    boundary_mass.Assemble()
    canonical_dofs = fes.GetDofs(mesh.Boundaries("canonical"))
    boundary_mass_inverse = boundary_mass.mat.Inverse(freedofs=canonical_dofs)
    ones = rhs.vec.CreateVector()
    ones.FV().NumPy()[:] = 1.0
    lumped = rhs.vec.CreateVector()
    lumped.data = boundary_mass.mat * ones
    lumped_boundary_mass = lumped.FV().NumPy().copy()
    node_numbers, node_coordinates = _outer_nodes(mesh)
    node_weights = np.maximum(np.abs(lumped_boundary_mass[node_numbers]), 1.0e-14)
    return (
        fes,
        helmholtz,
        rhs,
        inverse,
        boundary_mass_inverse,
        node_numbers,
        node_coordinates,
        node_weights,
    )


def _solve_local_boundary(
    mesh: Mesh,
    prepared,
    boundary_cf,
    interface_points: np.ndarray,
    flux_lmax: int,
) -> np.ndarray:
    (
        fes,
        helmholtz,
        rhs,
        inverse,
        boundary_mass_inverse,
        node_numbers,
        node_coordinates,
        node_weights,
    ) = prepared
    solution = GridFunction(fes)
    solution.Set(boundary_cf, definedon=mesh.Boundaries("canonical"))
    correction = rhs.vec.CreateVector()
    correction.data = rhs.vec - helmholtz.mat * solution.vec
    solution.vec.data += inverse * correction

    reaction = helmholtz.mat * solution.vec - rhs.vec
    flux = GridFunction(fes)
    flux.vec[:] = 0.0
    flux.vec.data += boundary_mass_inverse * reaction
    values = flux.vec.FV().NumPy()[node_numbers].astype(np.complex128, copy=True)
    return _harmonic_projection_between_meshes(
        values,
        node_coordinates,
        node_weights,
        interface_points,
        flux_lmax,
    )


def _solve_local_boundary_modes(
    mesh: Mesh,
    prepared,
    probe_modes: Sequence[tuple[int, int]],
    interface_points: np.ndarray,
    flux_lmax: int,
    batch_size: int = 16,
) -> np.ndarray:
    """Apply the FEM Schur operator to several boundary probes at once.

    The returned columns are the projected action

        Z = E M_Gamma^{-1} S_G P,

    where ``S_G`` is the local FEM Schur complement for the shell between the
    object (natural homogeneous-Neumann boundary) and the canonical sphere,
    ``P`` lifts spherical-harmonic probes to the outer FEM boundary, and ``E``
    projects the weak boundary reaction to the fixed interface points.  This
    is a reduced/projected Schur operator; the dense full boundary Schur matrix
    is never formed.

    NGSolve's MultiVector applies the already-factorized FEM inverse to a block
    of right-hand sides, avoiding 64 Python-level sparse back-solves while
    keeping peak memory bounded by ``batch_size``.
    """
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    (
        fes,
        helmholtz,
        rhs,
        inverse,
        boundary_mass_inverse,
        node_numbers,
        node_coordinates,
        node_weights,
    ) = prepared
    response = np.empty((len(interface_points), len(probe_modes)), dtype=np.complex128)

    for first in range(0, len(probe_modes), batch_size):
        batch_modes = probe_modes[first : first + batch_size]
        count = len(batch_modes)
        liftings = MultiVector(rhs.vec, count)
        for column, (ell, order) in enumerate(batch_modes):
            probe = SphericalHarmonicsCF(ell)
            probe.sh[ell, order] = 1.0
            lifting = GridFunction(fes)
            lifting.Set(probe, definedon=mesh.Boundaries("canonical"))
            liftings[column].data = lifting.vec

        stiffness_liftings = (helmholtz.mat * liftings).Evaluate()
        residuals = MultiVector(rhs.vec, count)
        for column in range(count):
            residuals[column].data = rhs.vec - stiffness_liftings[column]
        del liftings

        corrections = (inverse * residuals).Evaluate()
        del residuals
        stiffness_corrections = (helmholtz.mat * corrections).Evaluate()
        reactions = MultiVector(rhs.vec, count)
        for column in range(count):
            reactions[column].data = (
                stiffness_liftings[column] + stiffness_corrections[column] - rhs.vec
            )
        del corrections, stiffness_liftings, stiffness_corrections

        boundary_fluxes = (boundary_mass_inverse * reactions).Evaluate()
        del reactions
        for column in range(count):
            values = boundary_fluxes[column].FV().NumPy()[node_numbers].astype(
                np.complex128, copy=True
            )
            response[:, first + column] = _harmonic_projection_between_meshes(
                values,
                node_coordinates,
                node_weights,
                interface_points,
                flux_lmax,
            )
        del boundary_fluxes
    return response


def assemble_projected_schur_operator(
    spec: ObjectSpec,
    k: float,
    interface_points: np.ndarray,
    probe_modes: Sequence[tuple[int, int]],
    maxh: float,
    canonical_radius: float = DEFAULT_CANONICAL_RADIUS,
    flux_lmax: int = DEFAULT_FLUX_LMAX,
    batch_size: int = DEFAULT_SCHUR_BATCH_SIZE,
    graded_mesh_profile: str | None = None,
) -> np.ndarray:
    """Assemble the geometry-only, low-rank boundary Schur encoding ``Z``.

    No incident direction or incident-field response enters this operator.
    The obstacle geometry affects it through the assembled Helmholtz matrix on
    the canonical sphere-minus-obstacle FEM domain.
    """
    if graded_mesh_profile is None:
        raise ValueError("new FEM operators require an explicit graded_mesh_profile")
    from . import tmatrix_neumann_3d as graded_mesh

    profile = graded_mesh.GRADED_MESH_PROFILES[graded_mesh_profile]
    if not math.isclose(float(maxh), profile.background_h):
        raise ValueError("maxh must equal the selected graded profile background_h")
    mesh = graded_mesh.build_local_shell_mesh(
        spec,
        canonical_radius=canonical_radius,
        fem_order=2,
        profile=profile,
    )
    prepared = _prepare_local_system(mesh, k)
    return _solve_local_boundary_modes(
        mesh,
        prepared,
        probe_modes,
        interface_points,
        flux_lmax,
        batch_size=batch_size,
    )


def single_object_fingerprint(
    spec: ObjectSpec,
    k: float,
    interface_points: np.ndarray,
    probe_modes: Sequence[tuple[int, int]],
    maxh: float,
    canonical_radius: float = DEFAULT_CANONICAL_RADIUS,
    flux_lmax: int = DEFAULT_FLUX_LMAX,
    batch_size: int = DEFAULT_SCHUR_BATCH_SIZE,
    graded_mesh_profile: str | None = None,
) -> np.ndarray:
    """Backward-compatible name for projected Schur-operator assembly."""
    return assemble_projected_schur_operator(
        spec,
        k,
        interface_points,
        probe_modes,
        maxh,
        canonical_radius=canonical_radius,
        flux_lmax=flux_lmax,
        batch_size=batch_size,
        graded_mesh_profile=graded_mesh_profile,
    )


def single_object_flux_for_values(
    spec: ObjectSpec,
    k: float,
    interface_points: np.ndarray,
    boundary_values: np.ndarray,
    maxh: float,
    canonical_radius: float = DEFAULT_CANONICAL_RADIUS,
    boundary_lmax: int = DEFAULT_FLUX_LMAX,
    flux_lmax: int = DEFAULT_FLUX_LMAX,
    graded_mesh_profile: str | None = None,
) -> np.ndarray:
    """Reference DtN application for the rank-64 projection gate."""
    modes = spherical_modes(boundary_lmax)
    basis = spherical_harmonic_matrix(interface_points, modes)
    weights = equal_area_weights(len(interface_points), canonical_radius)
    coefficients = weighted_pseudoinverse(basis, weights) @ boundary_values
    boundary_cf = SphericalHarmonicsCF(boundary_lmax)
    for coefficient, (ell, order) in zip(coefficients, modes):
        boundary_cf.sh[ell, order] = complex(coefficient)

    if graded_mesh_profile is None:
        raise ValueError("new FEM flux solves require an explicit graded_mesh_profile")
    from . import tmatrix_neumann_3d as graded_mesh

    profile = graded_mesh.GRADED_MESH_PROFILES[graded_mesh_profile]
    if not math.isclose(float(maxh), profile.background_h):
        raise ValueError("maxh must equal the selected graded profile background_h")
    mesh = graded_mesh.build_local_shell_mesh(
        spec,
        canonical_radius=canonical_radius,
        fem_order=2,
        profile=profile,
    )
    return _solve_local_boundary(
        mesh,
        _prepare_local_system(mesh, k),
        boundary_cf,
        interface_points,
        flux_lmax,
    )


def _netgen_surface_arrays(spec: ObjectSpec, center: Sequence[float], maxh: float):
    shape = make_obstacle(spec, center)
    mesh = OCCGeometry(shape).GenerateMesh(maxh=maxh)
    vertices = np.asarray([tuple(point.p) for point in mesh.Points()], dtype=np.float64)
    triangles = np.asarray(
        [[vertex.nr - 1 for vertex in element.vertices] for element in mesh.Elements2D()],
        dtype=np.int64,
    )
    if triangles.ndim != 2 or triangles.shape[1] != 3:
        raise RuntimeError("BEM surface conversion requires a triangular Netgen surface mesh")

    # BEMPP expects outward normals. A radial-from-center test is invalid on
    # the inner torus wall (and can be unreliable in a peanut's neck), so use
    # the analytic implicit-surface gradient for the new OOD solids.
    a = vertices[triangles[:, 0]]
    b = vertices[triangles[:, 1]]
    c = vertices[triangles[:, 2]]
    normal = np.cross(b - a, c - a)
    midpoint = (a + b + c) / 3.0
    outward = midpoint - np.asarray(center, dtype=np.float64)
    if spec.shape in ("peanut", "torus"):
        rotation = np.asarray(spec.rotation, dtype=np.float64)
        local = outward @ rotation
        rho = np.linalg.norm(local[:, :2], axis=1).clip(min=1.0e-12)
        if spec.shape == "torus":
            major = float(spec.size[0])
            radial = (rho - major) / rho
            gradient = np.column_stack((radial * local[:, 0], radial * local[:, 1], local[:, 2]))
        else:
            r1, r2, distance = map(float, spec.size)
            shifted1 = local.copy()
            shifted2 = local.copy()
            shifted1[:, 2] += distance / 2.0
            shifted2[:, 2] -= distance / 2.0
            norm1 = np.linalg.norm(shifted1, axis=1).clip(min=1.0e-12)
            norm2 = np.linalg.norm(shifted2, axis=1).clip(min=1.0e-12)
            d1 = norm1 - r1
            d2 = norm2 - r2
            weights = np.exp(-30.0 * (d1 - np.minimum(d1, d2)))
            weights2 = np.exp(-30.0 * (d2 - np.minimum(d1, d2)))
            gradient = (weights[:, None] * shifted1 / norm1[:, None]
                        + weights2[:, None] * shifted2 / norm2[:, None])
        outward = gradient @ rotation.T
    reverse = np.einsum("ij,ij->i", normal, outward) < 0.0
    triangles[reverse, 1], triangles[reverse, 2] = (
        triangles[reverse, 2].copy(),
        triangles[reverse, 1].copy(),
    )
    return vertices, triangles


def make_bem_grid(objects: Sequence[ObjectSpec], centers: np.ndarray, maxh: float):
    bempp_api = _bempp_api()

    all_vertices = []
    all_triangles = []
    all_domains = []
    offset = 0
    for domain, (spec, center) in enumerate(zip(objects, centers), start=1):
        vertices, triangles = _netgen_surface_arrays(spec, center, maxh)
        all_vertices.append(vertices)
        all_triangles.append(triangles + offset)
        all_domains.append(np.full(len(triangles), domain, dtype=np.uint32))
        offset += len(vertices)
    return bempp_api.Grid(
        np.concatenate(all_vertices, axis=0).T,
        np.concatenate(all_triangles, axis=0).T,
        np.concatenate(all_domains),
    )


def solve_exterior_neumann_bem(
    objects: Sequence[ObjectSpec],
    centers: np.ndarray,
    k: float,
    incident_direction: np.ndarray,
    interface_points: np.ndarray,
    maxh: float,
    gmres_tolerance: float = DEFAULT_GMRES_TOL,
    assembler: str = "dense",
) -> tuple[np.ndarray, dict]:
    """Solve sound-hard scattering with a Burton--Miller combined potential.

    The scattered field is represented as ``D psi - i*k*S psi``.  With the
    BEMPP exterior trace convention this gives

    ``[W + i*k*(-1/2 I + K')] psi = d_n u_inc``.
    """
    bempp_api = _bempp_api(assembler)

    grid = make_bem_grid(objects, centers, maxh)
    space = bempp_api.function_space(grid, "P", 1)
    identity = bempp_api.operators.boundary.sparse.identity(space, space, space)
    kwargs = {"assembler": assembler}
    hypersingular = bempp_api.operators.boundary.helmholtz.hypersingular(
        space, space, space, k, **kwargs
    )
    adjoint = bempp_api.operators.boundary.helmholtz.adjoint_double_layer(
        space, space, space, k, **kwargs
    )
    normal_trace = -0.5 * identity + adjoint
    lhs = hypersingular + 1j * k * normal_trace
    direction = np.asarray(incident_direction, dtype=np.float64)

    @bempp_api.complex_callable
    def incident_normal(x, n, domain_index, result):
        phase = np.exp(1j * k * np.dot(direction, x))
        result[0] = 1j * k * np.dot(direction, n) * phase

    rhs = bempp_api.GridFunction(space, fun=incident_normal)
    density, info, residuals, iterations = bempp_api.linalg.gmres(
        lhs,
        rhs,
        tol=gmres_tolerance,
        return_residuals=True,
        return_iteration_count=True,
    )
    if info != 0:
        raise RuntimeError(f"BEM GMRES failed: info={info}, iterations={iterations}")

    global_points = np.concatenate(
        [interface_points + center[None, :] for center in centers], axis=0
    ).T
    double_layer = bempp_api.operators.potential.helmholtz.double_layer(
        space, global_points, k, **kwargs
    )
    single_layer = bempp_api.operators.potential.helmholtz.single_layer(
        space, global_points, k, **kwargs
    )
    scattered = (
        double_layer.evaluate(density) - 1j * k * single_layer.evaluate(density)
    ).reshape(-1)
    incident = np.exp(1j * k * np.dot(direction, global_points)).reshape(-1)
    total = (incident + scattered).reshape(len(objects), len(interface_points))
    final_residual = float(residuals[-1]) if residuals else float("nan")
    metadata = {
        "formulation": "burton-miller:D-i*k*S;W+i*k*(-1/2I+K')=d_n(u_inc)",
        "assembler": assembler,
        "gmres_tolerance": float(gmres_tolerance),
        "gmres_info": int(info),
        "gmres_iterations": int(iterations),
        "gmres_final_residual": final_residual,
        "bem_neumann_residual": final_residual,
        "neumann_boundary_residual": final_residual,
        "residual_definition": "relative GMRES residual of the weak Burton-Miller Neumann equation",
        "bempp_octree_jitclass": "disabled at import for dense assembly; octree unused",
        "solve_status": "converged" if int(info) == 0 else "failed",
        "mesh_maxh": float(maxh),
        "surface_vertices": int(grid.number_of_vertices),
        "surface_elements": int(grid.number_of_elements),
    }
    return total, metadata


def incident_on_interfaces(
    centers: np.ndarray,
    interface_points: np.ndarray,
    direction: np.ndarray,
    k: float,
) -> np.ndarray:
    global_points = centers[:, None, :] + interface_points[None, :, :]
    return np.exp(1j * k * np.einsum("d,ond->on", direction, global_points))


def generate_sample_arrays(
    spec: SampleSpec,
    local_maxh: float,
    bem_maxh: float,
    assembler: str = "dense",
    canonical_radius: float = DEFAULT_CANONICAL_RADIUS,
    interface_nodes: int = DEFAULT_INTERFACE_NODES,
    probe_lmax: int = DEFAULT_PROBE_LMAX,
    flux_lmax: int = DEFAULT_FLUX_LMAX,
    schur_batch_size: int = DEFAULT_SCHUR_BATCH_SIZE,
    gmres_tolerance: float = DEFAULT_GMRES_TOL,
    graded_mesh_profiles_by_family: dict[str, str] | None = None,
) -> tuple[dict[str, np.ndarray], dict]:
    started = time.perf_counter()
    interface_points, interface_faces = canonical_surface(interface_nodes, canonical_radius)
    probe_modes = spherical_modes(probe_lmax)
    probe_values = spherical_harmonic_matrix(interface_points, probe_modes)
    weights = equal_area_weights(interface_nodes, canonical_radius)
    phi_pinv = weighted_pseudoinverse(probe_values, weights)
    centers = np.asarray(spec.centers, dtype=np.float64)
    direction = np.asarray(spec.incident_direction, dtype=np.float64)

    if graded_mesh_profiles_by_family is None:
        raise ValueError("graded_mesh_profiles_by_family is required for every new dataset sample")
    if not set(SHAPE_NAMES).issubset(graded_mesh_profiles_by_family):
        raise ValueError("graded mesh profile map must cover every supported shape family")
    from . import tmatrix_neumann_3d as graded_mesh

    for family, name in graded_mesh_profiles_by_family.items():
        if name not in graded_mesh.GRADED_MESH_PROFILES:
            raise ValueError(f"unknown graded profile {name!r} for {family}")
    object_profiles = [graded_mesh_profiles_by_family[obj.shape] for obj in spec.objects]
    fingerprints = np.asarray([
        assemble_projected_schur_operator(
            obj,
            spec.k,
            interface_points,
            probe_modes,
            graded_mesh.GRADED_MESH_PROFILES[mesh_profile].background_h,
            canonical_radius=canonical_radius,
            flux_lmax=flux_lmax,
            batch_size=schur_batch_size,
            graded_mesh_profile=mesh_profile,
        )
        for obj, mesh_profile in zip(spec.objects, object_profiles)
    ], dtype=np.complex128)
    total, bem_metadata = solve_exterior_neumann_bem(
        spec.objects,
        centers,
        spec.k,
        direction,
        interface_points,
        bem_maxh,
        gmres_tolerance=gmres_tolerance,
        assembler=assembler,
    )
    incident = incident_on_interfaces(centers, interface_points, direction, spec.k)
    q64 = project_flux(fingerprints, total, phi_pinv)
    for name, value in {
        "Z": fingerprints,
        "u_total_gt": total,
        "u_inc": incident,
        "q64_gt": q64,
    }.items():
        if value.dtype != np.complex128:
            raise TypeError(f"{name} must be checked in complex128 before storage")
        if not np.isfinite(value).all():
            raise FloatingPointError(f"{name} contains non-finite values")
    if int(bem_metadata["gmres_info"]) != 0:
        raise RuntimeError(f"BEM GMRES failed: {bem_metadata}")
    if float(bem_metadata["gmres_final_residual"]) > gmres_tolerance:
        raise RuntimeError(
            "BEM GMRES residual exceeds the configured tolerance: "
            f"{bem_metadata['gmres_final_residual']:.3e} > {gmres_tolerance:.3e}"
        )
    rotations = np.asarray([obj.rotation for obj in spec.objects], dtype=np.float64)
    shape_parameters = np.full((len(spec.objects), 3), np.nan, dtype=np.float64)
    axis_permutations = np.full((len(spec.objects), 3), -1, dtype=np.int8)
    for row, obj in enumerate(spec.objects):
        shape_parameters[row, : len(obj.size)] = obj.size
        axis_permutations[row, : len(obj.axis_permutation)] = obj.axis_permutation

    arrays = {
        "Z": fingerprints.astype(np.complex64),
        "u_total_gt": total.astype(np.complex64),
        "u_inc": incident.astype(np.complex64),
        "q64_gt": q64.astype(np.complex64),
        "probe_values": probe_values.astype(np.complex64),
        "probe_modes": np.asarray(probe_modes, dtype=np.int8),
        "phi_pinv": phi_pinv.astype(np.complex64),
        "geometry_operator": np.asarray("projected_local_dtn_schur"),
        "spherical_harmonic_convention": np.asarray("ngsolve_no_condon_shortley_positive_m"),
        "interface_points": interface_points.astype(np.float32),
        "interface_faces": interface_faces.astype(np.int32),
        "centers": centers.astype(np.float32),
        "rotations": rotations.astype(np.float32),
        "shape_names": np.asarray([obj.shape for obj in spec.objects]),
        "shape_parameters": shape_parameters.astype(np.float32),
        "axis_permutations": axis_permutations,
        "k": np.asarray(spec.k, dtype=np.float32),
        "incident_direction": direction.astype(np.float32),
        "object_count": np.asarray(len(spec.objects), dtype=np.int32),
        "schema_version": np.asarray(SCHEMA_VERSION),
    }
    metadata = {
        "file": f"sample_{spec.index:06d}.npz",
        "index": spec.index,
        "split": spec.split,
        "sample_seed": spec.sample_seed,
        "k": spec.k,
        "incident_direction": list(spec.incident_direction),
        "centers": [list(row) for row in spec.centers],
        "objects": [asdict(obj) for obj in spec.objects],
        "mesh_strategy": "graded-local-FEM",
        "local_mesh_profiles_by_object": object_profiles,
        "graded_mesh_profiles_by_family": graded_mesh_profiles_by_family,
        "legacy_uniform_local_maxh_argument_ignored": float(local_maxh),
        "geometry_operator": {
            "definition": "E M_Gamma^{-1} S_G P",
            "incident_independent": True,
            "probe_count": len(probe_modes),
            "probe_modes_lmax": int(probe_lmax),
            "assembly": "NGSolve FEM with batched MultiVector Schur right-hand sides",
            "rhs_batch_size": int(schur_batch_size),
        },
        "bem_maxh": float(bem_maxh),
        "bem": bem_metadata,
        "wall_seconds": time.perf_counter() - started,
    }
    return arrays, metadata


def _write_npz_atomic(path: Path, arrays: dict[str, np.ndarray]) -> None:
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(path)


def _worker_generate(payload):
    spec, output, options = payload
    output_path = Path(output) / f"sample_{spec.index:06d}.npz"
    arrays, metadata = generate_sample_arrays(spec, **options)
    _write_npz_atomic(output_path, arrays)
    metadata["sha256"] = _sha256(output_path)
    return metadata


def _json_atomic(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _distribution_name(distribution: str) -> str:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return "unavailable"


def _git_provenance() -> dict:
    repository = Path(__file__).resolve().parents[3]
    try:
        commit = subprocess.check_output(
            ["git", "-C", str(repository), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        dirty = bool(
            subprocess.check_output(
                ["git", "-C", str(repository), "status", "--porcelain"],
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
        )
        return {"commit": commit, "dirty": dirty}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}


def _validated_gate(args: argparse.Namespace) -> dict | None:
    if args.unsafe_skip_gate:
        if args.num_samples > 2:
            raise ValueError("--unsafe-skip-gate is restricted to at most two-sample software smokes")
        return None
    if not args.gate_report:
        raise ValueError(
            "formal generation requires --gate-report; use --unsafe-skip-gate only for software smoke tests"
        )
    path = Path(args.gate_report).resolve()
    gate = json.loads(path.read_text(encoding="utf-8"))
    if not gate.get("passed"):
        raise ValueError(f"numerical gate did not pass: {path}")
    if gate.get("mesh_strategy") != "graded-local-FEM":
        raise ValueError("legacy uniform-mesh gate cannot authorize new data; a graded release gate is required")
    if not gate.get("bem_label_audit_passed"):
        raise ValueError("independent BEM label audit has not passed")
    calibration_path = Path(args.graded_calibration_report).resolve()
    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    if calibration.get("schema") != "graded-local-calibration-summary-v1" or calibration.get("overall_status") != "passed":
        raise ValueError(f"graded local calibration has not passed: {calibration_path}")
    calibration_hash = _sha256(calibration_path)
    if gate.get("graded_calibration_sha256") != calibration_hash:
        raise ValueError("release gate does not match the supplied graded calibration evidence")
    profiles = gate.get("graded_mesh_profiles_by_family")
    if not isinstance(profiles, dict) or not set(SHAPE_NAMES).issubset(profiles):
        raise ValueError("release gate must specify a graded profile for every geometry family")
    from . import tmatrix_neumann_3d as graded_mesh
    if any(name not in graded_mesh.GRADED_MESH_PROFILES for name in profiles.values()):
        raise ValueError("release gate contains an unknown graded mesh profile")
    selected_bem = float(gate["selected_bem_maxh"])
    if not math.isclose(args.bem_maxh, selected_bem):
        raise ValueError(
            f"bem_maxh must equal the BEM-audited setting h={selected_bem}"
        )
    expected = {
        "k": args.k,
        "canonical_radius": args.canonical_radius,
        "interface_nodes": args.interface_nodes,
        "probe_lmax": args.probe_lmax,
        "flux_lmax": args.flux_lmax,
    }
    for key, value in expected.items():
        if not math.isclose(float(gate[key]), float(value)):
            raise ValueError(f"gate/report mismatch for {key}: {gate[key]} != {value}")
    return {
        "path": str(path),
        "sha256": _sha256(path),
        "mesh_strategy": "graded-local-FEM",
        "graded_calibration_path": str(calibration_path),
        "graded_calibration_sha256": calibration_hash,
        "graded_mesh_profiles_by_family": profiles,
        "selected_bem_maxh": selected_bem,
        "thresholds": gate.get("thresholds"),
    }


def generate_dataset(args: argparse.Namespace) -> None:
    gate = _validated_gate(args)
    graded_profiles = (
        gate["graded_mesh_profiles_by_family"]
        if gate is not None else E1_CANDIDATE_GRADED_PROFILES
    )
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "manifest.json"
    existing = {}
    if manifest_path.exists():
        loaded = json.loads(manifest_path.read_text(encoding="utf-8"))
        if loaded.get("schema_version") != SCHEMA_VERSION:
            raise ValueError("existing manifest has a different schema_version")
        expected_resume = {
            "split": args.split,
            "seed": args.seed,
            "num_samples": args.num_samples,
            "k": args.k,
            "canonical_radius": args.canonical_radius,
            "interface_nodes": args.interface_nodes,
            "probe_lmax": args.probe_lmax,
            "flux_lmax": args.flux_lmax,
            "schur_batch_size": args.schur_batch_size,
            "count_range": [args.count_min, args.count_max],
            "legacy_uniform_local_maxh_argument_ignored": args.local_maxh,
            "graded_mesh_profiles_by_family": graded_profiles,
            "bem_maxh": args.bem_maxh,
            "gmres_tolerance": args.gmres_tolerance,
            "assembler": args.assembler,
        }
        mismatches = {
            key: (loaded.get(key), value)
            for key, value in expected_resume.items()
            if loaded.get(key) != value
        }
        if loaded.get("numerical_gate") != gate:
            mismatches["numerical_gate"] = (loaded.get("numerical_gate"), gate)
        if bool(loaded.get("formal_evidence")) != (gate is not None):
            mismatches["formal_evidence"] = (
                loaded.get("formal_evidence"),
                gate is not None,
            )
        if mismatches:
            raise ValueError(f"refusing incompatible resume: {mismatches}")
        existing = {item["index"]: item for item in loaded.get("samples", [])}

    specs = [
        sample_spec(
            index,
            args.split,
            args.seed,
            args.count_min,
            args.count_max,
            k=args.k,
        )
        for index in range(args.num_samples)
    ]
    pending = [
        spec
        for spec in specs
        if not (output / f"sample_{spec.index:06d}.npz").exists()
    ]
    options = {
        "legacy_uniform_local_maxh_argument_ignored": args.local_maxh,
        "graded_mesh_profiles_by_family": graded_profiles,
        "bem_maxh": args.bem_maxh,
        "assembler": args.assembler,
        "git": _git_provenance(),
        "canonical_radius": args.canonical_radius,
        "interface_nodes": args.interface_nodes,
        "probe_lmax": args.probe_lmax,
        "flux_lmax": args.flux_lmax,
        "schur_batch_size": args.schur_batch_size,
        "gmres_tolerance": args.gmres_tolerance,
    }

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "status": "in_progress",
        "split": args.split,
        "seed": args.seed,
        "num_samples": args.num_samples,
        "equation": "3-D scalar exterior Helmholtz",
        "boundary_condition": "homogeneous Neumann",
        "incident": "unit-amplitude zero-phase plane wave",
        "k": args.k,
        "canonical_radius": args.canonical_radius,
        "interface_nodes": args.interface_nodes,
        "probe_lmax": args.probe_lmax,
        "probe_count": (args.probe_lmax + 1) ** 2,
        "spherical_harmonic_convention": "ngsolve_no_condon_shortley_positive_m",
        "flux_lmax": args.flux_lmax,
        "schur_batch_size": args.schur_batch_size,
        "geometry_operator": {
            "definition": "E M_Gamma^{-1} S_G P",
            "incident_independent": True,
            "probe_count": (args.probe_lmax + 1) ** 2,
            "probe_modes_lmax": args.probe_lmax,
            "assembly": "NGSolve FEM with batched MultiVector Schur right-hand sides",
            "rhs_batch_size": args.schur_batch_size,
        },
        "count_range": [args.count_min, args.count_max],
        "legacy_uniform_local_maxh_argument_ignored": args.local_maxh,
        "mesh_strategy": "graded-local-FEM",
        "graded_mesh_profiles_by_family": graded_profiles,
        "graded_calibration": None if gate is None else {
            "path": gate["graded_calibration_path"],
            "sha256": gate["graded_calibration_sha256"],
        },
        "bem_maxh": args.bem_maxh,
        "gmres_tolerance": args.gmres_tolerance,
        "assembler": args.assembler,
        "numerical_gate": gate,
        "formal_evidence": gate is not None,
        "complex128_checks_before_storage": True,
        "runtime": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "numpy": _distribution_name("numpy"),
            "scipy": _distribution_name("scipy"),
            "netgen_mesher": _distribution_name("netgen-mesher"),
            "ngsolve": _distribution_name("ngsolve"),
            "bempp_cl": _distribution_name("bempp-cl"),
        },
        "samples": [existing[index] for index in sorted(existing)],
    }
    _json_atomic(manifest_path, manifest)

    payloads = [(spec, str(output), options) for spec in pending]
    if args.workers == 1:
        results: Iterable[dict] = map(_worker_generate, payloads)
    else:
        executor = ProcessPoolExecutor(
            max_workers=args.workers,
            max_tasks_per_child=args.tasks_per_worker if args.tasks_per_worker > 0 else None,
        )
        results = executor.map(_worker_generate, payloads, chunksize=1)

    try:
        for metadata in results:
            existing[int(metadata["index"])] = metadata
            manifest["samples"] = [existing[index] for index in sorted(existing)]
            _json_atomic(manifest_path, manifest)
            print(
                f"generated {metadata['file']} objects={len(metadata['objects'])} "
                f"bem_iter={metadata['bem']['gmres_iterations']} "
                f"seconds={metadata['wall_seconds']:.2f}",
                flush=True,
            )
    finally:
        if args.workers != 1:
            executor.shutdown(wait=True, cancel_futures=False)

    manifest["status"] = "complete" if len(existing) == args.num_samples else "in_progress"
    manifest["samples"] = [existing[index] for index in sorted(existing)]
    _json_atomic(manifest_path, manifest)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--split", choices=("train", "val", "test", "count_ood_n8"), required=True)
    parser.add_argument("--num-samples", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--count-min", type=int, default=2)
    parser.add_argument("--count-max", type=int, default=4)
    parser.add_argument("--k", type=float, default=DEFAULT_K)
    parser.add_argument("--canonical-radius", type=float, default=DEFAULT_CANONICAL_RADIUS)
    parser.add_argument("--interface-nodes", type=int, default=DEFAULT_INTERFACE_NODES)
    parser.add_argument("--probe-lmax", type=int, default=DEFAULT_PROBE_LMAX)
    parser.add_argument("--flux-lmax", type=int, default=DEFAULT_FLUX_LMAX)
    parser.add_argument("--schur-batch-size", type=int, default=DEFAULT_SCHUR_BATCH_SIZE)
    parser.add_argument(
        "--local-maxh", type=float, required=True,
        help="legacy compatibility field; FEM mesh sizes come only from the graded profile map",
    )
    parser.add_argument("--bem-maxh", type=float, required=True)
    parser.add_argument("--gmres-tolerance", type=float, default=DEFAULT_GMRES_TOL)
    parser.add_argument("--assembler", choices=("dense", "fmm"), default="dense")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument(
        "--tasks-per-worker",
        type=int,
        default=512,
        help="recycle worker after this many samples; use 0 to keep workers for the full split",
    )
    parser.add_argument("--gate-report", default="")
    parser.add_argument(
        "--graded-calibration-report",
        default="outputs/v3_graded_mesh_calibration_v2/summary.json",
    )
    parser.add_argument(
        "--unsafe-skip-gate",
        action="store_true",
        help="software smoke tests only; datasets created this way are not formal evidence",
    )
    args = parser.parse_args()
    if args.split == "count_ood_n8":
        args.count_min = args.count_max = 8
    if args.interface_nodes != 128 or args.probe_lmax != 7 or args.k != 2.0:
        raise ValueError("formal v1 is fixed to interface_nodes=128, probe_lmax=7, and k=2.0")
    if args.schur_batch_size < 1:
        raise ValueError("--schur-batch-size must be positive")
    if args.count_min < 1 or args.count_max < args.count_min:
        raise ValueError("invalid object-count range")
    return args


if __name__ == "__main__":
    generate_dataset(parse_args())
