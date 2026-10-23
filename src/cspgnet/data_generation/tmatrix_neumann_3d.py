"""Offline local DtN operators and online acoustic multiple scattering.

This is an experimental, fail-closed alternative to the dense global BEM
generator.  It does not label a dataset as formal evidence until the separate
numerical gate has compared it with independent BEM references.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Sequence

import numpy as np
from ngsolve import Conj, Preconditioner
from scipy.special import roots_legendre, spherical_jn, spherical_yn

from .block_rhs import (
    BatchedReducedSolver,
    NGSolveBDDCGMRESBlockSolver,
    NGSolveSuperLUBlockSolver,
)
from . import formal_neumann_3d as formal


@dataclass(frozen=True)
class AngularQuadrature:
    directions: np.ndarray
    weights: np.ndarray


@dataclass(frozen=True)
class GradedMeshProfile:
    name: str
    scatterer_h: float
    canonical_h: float
    background_h: float
    grading: float = 0.30

    def as_dict(self) -> dict[str, float | str]:
        return {
            "name": self.name,
            "scatterer_h": self.scatterer_h,
            "canonical_h": self.canonical_h,
            "background_h": self.background_h,
            "grading": self.grading,
        }


GRADED_MESH_PROFILES = {
    "G0": GradedMeshProfile("G0", 0.14, 0.22, 0.30),
    "G1": GradedMeshProfile("G1", 0.10, 0.18, 0.25),
    "G2": GradedMeshProfile("G2", 0.07, 0.14, 0.21),
    "G3": GradedMeshProfile("G3", 0.05, 0.10, 0.18),
}


def build_local_shell_mesh(
    spec: formal.ObjectSpec,
    *,
    canonical_radius: float = formal.DEFAULT_CANONICAL_RADIUS,
    fem_order: int = 2,
    profile: GradedMeshProfile | None = None,
    uniform_maxh: float | None = None,
):
    """Build the FEM shell, using explicit local grading when a profile is given.

    The uniform branch is retained only for historical diagnostics. New
    calibration/data code must supply a registered graded profile.
    """
    obstacle = formal.make_obstacle(spec)
    return build_local_shell_mesh_from_solid(
        obstacle,
        canonical_radius=canonical_radius,
        fem_order=fem_order,
        profile=profile,
        uniform_maxh=uniform_maxh,
    )


def build_local_shell_mesh_from_solid(
    obstacle,
    *,
    canonical_radius: float = formal.DEFAULT_CANONICAL_RADIUS,
    fem_order: int = 2,
    profile: GradedMeshProfile | None = None,
    uniform_maxh: float | None = None,
):
    """Build a local shell around an audited OCC solid using an explicit mesh profile.

    This is the generic geometry path for the V6 structural CAD library. The
    caller owns and records the solid parameters/topology audit; this routine
    only builds the graded FEM shell and never relabels a CAD failure as a
    numerical geometry failure.
    """
    if (profile is None) == (uniform_maxh is None):
        raise ValueError("provide exactly one of a graded profile or uniform_maxh")
    outer = formal.Sphere((0.0, 0.0, 0.0), canonical_radius)
    for face in outer.faces:
        face.name = "canonical"
        if profile is not None:
            face.maxh = profile.canonical_h
    for face in obstacle.faces:
        face.name = "scatterer"
        if profile is not None:
            face.maxh = profile.scatterer_h
    geometry = formal.OCCGeometry(outer - obstacle)
    if profile is None:
        ngmesh = geometry.GenerateMesh(maxh=float(uniform_maxh))
    else:
        if min(profile.scatterer_h, profile.canonical_h, profile.background_h) <= 0:
            raise ValueError("graded mesh sizes must be positive")
        if not (0.0 < profile.grading < 1.0):
            raise ValueError("Netgen grading must be between zero and one")
        ngmesh = geometry.GenerateMesh(
            maxh=profile.background_h,
            grading=profile.grading,
        )
    mesh = formal.Mesh(ngmesh)
    mesh.Curve(fem_order)
    return mesh


def _mesh_diagnostics(mesh) -> dict:
    """Record realized tetrahedral quality and per-boundary edge sizes."""
    ngmesh = mesh.ngmesh
    points = np.asarray([tuple(point.p) for point in ngmesh.Points()], dtype=np.float64)
    tetrahedra = np.asarray([
        [vertex.nr - 1 for vertex in element.vertices]
        for element in ngmesh.Elements3D()
    ], dtype=np.int64)
    if tetrahedra.ndim != 2 or tetrahedra.shape[1] != 4:
        raise RuntimeError("expected a tetrahedral FEM mesh")
    xyz = points[tetrahedra]
    edge_pairs = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))
    edge_lengths = np.stack([
        np.linalg.norm(xyz[:, first] - xyz[:, second], axis=1)
        for first, second in edge_pairs
    ], axis=1)
    signed_six_volume = np.linalg.det(xyz[:, 1:] - xyz[:, :1])
    volume = np.abs(signed_six_volume) / 6.0
    quality = 12.0 * np.maximum(3.0 * volume, 1.0e-30) ** (2.0 / 3.0) / np.maximum(
        np.sum(edge_lengths**2, axis=1), 1.0e-30
    )
    boundary_names = tuple(mesh.GetBoundaries())
    boundary_edges: dict[str, list[float]] = {}
    for element in ngmesh.Elements2D():
        name = boundary_names[element.index - 1]
        face_points = points[[vertex.nr - 1 for vertex in element.vertices]]
        lengths = [
            float(np.linalg.norm(face_points[i] - face_points[j]))
            for i in range(len(face_points))
            for j in range(i + 1, len(face_points))
        ]
        boundary_edges.setdefault(name, []).extend(lengths)
    boundary_profile = {
        name: {
            "edge_count": len(lengths),
            "median": float(np.median(lengths)),
            "p90": float(np.percentile(lengths, 90)),
            "max": float(np.max(lengths)),
        }
        for name, lengths in boundary_edges.items()
    }
    return {
        "mesh_vertices": int(mesh.nv),
        "tetrahedra": int(len(tetrahedra)),
        "element_longest_edge_median": float(np.median(edge_lengths.max(axis=1))),
        "element_longest_edge_p90": float(np.percentile(edge_lengths.max(axis=1), 90)),
        "element_quality_min": float(np.min(quality)),
        "element_quality_p01": float(np.percentile(quality, 1)),
        "elements_nonpositive_volume": int(np.count_nonzero(volume <= 1.0e-14)),
        "boundary_size_profile": boundary_profile,
    }


def angular_quadrature(n_polar: int = 24, n_azimuth: int = 48) -> AngularQuadrature:
    """Product Gauss--Legendre/trapezoid rule on the unit sphere."""
    if n_polar < 2 or n_azimuth < 4:
        raise ValueError("angular quadrature is too small")
    z, polar_weights = roots_legendre(n_polar)
    azimuth = 2.0 * np.pi * np.arange(n_azimuth) / n_azimuth
    zz, phi = np.meshgrid(z, azimuth, indexing="ij")
    radial = np.sqrt(np.maximum(0.0, 1.0 - zz * zz))
    directions = np.stack(
        (radial * np.cos(phi), radial * np.sin(phi), zz), axis=-1
    ).reshape(-1, 3)
    weights = np.repeat(polar_weights, n_azimuth) * (2.0 * np.pi / n_azimuth)
    return AngularQuadrature(directions, weights)


def radial_values(k: float, radius: float, lmax: int, outgoing: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """Diagonal regular/outgoing radial values and radial derivatives."""
    values = []
    derivatives = []
    for ell, _ in formal.spherical_modes(lmax):
        argument = k * radius
        value = spherical_jn(ell, argument)
        derivative = k * spherical_jn(ell, argument, derivative=True)
        if outgoing:
            value += 1j * spherical_yn(ell, argument)
            derivative += 1j * k * spherical_yn(ell, argument, derivative=True)
        values.append(value)
        derivatives.append(derivative)
    return np.asarray(values, dtype=np.complex128), np.asarray(derivatives, dtype=np.complex128)


def sphere_modal_dtn(
    radius: float,
    k: float = formal.DEFAULT_K,
    canonical_radius: float = formal.DEFAULT_CANONICAL_RADIUS,
    input_lmax: int = 9,
    output_lmax: int = 10,
) -> np.ndarray:
    """Exact DtN matrix for a concentric sound-hard sphere."""
    if not (0.0 < radius < canonical_radius):
        raise ValueError("the obstacle must fit strictly inside the canonical sphere")
    if output_lmax < input_lmax:
        raise ValueError("output_lmax must include every input mode")
    output = np.zeros(((output_lmax + 1) ** 2, (input_lmax + 1) ** 2), dtype=np.complex128)
    for column, (ell, _) in enumerate(formal.spherical_modes(input_lmax)):
        ja = spherical_jn(ell, k * radius, derivative=True)
        ha = ja + 1j * spherical_yn(ell, k * radius, derivative=True)
        coefficient = -ja / ha
        j_r = spherical_jn(ell, k * canonical_radius)
        h_r = j_r + 1j * spherical_yn(ell, k * canonical_radius)
        j_prime = k * spherical_jn(ell, k * canonical_radius, derivative=True)
        h_prime = k * (
            spherical_jn(ell, k * canonical_radius, derivative=True)
            + 1j * spherical_yn(ell, k * canonical_radius, derivative=True)
        )
        output[column, column] = (j_prime + coefficient * h_prime) / (j_r + coefficient * h_r)
    return output


def weak_modal_dtn(
    spec: formal.ObjectSpec | None,
    maxh: float,
    *,
    k: float = formal.DEFAULT_K,
    canonical_radius: float = formal.DEFAULT_CANONICAL_RADIUS,
    input_lmax: int = 9,
    output_lmax: int = 10,
    fem_order: int = 2,
    mesh_profile: GradedMeshProfile | None = None,
    obstacle_solid=None,
    solver_backend: str = "ngsolve",
    rhs_batch_size: int | None = None,
) -> tuple[np.ndarray, dict]:
    """Compute a consistent boundary-flux projection of the local FEM Schur map.

    The reaction vector is paired directly with conjugated harmonic traces.
    This avoids reconstructing a flux from vertex-only samples, which loses
    the extra boundary degrees of freedom of a quadratic FEM space.
    """
    if output_lmax < input_lmax:
        raise ValueError("output_lmax must include every input mode")
    if fem_order < 1:
        raise ValueError("fem_order must be positive")
    started = time.perf_counter()
    if mesh_profile is not None and not math.isclose(maxh, mesh_profile.background_h):
        raise ValueError("maxh must equal the graded profile's background_h")
    if (spec is None) == (obstacle_solid is None):
        raise ValueError("provide exactly one of ObjectSpec or obstacle_solid")
    mesh_builder = build_local_shell_mesh if spec is not None else build_local_shell_mesh_from_solid
    geometry = spec if spec is not None else obstacle_solid
    mesh = mesh_builder(
        geometry,
        canonical_radius=canonical_radius,
        fem_order=fem_order,
        profile=mesh_profile,
        uniform_maxh=None if mesh_profile is not None else maxh,
    )
    mesh_meta = _mesh_diagnostics(mesh)

    fes = formal.H1(mesh, order=fem_order, complex=True, dirichlet="canonical")
    trial, test = fes.TnT()
    helmholtz = formal.BilinearForm(fes)
    helmholtz += formal.grad(trial) * formal.grad(test) * formal.dx
    helmholtz += -(k * k) * trial * test * formal.dx
    bddc_preconditioner = None
    bddc_construction_seconds = 0.0
    if solver_backend == "ngsolve-bddc-gmres":
        preconditioner_started = time.perf_counter()
        bddc_preconditioner = Preconditioner(helmholtz, type="bddc")
        bddc_construction_seconds = time.perf_counter() - preconditioner_started
    helmholtz.Assemble()
    rhs = formal.LinearForm(fes)
    rhs.Assemble()
    free_dofs = np.asarray(fes.FreeDofs(), dtype=bool)
    factor_started = time.perf_counter()
    block_solver = None
    if solver_backend == "ngsolve":
        inverse = helmholtz.mat.Inverse(freedofs=fes.FreeDofs())
    elif solver_backend == "superlu-batched":
        # Legacy audit candidate: extracting and separately factoring the full
        # FEM matrix has not qualified for production use.
        block_solver = BatchedReducedSolver.from_ngsolve_matrix(helmholtz.mat, free_dofs)
        inverse = None
        # The extracted CSC operator and SuperLU factors are sufficient for
        # all modal and additional trace solves on this mesh. Release NGSolve's
        # duplicate assembled operator before allocating the RHS blocks.
        del helmholtz, rhs
    elif solver_backend == "ngsolve-superlu-batched":
        # Reuse NGSolve's optimized reduced-matrix extraction/factorization;
        # apply the exposed SuperLU factor to dense matrix-RHS blocks.
        block_solver = NGSolveSuperLUBlockSolver(helmholtz.mat, fes.FreeDofs())
        inverse = None
    elif solver_backend == "ngsolve-bddc-gmres":
        bddc_preconditioner.Update()
        block_solver = NGSolveBDDCGMRESBlockSolver(
            helmholtz.mat, bddc_preconditioner.mat, fes.FreeDofs(),
            tolerance=1.0e-10, max_steps=400,
        )
        block_solver.factorization_seconds = (
            bddc_construction_seconds + time.perf_counter() - factor_started
        )
        inverse = None
    else:
        raise ValueError(
            "solver_backend must be 'ngsolve', 'superlu-batched' (audit only), "
            "'ngsolve-superlu-batched' (benchmark candidate), or "
            "'ngsolve-bddc-gmres' (experimental iterative candidate)"
        )
    factorization_seconds = time.perf_counter() - factor_started
    boundary_mass = formal.BilinearForm(fes)
    boundary_mass += trial * test * formal.ds("canonical")
    boundary_mass.Assemble()

    boundary_dofs = np.flatnonzero(np.asarray(fes.GetDofs(mesh.Boundaries("canonical"))))
    output_modes = formal.spherical_modes(output_lmax)
    input_modes = formal.spherical_modes(input_lmax)
    traces = np.empty((fes.ndof, len(output_modes)), dtype=np.complex128)
    conjugate_traces = np.empty((len(boundary_dofs), len(output_modes)), dtype=np.complex128)
    trace = formal.GridFunction(fes)
    for column, (ell, order) in enumerate(output_modes):
        harmonic = formal.SphericalHarmonicsCF(ell)
        harmonic.sh[ell, order] = 1.0
        trace.Set(harmonic, definedon=mesh.Boundaries("canonical"))
        traces[:, column] = trace.vec.FV().NumPy()
        trace.Set(Conj(harmonic), definedon=mesh.Boundaries("canonical"))
        conjugate_traces[:, column] = trace.vec.FV().NumPy()[boundary_dofs]

    mass_times_traces = np.empty((len(boundary_dofs), len(output_modes)), dtype=np.complex128)
    input_vector = rhs.vec.CreateVector()
    mass_vector = rhs.vec.CreateVector()
    for column in range(len(output_modes)):
        input_vector.FV().NumPy()[:] = traces[:, column]
        mass_vector.data = boundary_mass.mat * input_vector
        mass_times_traces[:, column] = mass_vector.FV().NumPy()[boundary_dofs]
    gram = conjugate_traces.T @ mass_times_traces

    measurements = np.empty((len(output_modes), len(input_modes)), dtype=np.complex128)
    solution = formal.GridFunction(fes)
    if block_solver is None:
        correction = rhs.vec.CreateVector()
        reaction = rhs.vec.CreateVector()
    residuals = []
    block_solver_metadata = None
    if block_solver is not None:
        block_result = block_solver.solve_lifts(
            traces[:, :len(input_modes)], batch_size=rhs_batch_size
        )
        residuals = block_result.relative_free_residuals.tolist()
        measurements[:, :] = (
            conjugate_traces.T @ block_result.reactions[boundary_dofs, :]
        )
        block_solver_metadata = {
            "factorization_backend": (
                {
                    "superlu-batched": "scipy-csc-extracted",
                    "ngsolve-superlu-batched": "ngsolve-reduced-superlu",
                    "ngsolve-bddc-gmres": "ngsolve-bddc-gmres",
                }[solver_backend]
            ),
            "rhs_count": block_result.rhs_count,
            "rhs_batch_size": block_result.batch_size,
            "block_solve_seconds": block_result.solve_seconds,
        }
        if solver_backend == "superlu-batched":
            block_solver_metadata.update({
                "matrix_nnz": block_solver.matrix_nnz,
                "reduced_matrix_nnz": block_solver.reduced_nnz,
            })
    else:
        for column in range(len(input_modes)):
            solution.vec.FV().NumPy()[:] = traces[:, column]
            correction.data = rhs.vec - helmholtz.mat * solution.vec
            solution.vec.data += inverse * correction
            reaction.data = helmholtz.mat * solution.vec - rhs.vec
            reaction_values = reaction.FV().NumPy()
            residuals.append(float(
                np.linalg.norm(reaction_values[free_dofs])
                / max(np.linalg.norm(reaction_values), 1.0e-30)
            ))
            measurements[:, column] = conjugate_traces.T @ reaction_values[boundary_dofs]
    modal_dtn = np.linalg.solve(gram, measurements)
    if not np.isfinite(modal_dtn).all():
        raise FloatingPointError("local DtN matrix is not finite")
    metadata = {
        "shape": spec.shape if spec is not None else "custom-occ-solid",
        "maxh": float(maxh),
        "mesh_profile": None if mesh_profile is None else mesh_profile.as_dict(),
        "fem_order": int(fem_order),
        "curved_mesh": True,
        **mesh_meta,
        "fes_dofs": int(fes.ndof),
        "boundary_dofs": int(len(boundary_dofs)),
        "gram_condition": float(np.linalg.cond(gram)),
        "max_free_dof_relative_residual": max(residuals),
        "solver_backend": solver_backend,
        "factorization_seconds": float(block_solver.factorization_seconds
                                       if block_solver is not None else factorization_seconds),
        "block_solver": block_solver_metadata,
        "seconds": float(time.perf_counter() - started),
    }
    return modal_dtn, metadata


def weak_modal_dtn_for_solid(
    obstacle_solid,
    maxh: float,
    *,
    k: float = formal.DEFAULT_K,
    canonical_radius: float = formal.DEFAULT_CANONICAL_RADIUS,
    input_lmax: int = 9,
    output_lmax: int = 10,
    fem_order: int = 2,
    mesh_profile: GradedMeshProfile | None = None,
) -> tuple[np.ndarray, dict]:
    """Generic-OCC-solid wrapper for ``weak_modal_dtn`` using graded FEM."""
    return weak_modal_dtn(
        None, maxh, k=k, canonical_radius=canonical_radius,
        input_lmax=input_lmax, output_lmax=output_lmax,
        fem_order=fem_order, mesh_profile=mesh_profile,
        obstacle_solid=obstacle_solid,
    )


def weak_flux_for_harmonic_coefficients(
    spec: formal.ObjectSpec,
    maxh: float,
    coefficients: np.ndarray,
    *,
    k: float = formal.DEFAULT_K,
    canonical_radius: float = formal.DEFAULT_CANONICAL_RADIUS,
    output_lmax: int = 10,
    fem_order: int = 2,
    mesh_profile: GradedMeshProfile | None = None,
) -> tuple[np.ndarray, dict]:
    """Independently solve P2 FEM for prescribed harmonic boundary traces.

    Input coefficients are applied *on the actual FEM boundary DOFs*. The
    reaction is tested there against the output harmonics and divided by the
    consistent boundary Gram matrix. No 128-point trace interpolation or
    precomputed modal DtN multiplication occurs in this reference path.
    """
    coefficients = np.asarray(coefficients, dtype=np.complex128)
    if coefficients.ndim == 1:
        coefficients = coefficients[:, None]
    input_lmax = math.isqrt(coefficients.shape[0]) - 1
    if (input_lmax + 1) ** 2 != coefficients.shape[0]:
        raise ValueError("coefficient count must be a complete harmonic order")
    started = time.perf_counter()
    if mesh_profile is not None and not math.isclose(maxh, mesh_profile.background_h):
        raise ValueError("maxh must equal the graded profile's background_h")
    mesh = build_local_shell_mesh(
        spec,
        canonical_radius=canonical_radius,
        fem_order=fem_order,
        profile=mesh_profile,
        uniform_maxh=None if mesh_profile is not None else maxh,
    )
    mesh_meta = _mesh_diagnostics(mesh)
    fes = formal.H1(mesh, order=fem_order, complex=True, dirichlet="canonical")
    trial, test = fes.TnT()
    helmholtz = formal.BilinearForm(fes)
    helmholtz += formal.grad(trial) * formal.grad(test) * formal.dx
    helmholtz += -(k * k) * trial * test * formal.dx
    helmholtz.Assemble()
    rhs = formal.LinearForm(fes)
    rhs.Assemble()
    inverse = helmholtz.mat.Inverse(freedofs=fes.FreeDofs())
    mass = formal.BilinearForm(fes)
    mass += trial * test * formal.ds("canonical")
    mass.Assemble()
    boundary_dofs = np.flatnonzero(np.asarray(fes.GetDofs(mesh.Boundaries("canonical"))))
    output_modes = formal.spherical_modes(output_lmax)
    trace_vector = rhs.vec.CreateVector()
    mass_vector = rhs.vec.CreateVector()
    trace = formal.GridFunction(fes)
    conjugate_traces = np.empty((len(boundary_dofs), len(output_modes)), dtype=np.complex128)
    mass_traces = np.empty_like(conjugate_traces)
    for column, (ell, order) in enumerate(output_modes):
        harmonic = formal.SphericalHarmonicsCF(ell)
        harmonic.sh[ell, order] = 1.0
        trace.Set(harmonic, definedon=mesh.Boundaries("canonical"))
        trace_vector.FV().NumPy()[:] = trace.vec.FV().NumPy()
        mass_vector.data = mass.mat * trace_vector
        mass_traces[:, column] = mass_vector.FV().NumPy()[boundary_dofs]
        trace.Set(Conj(harmonic), definedon=mesh.Boundaries("canonical"))
        conjugate_traces[:, column] = trace.vec.FV().NumPy()[boundary_dofs]
    gram = conjugate_traces.T @ mass_traces
    measurements = np.empty((len(output_modes), coefficients.shape[1]), dtype=np.complex128)
    solution = formal.GridFunction(fes)
    correction = rhs.vec.CreateVector()
    reaction = rhs.vec.CreateVector()
    free = np.asarray(fes.FreeDofs(), dtype=bool)
    residuals = []
    for column in range(coefficients.shape[1]):
        boundary_cf = formal.SphericalHarmonicsCF(input_lmax)
        for value, (ell, order) in zip(coefficients[:, column], formal.spherical_modes(input_lmax)):
            boundary_cf.sh[ell, order] = complex(value)
        solution.Set(boundary_cf, definedon=mesh.Boundaries("canonical"))
        correction.data = rhs.vec - helmholtz.mat * solution.vec
        solution.vec.data += inverse * correction
        reaction.data = helmholtz.mat * solution.vec - rhs.vec
        values = reaction.FV().NumPy()
        residuals.append(float(np.linalg.norm(values[free]) / max(np.linalg.norm(values), 1.0e-15)))
        measurements[:, column] = conjugate_traces.T @ values[boundary_dofs]
    flux = np.linalg.solve(gram, measurements)
    if not np.isfinite(flux).all():
        raise FloatingPointError("direct FEM flux is non-finite")
    return flux, {
        "max_free_dof_relative_residual": max(residuals),
        "gram_condition": float(np.linalg.cond(gram)),
        "mesh_profile": None if mesh_profile is None else mesh_profile.as_dict(),
        **mesh_meta,
        "fes_dofs": int(fes.ndof),
        "boundary_dofs": int(len(boundary_dofs)),
        "seconds": float(time.perf_counter() - started),
    }


def nodal_fingerprint(
    modal_dtn: np.ndarray,
    interface_points: np.ndarray,
    *,
    input_lmax: int = 7,
) -> np.ndarray:
    """Return the fixed 128-by-64 network input from a modal DtN matrix."""
    output_lmax = int(round(math.sqrt(modal_dtn.shape[0]) - 1))
    if (output_lmax + 1) ** 2 != modal_dtn.shape[0]:
        raise ValueError("modal DtN output dimension is not a full harmonic order")
    count = (input_lmax + 1) ** 2
    if modal_dtn.shape[1] < count:
        raise ValueError("modal DtN lacks the requested input modes")
    phi = formal.spherical_harmonic_matrix(interface_points, formal.spherical_modes(output_lmax))
    return phi @ modal_dtn[:, :count]


def transition_matrix(
    modal_dtn: np.ndarray,
    *,
    k: float = formal.DEFAULT_K,
    canonical_radius: float = formal.DEFAULT_CANONICAL_RADIUS,
    lmax: int = 9,
) -> tuple[np.ndarray, dict]:
    """Convert local DtN into the outgoing/regular acoustic T matrix."""
    count = (lmax + 1) ** 2
    if modal_dtn.shape[0] < count or modal_dtn.shape[1] < count:
        raise ValueError("modal DtN is too small for the requested T matrix")
    local = modal_dtn[:count, :count]
    regular, regular_prime = radial_values(k, canonical_radius, lmax)
    outgoing, outgoing_prime = radial_values(k, canonical_radius, lmax, outgoing=True)
    lhs = np.diag(outgoing_prime) - local * outgoing[None, :]
    rhs = local * regular[None, :] - np.diag(regular_prime)
    result = np.linalg.solve(lhs, rhs)
    residual = np.linalg.norm(lhs @ result - rhs) / max(np.linalg.norm(rhs), 1.0e-15)
    if not np.isfinite(result).all():
        raise FloatingPointError("T matrix contains non-finite entries")
    return result, {"conversion_condition": float(np.linalg.cond(lhs)), "conversion_residual": float(residual)}


def rotation_matrix(
    rotation: np.ndarray,
    lmax: int,
    quadrature: AngularQuadrature,
) -> np.ndarray:
    """Harmonic representation of f(n) -> f(R^T n)."""
    rotation = np.asarray(rotation, dtype=np.float64)
    if (
        rotation.shape != (3, 3)
        or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1.0e-8)
        or not np.isclose(np.linalg.det(rotation), 1.0, atol=1.0e-8)
    ):
        raise ValueError("rotation must be a proper SO(3) matrix")
    modes = formal.spherical_modes(lmax)
    basis = formal.spherical_harmonic_matrix(quadrature.directions, modes)
    rotated_basis = formal.spherical_harmonic_matrix(quadrature.directions @ rotation, modes)
    transform = basis.conj().T @ (quadrature.weights[:, None] * rotated_basis)
    return transform


def spherical_wave_values(
    vectors: np.ndarray,
    k: float,
    lmax: int,
    *,
    outgoing: bool,
) -> np.ndarray:
    """Evaluate regular or outgoing spherical waves at Cartesian vectors."""
    vectors = np.asarray(vectors, dtype=np.float64)
    radii = np.linalg.norm(vectors, axis=1)
    if np.any(radii <= 0.0):
        raise ValueError("a spherical wave was evaluated at its expansion center")
    phi = formal.spherical_harmonic_matrix(vectors, formal.spherical_modes(lmax))
    result = np.empty_like(phi)
    for column, (ell, _) in enumerate(formal.spherical_modes(lmax)):
        radial = spherical_jn(ell, k * radii)
        if outgoing:
            radial = radial + 1j * spherical_yn(ell, k * radii)
        result[:, column] = radial * phi[:, column]
    return result


def translation_matrix(
    receiver_minus_source: np.ndarray,
    k: float,
    canonical_radius: float,
    lmax: int,
    quadrature: AngularQuadrature,
    basis: np.ndarray | None = None,
) -> np.ndarray:
    """Project another center's outgoing waves onto regular incoming waves."""
    shift = np.asarray(receiver_minus_source, dtype=np.float64)
    if np.linalg.norm(shift) <= 2.0 * canonical_radius:
        raise ValueError("canonical spheres must not touch or overlap")
    modes = formal.spherical_modes(lmax)
    if basis is None:
        basis = formal.spherical_harmonic_matrix(quadrature.directions, modes)
    evaluation = shift[None, :] + canonical_radius * quadrature.directions
    outgoing = spherical_wave_values(evaluation, k, lmax, outgoing=True)
    regular, _ = radial_values(k, canonical_radius, lmax)
    if np.min(np.abs(regular)) <= 1.0e-12:
        raise ArithmeticError("regular radial basis is nearly zero at the projection radius")
    return (basis.conj().T @ (quadrature.weights[:, None] * outgoing)) / regular[:, None]


def solve_scene(
    transitions: Sequence[np.ndarray],
    centers: np.ndarray,
    incident_direction: np.ndarray,
    interface_points: np.ndarray,
    *,
    k: float = formal.DEFAULT_K,
    canonical_radius: float = formal.DEFAULT_CANONICAL_RADIUS,
    lmax: int = 9,
    quadrature: AngularQuadrature | None = None,
) -> tuple[np.ndarray, dict]:
    """Solve the Foldy--Lax/T-matrix coupling and sample total fields."""
    quadrature = angular_quadrature() if quadrature is None else quadrature
    centers = np.asarray(centers, dtype=np.float64)
    direction = np.array(incident_direction, dtype=np.float64, copy=True)
    direction /= np.linalg.norm(direction)
    count = len(transitions)
    modes = formal.spherical_modes(lmax)
    width = len(modes)
    if centers.shape != (count, 3):
        raise ValueError("centers and transitions have different object counts")
    for transition in transitions:
        if transition.shape != (width, width):
            raise ValueError("one T matrix has an incompatible harmonic order")
    basis = formal.spherical_harmonic_matrix(quadrature.directions, modes)
    coupling = np.zeros((count * width, count * width), dtype=np.complex128)
    blocks = np.zeros_like(coupling)
    for receiver in range(count):
        sl = slice(receiver * width, (receiver + 1) * width)
        blocks[sl, sl] = transitions[receiver]
        for source in range(count):
            if receiver == source:
                continue
            sj = slice(source * width, (source + 1) * width)
            coupling[sl, sj] = translation_matrix(
                centers[receiver] - centers[source],
                k,
                canonical_radius,
                lmax,
                quadrature,
                basis,
            )
    direction_basis = formal.spherical_harmonic_matrix(direction[None, :], modes)[0]
    incident_coefficients = np.asarray(
        [
            np.exp(1j * k * (center @ direction))
            * 4.0
            * np.pi
            * np.asarray([(1j**ell) * np.conj(direction_basis[column]) for column, (ell, _) in enumerate(modes)])
            for center in centers
        ],
        dtype=np.complex128,
    ).reshape(-1)
    lhs = np.eye(count * width, dtype=np.complex128) - blocks @ coupling
    rhs = blocks @ incident_coefficients
    outgoing_coefficients = np.linalg.solve(lhs, rhs)
    residual = np.linalg.norm(lhs @ outgoing_coefficients - rhs) / max(np.linalg.norm(rhs), 1.0e-15)
    if residual > 1.0e-10:
        raise ArithmeticError(f"multiple-scattering linear residual {residual:.3e} is too large")
    total = np.empty((count, len(interface_points)), dtype=np.complex128)
    for receiver, center in enumerate(centers):
        points = center[None, :] + interface_points
        values = np.exp(1j * k * (points @ direction))
        for source in range(count):
            sl = slice(source * width, (source + 1) * width)
            values += spherical_wave_values(points - centers[source], k, lmax, outgoing=True) @ outgoing_coefficients[sl]
        total[receiver] = values
    if not np.isfinite(total).all():
        raise FloatingPointError("total field contains non-finite entries")
    return total, {"linear_residual": float(residual), "linear_condition": float(np.linalg.cond(lhs)), "modes": width}
