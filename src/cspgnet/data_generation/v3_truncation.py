"""High-angular-resolution GT traces for independent direct-FEM truncation.

The 128 network nodes never enter the reference trace projection. This module
retains a 100-mode Foldy--Lax label solve, evaluates its field on a dense
Gauss--Legendre sphere, and fits l<=13 before applying each truncated trace
separately to the true curved-P2 FEM boundary.
"""

from __future__ import annotations

import numpy as np

from . import formal_neumann_3d as formal
from . import tmatrix_neumann_3d as tmatrix


def outgoing_coefficients(transitions: list[np.ndarray], centers: np.ndarray,
                          incident_direction: np.ndarray,
                          quadrature: tmatrix.AngularQuadrature | None = None):
    quadrature = tmatrix.angular_quadrature() if quadrature is None else quadrature
    centers = np.asarray(centers, dtype=np.float64)
    direction = np.asarray(incident_direction, dtype=np.float64)
    direction /= np.linalg.norm(direction)
    count = len(transitions)
    width = 100
    basis = formal.spherical_harmonic_matrix(quadrature.directions, formal.spherical_modes(9))
    blocks = np.zeros((count * width, count * width), dtype=np.complex128)
    coupling = np.zeros_like(blocks)
    for receiver in range(count):
        row = slice(receiver * width, (receiver + 1) * width)
        blocks[row, row] = transitions[receiver]
        for source in range(count):
            if source == receiver:
                continue
            col = slice(source * width, (source + 1) * width)
            coupling[row, col] = tmatrix.translation_matrix(
                centers[receiver] - centers[source], 2.0, 1.5, 9, quadrature, basis,
            )
    direction_basis = formal.spherical_harmonic_matrix(
        direction[None, :], formal.spherical_modes(9),
    )[0]
    incident = np.asarray([
        np.exp(2j * (center @ direction)) * 4.0 * np.pi
        * np.asarray([(1j ** ell) * np.conj(direction_basis[column])
                      for column, (ell, _) in enumerate(formal.spherical_modes(9))])
        for center in centers
    ]).reshape(-1)
    lhs = np.eye(count * width) - blocks @ coupling
    rhs = blocks @ incident
    coefficients = np.linalg.solve(lhs, rhs).reshape(count, width)
    residual = float(np.linalg.norm(lhs @ coefficients.ravel() - rhs) / max(np.linalg.norm(rhs), 1.0e-15))
    if residual > 1.0e-10:
        raise ArithmeticError("100-mode Foldy--Lax coefficient residual is too large")
    return coefficients, {"relative_residual": residual, "dimension": count * width}


def high_order_total_trace_coefficients(outgoing: np.ndarray, centers: np.ndarray,
                                         direction: np.ndarray, receiver: int,
                                         *, trace_lmax: int = 13,
                                         projection_quadrature=None):
    if projection_quadrature is None:
        projection_quadrature = tmatrix.angular_quadrature(40, 80)
    normals = projection_quadrature.directions
    sphere = centers[receiver][None, :] + 1.5 * normals
    field = np.exp(2j * (sphere @ direction))
    for source in range(len(outgoing)):
        field += tmatrix.spherical_wave_values(
            sphere - centers[source], 2.0, 9, outgoing=True,
        ) @ outgoing[source]
    basis = formal.spherical_harmonic_matrix(normals, formal.spherical_modes(trace_lmax))
    weights = projection_quadrature.weights
    gram = basis.conj().T @ (weights[:, None] * basis)
    coefficients = np.linalg.solve(gram, basis.conj().T @ (weights * field))
    fit = float(np.linalg.norm(np.sqrt(weights) * (basis @ coefficients - field))
                / np.linalg.norm(np.sqrt(weights) * field))
    return coefficients, {"quadrature_nodes": len(normals), "relative_trace_fit": fit,
                          "projection_condition": float(np.linalg.cond(gram))}


def direct_fem_truncation_for_object(spec: formal.ObjectSpec, maxh: float,
                                     trace_coefficients: np.ndarray, *,
                                     graded_mesh_profile: str):
    trace_coefficients = np.asarray(trace_coefficients, dtype=np.complex128)
    if trace_coefficients.shape != (196,):
        raise ValueError("independent high-resolution trace must contain l<=13")
    traces = np.zeros((196, 5), dtype=np.complex128)
    traces[:, 0] = trace_coefficients
    for column, lmax in enumerate((3, 5, 7, 9), start=1):
        traces[:(lmax + 1) ** 2, column] = trace_coefficients[:(lmax + 1) ** 2]
    flux, metadata = tmatrix.weak_flux_for_harmonic_coefficients(
        spec, maxh, traces, output_lmax=10,
        mesh_profile=tmatrix.GRADED_MESH_PROFILES[graded_mesh_profile],
    )
    errors = {}
    for column, lmax in enumerate((3, 5, 7, 9), start=1):
        errors[str((lmax + 1) ** 2)] = float(np.linalg.norm(flux[:, column] - flux[:, 0])
                                          / max(np.linalg.norm(flux[:, 0]), 1.0e-15))
    return errors, metadata
