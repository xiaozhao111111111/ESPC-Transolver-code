"""Classical 64-mode multiple scattering using only a sample's stored Z.

The label generator's 100-mode local matrices are intentionally inaccessible
here. In particular, ``P @ Z`` is a *modal DtN*, not an outgoing T matrix.
"""

from __future__ import annotations

import time

import numpy as np
from scipy.linalg import lu_factor, lu_solve
from scipy.sparse.linalg import gmres

from . import formal_neumann_3d as formal
from . import tmatrix_neumann_3d as tmatrix


def modal_dtn_from_stored_z(sample: dict[str, np.ndarray]) -> tuple[np.ndarray, dict]:
    z_stored = np.asarray(sample["Z"])
    if z_stored.dtype != np.complex64 or z_stored.ndim != 3 or z_stored.shape[1:] != (128, 64):
        raise ValueError("E8 requires saved complex64 Z with shape [N,128,64]")
    points = np.asarray(sample["interface_points"], dtype=np.float64)
    radius = float(np.mean(np.linalg.norm(points, axis=1)))
    if not np.allclose(np.linalg.norm(points, axis=1), radius, atol=1.0e-5):
        raise ValueError("interface points are not on one canonical sphere")
    phi = formal.spherical_harmonic_matrix(points, formal.spherical_modes(7))
    weights = formal.equal_area_weights(len(points), radius)
    pinv = formal.weighted_pseudoinverse(phi, weights)
    b64 = np.einsum("mn,onp->omp", pinv, z_stored.astype(np.complex128), optimize=True)
    return b64, {
        "projection_condition": float(np.linalg.cond(phi.conj().T @ (weights[:, None] * phi))),
        "radius": radius,
        "input_dtype": str(z_stored.dtype),
    }


def solve_from_stored_z(
    sample: dict[str, np.ndarray],
    *,
    quadrature: tmatrix.AngularQuadrature | None = None,
    check_gmres: bool = True,
) -> tuple[np.ndarray, dict]:
    """Return total field and diagnostics from ``B64 -> T_out -> Foldy--Lax``.

    ``b`` denotes *outgoing* coefficients. Thus even N=1 has a nonzero
    self-scattered field, unlike a cross-Green total-trace equation without a
    self term. No label or hidden 100-mode matrix is read by this routine.
    """
    started = time.perf_counter()
    b64, projection = modal_dtn_from_stored_z(sample)
    centers = np.asarray(sample["centers"], dtype=np.float64)
    points = np.asarray(sample["interface_points"], dtype=np.float64)
    k = float(sample["k"])
    radius = projection["radius"]
    direction = np.asarray(sample["incident_direction"], dtype=np.float64)
    direction /= np.linalg.norm(direction)
    count = len(b64)
    if centers.shape != (count, 3):
        raise ValueError("center count differs from Z object count")
    quadrature = tmatrix.angular_quadrature() if quadrature is None else quadrature
    transitions = []
    conversions = []
    for local in b64:
        transition, conversion = tmatrix.transition_matrix(
            local, k=k, canonical_radius=radius, lmax=7,
        )
        if conversion["conversion_residual"] > 1.0e-10:
            raise ArithmeticError("modal DtN to outgoing T conversion did not converge")
        transitions.append(transition)
        conversions.append(conversion)

    width = 64
    basis = formal.spherical_harmonic_matrix(quadrature.directions, formal.spherical_modes(7))
    coupling = np.zeros((count * width, count * width), dtype=np.complex128)
    blocks = np.zeros_like(coupling)
    for receiver in range(count):
        row = slice(receiver * width, (receiver + 1) * width)
        blocks[row, row] = transitions[receiver]
        for source in range(count):
            if source != receiver:
                col = slice(source * width, (source + 1) * width)
                coupling[row, col] = tmatrix.translation_matrix(
                    centers[receiver] - centers[source], k, radius, 7, quadrature, basis,
                )
    modes = formal.spherical_modes(7)
    direction_basis = formal.spherical_harmonic_matrix(direction[None, :], modes)[0]
    incident_coefficients = np.asarray([
        np.exp(1j * k * (center @ direction)) * 4.0 * np.pi
        * np.asarray([(1j ** ell) * np.conj(direction_basis[j])
                      for j, (ell, _) in enumerate(modes)])
        for center in centers
    ]).reshape(-1)
    lhs = np.eye(count * width, dtype=np.complex128) - blocks @ coupling
    rhs = blocks @ incident_coefficients
    lu, piv = lu_factor(lhs, check_finite=True)
    outgoing = lu_solve((lu, piv), rhs)
    residual = float(np.linalg.norm(lhs @ outgoing - rhs) / max(np.linalg.norm(rhs), 1.0e-15))
    if residual > 1.0e-10:
        raise ArithmeticError(f"64N dense LU residual {residual:.3e} exceeds 1e-10")
    gmres_meta = None
    if check_gmres:
        iterations = [0]
        def on_iteration(_):
            iterations[0] += 1
        iterative, info = gmres(
            lhs, rhs, rtol=1.0e-10, atol=0.0, restart=min(100, count * width),
            maxiter=200, callback=on_iteration, callback_type="pr_norm",
        )
        discrepancy = float(np.linalg.norm(iterative - outgoing) / max(np.linalg.norm(outgoing), 1.0e-15))
        gmres_meta = {"info": int(info), "iterations": iterations[0], "lu_relative_difference": discrepancy}
        if info != 0 or discrepancy > 1.0e-8:
            raise ArithmeticError(f"64N GMRES/LU cross-check failed: {gmres_meta}")

    total = np.empty((count, len(points)), dtype=np.complex128)
    for receiver, center in enumerate(centers):
        locations = center[None, :] + points
        field = np.exp(1j * k * (locations @ direction))
        for source in range(count):
            col = slice(source * width, (source + 1) * width)
            field += tmatrix.spherical_wave_values(
                locations - centers[source], k, 7, outgoing=True,
            ) @ outgoing[col]
        total[receiver] = field
    return total, {
        "schema": "classical-z64-foldy-lax-v1",
        "objects": count,
        "dimension": count * width,
        "projection": projection,
        "conversion": conversions,
        "lu_relative_residual": residual,
        "matrix_condition": float(np.linalg.cond(lhs)),
        "gmres": gmres_meta,
        "seconds": float(time.perf_counter() - started),
    }
