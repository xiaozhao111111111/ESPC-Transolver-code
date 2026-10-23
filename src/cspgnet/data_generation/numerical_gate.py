"""Historical uniform-mesh gate; cannot authorize new dataset generation.

The command exits with status 0 only when a mesh size satisfies every hard
criterion in the approved experiment plan.  A failing report is a normal and
useful outcome: formal large-scale generation must not start after such a
failure.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from scipy.special import eval_legendre, spherical_jn, spherical_yn

from .formal_neumann_3d import (
    DEFAULT_CANONICAL_RADIUS,
    DEFAULT_FLUX_LMAX,
    DEFAULT_GMRES_TOL,
    DEFAULT_INTERFACE_NODES,
    DEFAULT_K,
    DEFAULT_PROBE_LMAX,
    Mesh,
    OCCGeometry,
    ObjectSpec,
    SphericalHarmonicsCF,
    Sphere,
    canonical_surface,
    equal_area_weights,
    project_flux,
    random_object,
    single_object_fingerprint,
    single_object_flux_for_values,
    solve_exterior_neumann_bem,
    spherical_harmonic_matrix,
    spherical_modes,
    weighted_pseudoinverse,
)


THRESHOLDS = {
    "analytic_sphere": 0.01,
    "neumann_boundary_residual": 1.0e-8,
    "bem_refinement": 0.02,
    "z_frobenius": 0.03,
    "z_mode_p95": 0.05,
    "schur_linearity": 1.0e-8,
    "projection_q": 0.05,
    "probe_basis_convention": 1.0e-10,
}


def relative_l2(left: np.ndarray, right: np.ndarray) -> float:
    denominator = max(float(np.linalg.norm(right)), 1.0e-15)
    return float(np.linalg.norm(left - right) / denominator)


def analytic_neumann_sphere_total(
    points: np.ndarray,
    radius: float,
    k: float,
    direction: np.ndarray,
    center: np.ndarray | None = None,
    lmax: int | None = None,
) -> np.ndarray:
    """Mie-series total field for a sound-hard sphere and a plane wave."""
    center = np.zeros(3) if center is None else np.asarray(center, dtype=np.float64)
    relative = points - center[None, :]
    radial = np.linalg.norm(relative, axis=1)
    unit = relative / radial[:, None]
    direction = np.asarray(direction, dtype=np.float64)
    direction /= np.linalg.norm(direction)
    cosine = np.clip(unit @ direction, -1.0, 1.0)
    if lmax is None:
        lmax = max(20, int(math.ceil(k * radial.max() + 12)))

    scattered = np.zeros(len(points), dtype=np.complex128)
    ka = k * radius
    kr = k * radial
    for ell in range(lmax + 1):
        j_derivative = spherical_jn(ell, ka, derivative=True)
        h_derivative = (
            spherical_jn(ell, ka, derivative=True)
            + 1j * spherical_yn(ell, ka, derivative=True)
        )
        coefficient = -j_derivative / h_derivative
        hankel = spherical_jn(ell, kr) + 1j * spherical_yn(ell, kr)
        scattered += (
            (1j**ell)
            * (2 * ell + 1)
            * coefficient
            * hankel
            * eval_legendre(ell, cosine)
        )
    incident = np.exp(1j * k * (points @ direction))
    return incident + scattered


def _identity_rotation():
    return tuple(tuple(float(value) for value in row) for row in np.eye(3))


def _gate_objects(cases_per_shape: int, seed: int) -> list[ObjectSpec]:
    rng = np.random.default_rng(seed)
    by_shape: dict[str, list[ObjectSpec]] = {name: [] for name in ("sphere", "ellipsoid", "box")}
    while any(len(values) < cases_per_shape for values in by_shape.values()):
        candidate = random_object(rng)
        values = by_shape[candidate.shape]
        if len(values) < cases_per_shape:
            values.append(candidate)
    return [item for shape in ("sphere", "ellipsoid", "box") for item in by_shape[shape]]


def probe_basis_convention_error(
    interface_nodes: int,
    canonical_radius: float,
    probe_lmax: int,
) -> float:
    """Compare every SciPy probe with NGSolve's coefficient-function basis."""
    probe_points = np.asarray(
        [
            [0.31, -0.47, 0.72],
            [-0.63, 0.22, 0.49],
            [0.54, 0.66, -0.31],
            [-0.28, -0.71, -0.41],
            [0.77, -0.19, 0.33],
        ],
        dtype=np.float64,
    )
    probe_points *= 0.7 / np.linalg.norm(probe_points, axis=1, keepdims=True)
    modes = spherical_modes(probe_lmax)
    expected = spherical_harmonic_matrix(probe_points, modes)
    mesh = Mesh(
        OCCGeometry(Sphere((0.0, 0.0, 0.0), canonical_radius)).GenerateMesh(maxh=0.8)
    )
    maximum_error = 0.0
    for column, (ell, order) in enumerate(modes):
        probe = SphericalHarmonicsCF(ell)
        probe.sh[ell, order] = 1.0
        for row, point in enumerate(probe_points):
            actual = complex(probe(mesh(*(float(value) for value in point))))
            maximum_error = max(maximum_error, abs(actual - expected[row, column]))
    return float(maximum_error)


def run_gate(args: argparse.Namespace) -> dict:
    raise RuntimeError(
        "This legacy release gate uses uniform FEM meshes and is retained only "
        "as historical evidence. It cannot authorize new data. Use the v3 "
        "graded local calibration followed by an independent BEM label audit."
    )

    points, _ = canonical_surface(args.interface_nodes, args.canonical_radius)
    probe_modes = spherical_modes(args.probe_lmax)
    phi = spherical_harmonic_matrix(points, probe_modes)
    phi_pinv = weighted_pseudoinverse(
        phi,
        equal_area_weights(len(points), args.canonical_radius),
    )
    basis_convention_error = probe_basis_convention_error(
        args.interface_nodes, args.canonical_radius, args.probe_lmax
    )
    sphere = ObjectSpec("sphere", (0.4,), _identity_rotation())
    direction = np.asarray([1.0, 0.0, 0.0])
    analytic = analytic_neumann_sphere_total(points, 0.4, args.k, direction)
    cases = _gate_objects(args.cases_per_shape, args.seed)

    z_by_h = {}
    for maxh in args.mesh_sizes:
        print(f"gate mesh h={maxh:.5f}: {len(cases)} local fingerprints", flush=True)
        z_by_h[maxh] = [
            single_object_fingerprint(
                obj,
                args.k,
                points,
                probe_modes,
                maxh,
                canonical_radius=args.canonical_radius,
                flux_lmax=args.flux_lmax,
            )
            for obj in cases
        ]

    # Local DtN convergence is a hard release gate.  Test it before assembling
    # the much more expensive global BEM systems; if every adjacent mesh pair
    # fails, no mesh can be selected and the formal data pipeline must stop.
    z_comparisons = []
    z_eligible_coarse = set()
    for coarse, fine in zip(args.mesh_sizes[:-1], args.mesh_sizes[1:]):
        z_numerator = math.sqrt(
            sum(float(np.linalg.norm(a - b) ** 2) for a, b in zip(z_by_h[coarse], z_by_h[fine]))
        )
        z_denominator = math.sqrt(
            sum(float(np.linalg.norm(b) ** 2) for b in z_by_h[fine])
        )
        z_frobenius = z_numerator / max(z_denominator, 1.0e-15)
        mode_errors = []
        for coarse_z, fine_z in zip(z_by_h[coarse], z_by_h[fine]):
            for mode in range(fine_z.shape[1]):
                mode_errors.append(relative_l2(coarse_z[:, mode], fine_z[:, mode]))
        mode_p95 = float(np.quantile(mode_errors, 0.95))
        z_passed = bool(
            z_frobenius <= THRESHOLDS["z_frobenius"]
            and mode_p95 <= THRESHOLDS["z_mode_p95"]
        )
        if z_passed:
            z_eligible_coarse.add(coarse)
        z_comparisons.append(
            {
                "coarse_h": coarse,
                "fine_h": fine,
                "z_frobenius": z_frobenius,
                "z_mode_p95": mode_p95,
                "z_gate_passed": z_passed,
            }
        )

    common = {
        "k": args.k,
        "canonical_radius": args.canonical_radius,
        "interface_nodes": args.interface_nodes,
        "probe_lmax": args.probe_lmax,
        "probe_count": len(probe_modes),
        "flux_lmax": args.flux_lmax,
        "mesh_sizes": args.mesh_sizes,
        "thresholds": THRESHOLDS,
        "probe_basis_convention_max_abs": basis_convention_error,
        "cases": [
            {"shape": obj.shape, "size": obj.size, "rotation": obj.rotation}
            for obj in cases
        ],
    }
    if not z_eligible_coarse:
        return {
            "passed": False,
            "selected_maxh": None,
            "stopped_early": True,
            "stop_reason": "all adjacent mesh pairs fail the mandatory local Z convergence thresholds; global BEM and projection gates were not run",
            "analytic_errors": {},
            "comparisons": z_comparisons,
            "projection_q": None,
            "schur_linearity": None,
            "bem_metadata": {},
            **common,
        }

    bem_solutions = {}
    bem_metadata = {}
    for maxh in args.mesh_sizes:
        print(f"gate mesh h={maxh:.5f}: BEM sphere", flush=True)
        bem, metadata = solve_exterior_neumann_bem(
            [sphere],
            np.zeros((1, 3)),
            args.k,
            direction,
            points,
            maxh,
            gmres_tolerance=args.gmres_tolerance,
            assembler=args.assembler,
        )
        bem_solutions[maxh] = bem[0]
        bem_metadata[maxh] = metadata

    analytic_errors = {
        str(maxh): relative_l2(bem_solutions[maxh], analytic) for maxh in args.mesh_sizes
    }
    comparisons = []
    selected = None
    for z_row, (coarse, fine) in zip(z_comparisons, zip(args.mesh_sizes[:-1], args.mesh_sizes[1:])):
        bem_difference = relative_l2(bem_solutions[coarse], bem_solutions[fine])
        row = {
            **z_row,
            "coarse_h": coarse,
            "fine_h": fine,
            "analytic_sphere": analytic_errors[str(coarse)],
            "neumann_boundary_residual": float(
                bem_metadata[coarse]["neumann_boundary_residual"]
            ),
            "bem_refinement": bem_difference,
        }
        row["pre_projection_passed"] = bool(
            row["analytic_sphere"] <= THRESHOLDS["analytic_sphere"]
            and row["neumann_boundary_residual"]
            <= THRESHOLDS["neumann_boundary_residual"]
            and row["bem_refinement"] <= THRESHOLDS["bem_refinement"]
            and row["z_gate_passed"]
            and basis_convention_error <= THRESHOLDS["probe_basis_convention"]
        )
        comparisons.append(row)
        if selected is None and row["pre_projection_passed"]:
            selected = coarse

    projection_error = None
    schur_linearity_error = None
    if selected is not None:
        sphere_z = single_object_fingerprint(
            sphere,
            args.k,
            points,
            probe_modes,
            selected,
            canonical_radius=args.canonical_radius,
            flux_lmax=args.flux_lmax,
        )
        sphere_total = bem_solutions[selected][None, :]
        projected = project_flux(sphere_z[None, :, :], sphere_total, phi_pinv)[0]
        rng = np.random.default_rng(args.seed + 9173)
        coefficients = rng.normal(size=len(probe_modes)) + 1j * rng.normal(
            size=len(probe_modes)
        )
        linear_boundary = phi @ coefficients
        linear_direct = single_object_flux_for_values(
            sphere,
            args.k,
            points,
            linear_boundary,
            selected,
            canonical_radius=args.canonical_radius,
            boundary_lmax=args.probe_lmax,
            flux_lmax=args.flux_lmax,
        )
        schur_linearity_error = relative_l2(sphere_z @ coefficients, linear_direct)
        full = single_object_flux_for_values(
            sphere,
            args.k,
            points,
            sphere_total[0],
            selected,
            canonical_radius=args.canonical_radius,
            boundary_lmax=args.flux_lmax,
            flux_lmax=args.flux_lmax,
        )
        projection_error = relative_l2(projected, full)

    passed = bool(
        selected is not None
        and projection_error is not None
        and schur_linearity_error is not None
        and schur_linearity_error <= THRESHOLDS["schur_linearity"]
        and projection_error <= THRESHOLDS["projection_q"]
    )
    return {
        "passed": passed,
        "selected_maxh": selected if passed else None,
        "analytic_errors": analytic_errors,
        "comparisons": comparisons,
        "projection_q": projection_error,
        "schur_linearity": schur_linearity_error,
        "bem_metadata": {str(key): value for key, value in bem_metadata.items()},
        **common,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--mesh-sizes", nargs="+", type=float, default=[0.20, 0.14, 0.10, 0.07])
    parser.add_argument("--cases-per-shape", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260923)
    parser.add_argument("--k", type=float, default=DEFAULT_K)
    parser.add_argument("--canonical-radius", type=float, default=DEFAULT_CANONICAL_RADIUS)
    parser.add_argument("--interface-nodes", type=int, default=DEFAULT_INTERFACE_NODES)
    parser.add_argument("--probe-lmax", type=int, default=DEFAULT_PROBE_LMAX)
    parser.add_argument("--flux-lmax", type=int, default=DEFAULT_FLUX_LMAX)
    parser.add_argument("--gmres-tolerance", type=float, default=DEFAULT_GMRES_TOL)
    parser.add_argument("--assembler", choices=("dense", "fmm"), default="dense")
    args = parser.parse_args()
    args.mesh_sizes = sorted(set(args.mesh_sizes), reverse=True)
    if len(args.mesh_sizes) < 2:
        raise ValueError("at least two mesh sizes are required")
    return args


def main() -> None:
    args = parse_args()
    report = run_gate(args)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"passed": report["passed"], "selected_maxh": report["selected_maxh"], "report": str(output)}, ensure_ascii=False))
    if not report["passed"]:
        raise SystemExit(3)


if __name__ == "__main__":
    main()
