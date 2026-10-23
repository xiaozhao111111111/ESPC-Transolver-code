"""Canonical-surface 3D Helmholtz data generation.

This is the first executable CSP-GNet data version based on the supplied
method proposal:

* each object is embedded in the same spherical canonical surface;
* spherical-harmonic Dirichlet probes produce a single-object response
  fingerprint;
* a global finite-element solve supplies multi-object boundary labels.

The FEM boundary mesh is geometry-dependent.  Responses are therefore
resampled onto a fixed Fibonacci sphere interface before being written to the
dataset.  This keeps the neural-network interface fixed while retaining the
finite-element Schur/reaction computation inside each geometry.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
try:
    # SciPy <= 1.16 exposes the legacy argument order used by this generator.
    from scipy.special import sph_harm as _sph_harm
except ImportError:
    # SciPy >= 1.17 uses (degree, order, polar, azimuth) instead.
    from scipy.special import sph_harm_y

    def _sph_harm(order, degree, azimuth, polar):
        return sph_harm_y(degree, order, polar, azimuth)
from scipy.spatial import cKDTree

from netgen.occ import (
    Axes,
    Box,
    Ellipsoid,
    Sphere,
    Vec,
    OCCGeometry,
)
from ngsolve import (
    BilinearForm,
    GridFunction,
    H1,
    LinearForm,
    Mesh,
    NodeId,
    VERTEX,
    dx,
    grad,
    ds,
    exp,
    specialcf,
    x,
    y,
    z,
)
from ngsolve.bem import SphericalHarmonicsCF


@dataclass(frozen=True)
class ObjectSpec:
    shape: str
    size: tuple[float, ...]


def fibonacci_sphere(num_points: int, radius: float) -> np.ndarray:
    """Return deterministic approximately uniform points on a sphere."""
    indices = np.arange(num_points, dtype=np.float64) + 0.5
    zc = 1.0 - 2.0 * indices / num_points
    r = np.sqrt(np.maximum(0.0, 1.0 - zc * zc))
    golden = math.pi * (3.0 - math.sqrt(5.0))
    phi = golden * indices
    return radius * np.column_stack((r * np.cos(phi), r * np.sin(phi), zc))


def spherical_modes(lmax: int) -> list[tuple[int, int]]:
    return [(ell, order) for ell in range(lmax + 1) for order in range(-ell, ell + 1)]


def spherical_probe_values(points: np.ndarray, modes: Sequence[tuple[int, int]]) -> np.ndarray:
    """Evaluate normalized complex spherical harmonics on fixed sphere points."""
    unit = points / np.linalg.norm(points, axis=1, keepdims=True)
    polar = np.arccos(np.clip(unit[:, 2], -1.0, 1.0))
    azimuth = np.arctan2(unit[:, 1], unit[:, 0])
    # scipy uses sph_harm(m, n, azimuth, polar), matching the convention used
    # by NGSolve's SphericalHarmonicsCF for the canonical sphere.
    return np.column_stack(
        [_sph_harm(order, ell, azimuth, polar) for ell, order in modes]
    )


def make_obstacle(spec: ObjectSpec):
    """Create one centered OCC obstacle for the supported first-version shapes."""
    if spec.shape == "sphere":
        return Sphere((0, 0, 0), float(spec.size[0]))
    if spec.shape == "ellipsoid":
        # Netgen/OCC has a deterministic surface-meshing failure for some
        # ascending semi-axis orders.  Sorting the radii only permutes the
        # ellipsoid axes and preserves the sampled size distribution while
        # keeping the geometry robust for long-running dataset generation.
        r1, r2, r3 = sorted((float(v) for v in spec.size), reverse=True)
        return Ellipsoid(
            Axes((0, 0, 0), (0, 0, 1)),
            r1,
            r2,
            r3,
        )
    if spec.shape == "box":
        hx, hy, hz = (float(v) for v in spec.size)
        return Box((-hx, -hy, -hz), (hx, hy, hz))
    raise ValueError(f"Unsupported first-version shape: {spec.shape}")


def move_shape(shape, center: Sequence[float]):
    return shape.Move(Vec(*(float(v) for v in center)))


def _outer_nodes(mesh, boundary_name: str) -> tuple[np.ndarray, np.ndarray]:
    region = mesh.Boundaries(boundary_name)
    node_numbers = sorted({vertex.nr for element in region.Elements() for vertex in element.vertices})
    coords = np.asarray(
        [tuple(mesh[NodeId(VERTEX, node_number)].point) for node_number in node_numbers],
        dtype=np.float64,
    )
    return np.asarray(node_numbers, dtype=np.int64), coords


def _resample_surface_values(
    source_points: np.ndarray,
    source_values: np.ndarray,
    target_points: np.ndarray,
    neighbors: int = 8,
) -> np.ndarray:
    """Inverse-distance resampling from a FEM boundary to the fixed interface."""
    tree = cKDTree(source_points)
    distances, indices = tree.query(target_points, k=min(neighbors, len(source_points)))
    if distances.ndim == 1:
        distances = distances[:, None]
        indices = indices[:, None]
    weights = 1.0 / np.maximum(distances, 1.0e-12)
    weights /= weights.sum(axis=1, keepdims=True)
    return np.sum(weights * source_values[indices], axis=1)


def _prepare_local_system(mesh, k: float, boundary_condition: str):
    """Assemble and factor one local FEM system for all probe RHS vectors."""
    if boundary_condition not in {"dirichlet", "neumann"}:
        raise ValueError("boundary_condition must be 'dirichlet' or 'neumann'")

    dirichlet = "canonical|scatterer" if boundary_condition == "dirichlet" else "canonical"
    fes = H1(mesh, order=1, complex=True, dirichlet=dirichlet)
    trial, test = fes.TnT()
    form = BilinearForm(fes)
    form += grad(trial) * grad(test) * dx - k * k * trial * test * dx
    form.Assemble()

    source = LinearForm(fes)
    source.Assemble()

    # This is the A_II factorization in the proposal.  It is deliberately
    # created once and reused for every spherical-harmonic RHS.
    inverse = form.mat.Inverse(freedofs=fes.FreeDofs())

    mass = BilinearForm(fes)
    mass += trial * test * ds("canonical")
    mass.Assemble()
    ones = source.vec.CreateVector()
    ones.FV().NumPy()[:] = 1.0
    lumped_mass = mass.mat * ones
    lumped_vector = source.vec.CreateVector()
    lumped_vector.data = lumped_mass
    mass_values = np.maximum(np.abs(lumped_vector.FV().NumPy()), 1.0e-14)

    node_numbers, coordinates = _outer_nodes(mesh, "canonical")
    return fes, form, source, inverse, mass_values, node_numbers, coordinates


def _solve_all_boundary_probes(
    mesh,
    k: float,
    boundary_condition: str,
    modes: Sequence[tuple[int, int]],
    interface_points: np.ndarray,
) -> np.ndarray:
    """Compute all columns of Z=S Phi with one FEM factorization."""
    fes, form, source, inverse, mass_values, node_numbers, coordinates = _prepare_local_system(
        mesh, k, boundary_condition
    )

    response = np.empty((len(interface_points), len(modes)), dtype=np.complex128)
    for column, mode in enumerate(modes):
        probe = SphericalHarmonicsCF(max(mode[0], 0))
        probe.sh[mode[0], mode[1]] = 1
        solution = GridFunction(fes)
        solution.Set(probe, definedon=mesh.Boundaries("canonical"))

        correction = source.vec.CreateVector()
        correction.data = source.vec - form.mat * solution.vec
        solution.vec.data += inverse * correction

        # The boundary reaction is the FEM residual on the constrained
        # interface, i.e. the Schur response for this probe.
        reaction = form.mat * solution.vec - source.vec
        reaction_vector = reaction.CreateVector()
        reaction_vector.data = reaction
        flux_density = reaction_vector.FV().NumPy() / mass_values
        values = flux_density[node_numbers]
        response[:, column] = _resample_surface_values(coordinates, values, interface_points)
    return response


def single_object_fingerprint(
    spec: ObjectSpec,
    k: float,
    boundary_condition: str,
    canonical_radius: float,
    interface_points: np.ndarray,
    modes: Sequence[tuple[int, int]],
    maxh: float,
) -> np.ndarray:
    """Compute Z=S Phi for one geometry without forming a full Schur matrix."""
    outer = Sphere((0, 0, 0), canonical_radius)
    outer.faces.name = "canonical"
    obstacle = make_obstacle(spec)
    obstacle.faces.name = "scatterer"
    geometry = OCCGeometry(outer - obstacle)
    mesh = Mesh(geometry.GenerateMesh(maxh=maxh))

    return _solve_all_boundary_probes(mesh, k, boundary_condition, modes, interface_points)


def _global_shape(spec: ObjectSpec, center: Sequence[float], index: int):
    shape = move_shape(make_obstacle(spec), center)
    shape.faces.name = f"obstacle_{index}"
    return shape


def solve_global_label(
    objects: Sequence[ObjectSpec],
    centers: np.ndarray,
    k: float,
    boundary_condition: str,
    interface_points: np.ndarray,
    incident_direction: np.ndarray,
    maxh: float,
    outer_radius: float,
) -> np.ndarray:
    """Solve the multi-object total field on a truncated 3D FEM domain."""
    outer = Sphere((0, 0, 0), outer_radius)
    outer.faces.name = "radiation"
    domain = outer
    for index, (spec, center) in enumerate(zip(objects, centers)):
        domain = domain - _global_shape(spec, center, index)

    mesh = Mesh(OCCGeometry(domain).GenerateMesh(maxh=maxh))
    if boundary_condition == "dirichlet":
        fes = H1(mesh, order=1, complex=True, dirichlet="obstacle_.*")
    elif boundary_condition == "neumann":
        fes = H1(mesh, order=1, complex=True)
    else:
        raise ValueError("boundary_condition must be 'dirichlet' or 'neumann'")

    trial, test = fes.TnT()
    form = BilinearForm(fes)
    form += grad(trial) * grad(test) * dx - k * k * trial * test * dx
    form += -1j * k * trial * test * ds("radiation")
    form.Assemble()

    direction = tuple(float(v) for v in incident_direction)
    incident = exp(1j * k * (direction[0] * x + direction[1] * y + direction[2] * z))
    normal = specialcf.normal(3)
    incident_normal = 1j * k * (direction[0] * normal[0] + direction[1] * normal[1] + direction[2] * normal[2]) * incident
    rhs = LinearForm(fes)
    rhs += (incident_normal - 1j * k * incident) * test * ds("radiation")
    rhs.Assemble()

    solution = GridFunction(fes)
    if boundary_condition == "dirichlet":
        solution.Set(0, definedon=mesh.Boundaries("obstacle_.*"))
    correction = rhs.vec.CreateVector()
    correction.data = rhs.vec - form.mat * solution.vec
    solution.vec.data += form.mat.Inverse(freedofs=fes.FreeDofs()) * correction

    values = []
    for center in centers:
        shifted = interface_points + np.asarray(center, dtype=np.float64)[None, :]
        values.append(np.asarray([solution(mesh(*point)) for point in shifted], dtype=np.complex128))
    return np.asarray(values)


def random_object(rng: np.random.Generator, shape: str | None = None) -> ObjectSpec:
    shape = shape or str(rng.choice(["sphere", "ellipsoid", "box"]))
    if shape == "sphere":
        return ObjectSpec(shape, (float(rng.uniform(0.25, 0.55)),))
    if shape == "ellipsoid":
        return ObjectSpec(shape, tuple(float(v) for v in rng.uniform(0.25, 0.55, size=3)))
    if shape == "box":
        return ObjectSpec(shape, tuple(float(v) for v in rng.uniform(0.22, 0.5, size=3)))
    raise ValueError(shape)


def random_centers(rng: np.random.Generator, count: int, min_distance: float) -> np.ndarray:
    centers: list[np.ndarray] = []
    while len(centers) < count:
        candidate = rng.uniform(-2.8, 2.8, size=3)
        if all(np.linalg.norm(candidate - existing) >= min_distance for existing in centers):
            centers.append(candidate)
    return np.asarray(centers, dtype=np.float64)


def generate_dataset(args: argparse.Namespace) -> None:
    raise RuntimeError(
        "The legacy canonical_3d generator uses uniform FEM meshes and is "
        "disabled for new data. Use the graded-mesh-gated Neumann generator."
    )

    rng = np.random.default_rng(args.seed)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    interface = fibonacci_sphere(args.interface_nodes, args.canonical_radius)
    modes = spherical_modes(args.probe_lmax)
    probes = spherical_probe_values(interface, modes)

    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest.setdefault("samples", [])
    else:
        manifest = {
            "method": "CSP-GNet canonical-surface 3D prototype",
            "equation": "scalar Helmholtz",
            "canonical_radius": args.canonical_radius,
            "interface_nodes": args.interface_nodes,
            "probe_lmax": args.probe_lmax,
            "probe_modes": modes,
            "boundary_condition": args.boundary_condition,
            "samples": [],
        }
    completed = {item["file"] for item in manifest["samples"]}

    for sample_index in range(args.num_samples):
        k = float(rng.uniform(args.k_min, args.k_max))
        objects = [
            random_object(rng, shape if shape else None)
            for shape in args.shapes[: args.num_objects]
        ]
        while len(objects) < args.num_objects:
            objects.append(random_object(rng))
        centers = random_centers(rng, len(objects), min_distance=2.4 * args.canonical_radius)
        direction = rng.normal(size=3)
        direction /= np.linalg.norm(direction)

        sample_name = f"sample_{sample_index:06d}.npz"
        if sample_name in completed and (output / sample_name).exists():
            continue

        fingerprints = []
        for spec in objects:
            fingerprints.append(
                single_object_fingerprint(
                    spec,
                    k,
                    args.boundary_condition,
                    args.canonical_radius,
                    interface,
                    modes,
                    args.local_maxh,
                )
            )
        z_response = np.asarray(fingerprints)
        labels = solve_global_label(
            objects,
            centers,
            k,
            args.boundary_condition,
            interface,
            direction,
            args.global_maxh,
            args.global_outer_radius,
        )
        incident = np.asarray(
            [
                np.exp(1j * k * np.dot(direction, point + center))
                for center in centers
                for point in interface
            ],
            dtype=np.complex128,
        ).reshape(len(centers), len(interface))

        np.savez_compressed(
            output / sample_name,
            Z=z_response,
            Z_real=z_response.real.astype(np.float32),
            Z_imag=z_response.imag.astype(np.float32),
            u_surface_gt=labels,
            u_surface_gt_real=labels.real.astype(np.float32),
            u_surface_gt_imag=labels.imag.astype(np.float32),
            incident_surface=incident,
            interface_points=interface.astype(np.float32),
            probe_values=probes,
            centers=centers.astype(np.float32),
            k=np.asarray(k, dtype=np.float32),
            incident_direction=direction.astype(np.float32),
        )
        manifest["samples"].append(
            {
                "file": sample_name,
                "k": k,
                "incident_direction": direction.tolist(),
                "centers": centers.tolist(),
                "objects": [asdict(spec) for spec in objects],
            }
        )
        # Persist progress after every sample so a long 10k/1k generation
        # can be resumed safely after an interruption.
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print(f"generated {sample_name}: objects={len(objects)} k={k:.5f}")

    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--num-samples", type=int, default=1)
    parser.add_argument("--num-objects", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--boundary-condition", choices=["dirichlet", "neumann"], default="dirichlet")
    parser.add_argument("--canonical-radius", type=float, default=1.5)
    parser.add_argument("--interface-nodes", type=int, default=128)
    parser.add_argument("--probe-lmax", type=int, default=3)
    parser.add_argument("--k-min", type=float, default=1.2)
    parser.add_argument("--k-max", type=float, default=2.0)
    parser.add_argument("--local-maxh", type=float, default=0.65)
    parser.add_argument("--global-maxh", type=float, default=0.8)
    parser.add_argument("--global-outer-radius", type=float, default=8.0)
    parser.add_argument(
        "--shapes",
        nargs="*",
        default=["sphere", "ellipsoid", "box"],
        help="Shapes used in order before random shapes fill the requested object count.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    generate_dataset(parse_args())
