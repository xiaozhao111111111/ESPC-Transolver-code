"""Reusable scene-label core for the structural Neumann V6 pipeline.

This module consumes already-qualified single-body modal DtN operators.  It
does not build CAD/FEM/BEM meshes, accept geometries, write datasets, or
provide a release gate.  It rotates cached local operators, solves the
Foldy--Lax system, and derives the agreed scene fields in complex128.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from . import formal_neumann_3d as formal
from . import tmatrix_neumann_3d as tmatrix


@dataclass(frozen=True)
class LocalModalOperator:
    """A qualified base-orientation local operator cached by geometry ID."""

    geometry_id: str
    modal_dtn: np.ndarray  # [121, 100], output l<=10, input l<=9
    transition: np.ndarray  # [100, 100], outgoing T matrix through l=9

    def __post_init__(self) -> None:
        modal = np.asarray(self.modal_dtn)
        transition = np.asarray(self.transition)
        if not self.geometry_id:
            raise ValueError("geometry_id must be non-empty")
        if modal.shape != (121, 100):
            raise ValueError("modal_dtn must have shape [121, 100] (lout=10, lin=9)")
        if transition.shape != (100, 100):
            raise ValueError("transition must have shape [100, 100] (lmax=9)")
        if not np.isfinite(modal).all() or not np.isfinite(transition).all():
            raise FloatingPointError("local operator contains non-finite values")

    @classmethod
    def from_modal_dtn(
        cls,
        geometry_id: str,
        modal_dtn: np.ndarray,
        *,
        k: float = formal.DEFAULT_K,
        canonical_radius: float = formal.DEFAULT_CANONICAL_RADIUS,
        lmax: int = 9,
    ) -> "LocalModalOperator":
        transition, metadata = tmatrix.transition_matrix(
            np.asarray(modal_dtn, dtype=np.complex128),
            k=k,
            canonical_radius=canonical_radius,
            lmax=lmax,
        )
        if metadata["conversion_residual"] > 1.0e-10:
            raise ArithmeticError("local DtN-to-T conversion residual exceeds 1e-10")
        return cls(geometry_id, np.asarray(modal_dtn, dtype=np.complex128), transition)


class HarmonicRotationCache:
    """Cache one harmonic quadrature basis and derive nested SO(3) matrices."""

    def __init__(self, quadrature: tmatrix.AngularQuadrature, lmax: int = 10):
        if lmax < 9:
            raise ValueError("V6 scene labels require rotation matrices through l=10")
        self.quadrature = quadrature
        self.lmax = int(lmax)
        self.modes = formal.spherical_modes(self.lmax)
        self.basis = formal.spherical_harmonic_matrix(quadrature.directions, self.modes)
        self.weights = np.asarray(quadrature.weights, dtype=np.float64)
        if self.basis.shape != (len(self.weights), (self.lmax + 1) ** 2):
            raise ValueError("quadrature basis has an incompatible shape")

    def matrix(self, rotation: np.ndarray) -> np.ndarray:
        rotation = np.asarray(rotation, dtype=np.float64)
        if (
            rotation.shape != (3, 3)
            or not np.allclose(rotation.T @ rotation, np.eye(3), atol=2.0e-6)
            or not np.isclose(np.linalg.det(rotation), 1.0, atol=2.0e-6)
        ):
            raise ValueError("rotation must be a proper SO(3) matrix")
        left, _, right = np.linalg.svd(rotation)
        rotation = left @ right
        if np.linalg.det(rotation) < 0.0:
            left[:, -1] *= -1.0
            rotation = left @ right
        rotated = formal.spherical_harmonic_matrix(
            self.quadrature.directions @ rotation, self.modes
        )
        result = self.basis.conj().T @ (self.weights[:, None] * rotated)
        defect = np.linalg.norm(result.conj().T @ result - np.eye(len(self.modes))) / np.sqrt(
            len(self.modes)
        )
        if not np.isfinite(result).all() or defect > 1.0e-8:
            raise ArithmeticError(f"harmonic rotation is not unitary (defect={defect:.3e})")
        return result


def rotate_local_operator(
    operator: LocalModalOperator,
    rotation: np.ndarray,
    rotation_cache: HarmonicRotationCache,
) -> tuple[np.ndarray, np.ndarray]:
    """Return rotated [121,100] DtN and [100,100] outgoing T matrices.

    The l<=9 representation is the leading principal block of the l<=10
    representation; this nested convention is independently regression-
    tested against the project's direct l<=9 rotation implementation.
    """
    d10 = rotation_cache.matrix(rotation)
    d9 = d10[:100, :100]
    modal = d10 @ np.asarray(operator.modal_dtn, dtype=np.complex128) @ d9.conj().T
    transition = d9 @ np.asarray(operator.transition, dtype=np.complex128) @ d9.conj().T
    if not np.isfinite(modal).all() or not np.isfinite(transition).all():
        raise FloatingPointError(f"rotated operator is non-finite: {operator.geometry_id}")
    return modal, transition


def generate_scene_labels(
    geometry_ids: Sequence[str],
    operators: Mapping[str, LocalModalOperator],
    centers: np.ndarray,
    rotations: np.ndarray,
    incident_direction: np.ndarray,
    interface_points: np.ndarray,
    probe_values: np.ndarray,
    phi_pinv: np.ndarray,
    *,
    k: float = formal.DEFAULT_K,
    canonical_radius: float = formal.DEFAULT_CANONICAL_RADIUS,
    lmax: int = 9,
    quadrature: tmatrix.AngularQuadrature | None = None,
    rotation_cache: HarmonicRotationCache | None = None,
) -> tuple[dict[str, np.ndarray], dict]:
    """Build one in-memory scene label from prequalified local operators.

    The caller must separately enforce the V6 library, BEM, occupancy,
    provenance, dataset-integrity, and publication gates.  No files are
    written here.
    """
    if lmax != 9:
        raise ValueError("V6 label reference currently requires lmax=9")
    geometry_ids = tuple(str(value) for value in geometry_ids)
    count = len(geometry_ids)
    if count < 1:
        raise ValueError("a scene must contain at least one object")
    centers = np.asarray(centers, dtype=np.float64)
    rotations = np.asarray(rotations, dtype=np.float64)
    direction = np.asarray(incident_direction, dtype=np.float64)
    points = np.asarray(interface_points, dtype=np.float64)
    probes = np.asarray(probe_values, dtype=np.complex128)
    phi_pinv = np.asarray(phi_pinv, dtype=np.complex128)
    if centers.shape != (count, 3) or rotations.shape != (count, 3, 3):
        raise ValueError("centers/rotations do not match geometry_ids")
    if direction.shape != (3,) or not np.isfinite(direction).all():
        raise ValueError("incident_direction must be a finite 3-vector")
    direction_norm = float(np.linalg.norm(direction))
    if not np.isclose(direction_norm, 1.0, atol=2.0e-6):
        raise ValueError("incident_direction must be unit length")
    if points.shape != (formal.DEFAULT_INTERFACE_NODES, 3):
        raise ValueError("interface_points must match the V6 [128,3] canonical surface")
    radii = np.linalg.norm(points, axis=1)
    if not np.allclose(radii, canonical_radius, rtol=0.0, atol=5.0e-6):
        raise ValueError("interface_points must lie on the canonical sphere")
    if probes.shape != (len(points), 64) or phi_pinv.shape != (64, len(points)):
        raise ValueError("probe_values/phi_pinv must be [M,64]/[64,M]")
    if not all(np.isfinite(array).all() for array in (centers, rotations, points, probes, phi_pinv)):
        raise FloatingPointError("scene inputs contain non-finite values")
    expected_probes = formal.spherical_harmonic_matrix(
        points, formal.spherical_modes(formal.DEFAULT_PROBE_LMAX)
    )
    if np.max(np.abs(probes - expected_probes)) > 2.0e-5:
        raise ValueError("probe_values do not match interface_points and the l<=7 mode order")
    projection_defect = np.linalg.norm(phi_pinv @ probes - np.eye(64)) / 8.0
    if projection_defect > 2.0e-5:
        raise ValueError(f"phi_pinv is not a left inverse of probe_values ({projection_defect:.3e})")
    if count > 1:
        distances = np.linalg.norm(centers[:, None, :] - centers[None, :, :], axis=-1)
        distances += np.eye(count) * (2.0 * canonical_radius + 1.0)
        if np.any(distances <= 2.0 * canonical_radius + 1.0e-10):
            raise ValueError("canonical spheres overlap or touch")

    quadrature = tmatrix.angular_quadrature() if quadrature is None else quadrature
    rotation_cache = HarmonicRotationCache(quadrature, 10) if rotation_cache is None else rotation_cache
    if rotation_cache.quadrature is not quadrature:
        raise ValueError("rotation cache and scene solver must share the same quadrature")

    if probes.shape[1] != len(formal.spherical_modes(formal.DEFAULT_PROBE_LMAX)):
        raise ValueError("probe width does not match l<=7")
    transitions: list[np.ndarray] = []
    z_rows: list[np.ndarray] = []
    for geometry_id, rotation in zip(geometry_ids, rotations):
        if geometry_id not in operators:
            raise KeyError(f"missing cached local operator {geometry_id!r}")
        operator = operators[geometry_id]
        if operator.geometry_id != geometry_id:
            raise ValueError("operator mapping key does not match geometry_id")
        if (
            not np.allclose(rotation.T @ rotation, np.eye(3), atol=2.0e-6)
            or not np.isclose(np.linalg.det(rotation), 1.0, atol=2.0e-6)
        ):
            raise ValueError(f"object rotation is not a proper SO(3) matrix: {geometry_id}")
        modal, transition = rotate_local_operator(operator, rotation, rotation_cache)
        transitions.append(transition)
        z_rows.append(tmatrix.nodal_fingerprint(modal, points, input_lmax=7))

    total, solve_metadata = tmatrix.solve_scene(
        transitions,
        centers,
        direction,
        points,
        k=k,
        canonical_radius=canonical_radius,
        lmax=lmax,
        quadrature=quadrature,
    )
    incident = formal.incident_on_interfaces(centers, points, direction, k)
    scattered = total - incident
    z_array = np.asarray(z_rows, dtype=np.complex128)
    q64 = formal.project_flux(z_array, total, phi_pinv)
    fields = {
        "Z": z_array,
        "u_total_gt": np.asarray(total, dtype=np.complex128),
        "u_inc": np.asarray(incident, dtype=np.complex128),
        "u_scattered_gt": np.asarray(scattered, dtype=np.complex128),
        "q64_gt": np.asarray(q64, dtype=np.complex128),
    }
    expected_shapes = {
        "Z": (count, len(points), 64),
        "u_total_gt": (count, len(points)),
        "u_inc": (count, len(points)),
        "u_scattered_gt": (count, len(points)),
        "q64_gt": (count, len(points)),
    }
    for name, array in fields.items():
        if array.shape != expected_shapes[name] or not np.isfinite(array).all():
            raise FloatingPointError(f"invalid scene field {name}: shape={array.shape}")
    if solve_metadata["linear_residual"] > 1.0e-10:
        raise ArithmeticError("Foldy--Lax residual exceeds 1e-10")
    identity_error = float(np.max(np.abs(fields["u_total_gt"] - fields["u_inc"] - fields["u_scattered_gt"])))
    q_recompute = formal.project_flux(z_array, fields["u_total_gt"], phi_pinv)
    q_error = float(np.max(np.abs(q_recompute - fields["q64_gt"])))
    if identity_error > 1.0e-12 or q_error > 1.0e-12:
        raise ArithmeticError("scene-field or projected-flux derivation identity failed")
    metadata = {
        **solve_metadata,
        "geometry_ids": list(geometry_ids),
        "object_count": count,
        "modal_input_lmax": 9,
        "modal_output_lmax": 10,
        "network_probe_lmax": 7,
        "q64_gt_type": "derived_ZP_u_total_not_independent_flux",
        "rotation_unitarity_gate": "passed-per-object-in-HarmonicRotationCache",
        "field_identity_max_abs_error": identity_error,
        "q64_recompute_max_abs_error": q_error,
        "labels_written": False,
        "release_gate_passed": False,
    }
    return fields, metadata
