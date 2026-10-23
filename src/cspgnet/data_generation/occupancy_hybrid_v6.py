"""Experimental analytic-prefiltered occupancy for filleted ellipsoid bores.

This helper is deliberately not connected to the V6 dataset writer. It uses an
analytic ellipsoid-minus-cylinder classifier away from the bore-lip
intersection band and delegates only that uncertainty band to a caller-supplied
solid classifier (for example, a curved NGSolve mesh's ``Contains`` method).
The uncertainty-band width must be explicit and independently audited before
any released dataset uses the result.
"""
from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np


def _validated_inputs(
    points: np.ndarray,
    axes: Sequence[float],
    bores: Sequence[Sequence[float]],
    uncertainty_band: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    coordinates = np.asarray(points, dtype=np.float64)
    semiaxes = np.asarray(axes, dtype=np.float64)
    bore_rows = np.asarray(bores, dtype=np.float64)
    if coordinates.ndim != 2 or coordinates.shape[1] != 3:
        raise ValueError("points must have shape [n,3]")
    if not np.isfinite(coordinates).all():
        raise ValueError("points must be finite")
    if semiaxes.shape != (3,) or not np.isfinite(semiaxes).all() or np.any(semiaxes <= 0):
        raise ValueError("axes must contain three finite positive semiaxes")
    if bore_rows.size == 0:
        bore_rows = np.empty((0, 3), dtype=np.float64)
    if bore_rows.ndim != 2 or bore_rows.shape[1] != 3 or not np.isfinite(bore_rows).all():
        raise ValueError("bores must have shape [n,3] with finite (x,y,radius) rows")
    if len(bore_rows) and np.any(bore_rows[:, 2] <= 0):
        raise ValueError("bore radii must be positive")
    band = float(uncertainty_band)
    if not np.isfinite(band) or band <= 0:
        raise ValueError("uncertainty_band must be finite and positive")
    return coordinates, semiaxes, bore_rows, band


def analytic_bore_mask(
    points: np.ndarray,
    axes: Sequence[float],
    bores: Sequence[Sequence[float]],
) -> np.ndarray:
    """Classify the unfilleted ellipsoid minus through-cylinder solids."""
    coordinates = np.asarray(points, dtype=np.float64)
    semiaxes = np.asarray(axes, dtype=np.float64)
    bore_rows = np.asarray(bores, dtype=np.float64)
    if coordinates.ndim != 2 or coordinates.shape[1] != 3 or not np.isfinite(coordinates).all():
        raise ValueError("points must be a finite [n,3] array")
    if semiaxes.shape != (3,) or not np.isfinite(semiaxes).all() or np.any(semiaxes <= 0):
        raise ValueError("axes must contain three finite positive semiaxes")
    if bore_rows.size == 0:
        bore_rows = np.empty((0, 3), dtype=np.float64)
    if bore_rows.ndim != 2 or bore_rows.shape[1] != 3 or not np.isfinite(bore_rows).all():
        raise ValueError("bores must have shape [n,3]")
    if len(bore_rows) and np.any(bore_rows[:, 2] <= 0):
        raise ValueError("bore radii must be positive")

    normalized = coordinates / semiaxes[None, :]
    inside_parent = np.einsum("ij,ij->i", normalized, normalized) <= 1.0
    in_bore = np.zeros(len(coordinates), dtype=bool)
    for x_center, y_center, radius in bore_rows:
        radial_squared = ((coordinates[:, 0] - x_center) ** 2
                          + (coordinates[:, 1] - y_center) ** 2)
        in_bore |= radial_squared <= radius * radius
    return inside_parent & ~in_bore


def _approximate_ellipsoid_distance(points: np.ndarray, axes: np.ndarray) -> np.ndarray:
    """First-order signed-distance estimate; used only to select fallback rows."""
    scaled = points / axes[None, :]
    radius = np.linalg.norm(scaled, axis=1)
    phi = radius - 1.0
    gradient = points / (axes[None, :] ** 2)
    gradient_norm = np.linalg.norm(gradient, axis=1)
    distance = np.full(len(points), np.inf, dtype=np.float64)
    np.divide(np.abs(phi), gradient_norm, out=distance, where=gradient_norm > 0)
    return distance


def hybrid_bore_mask(
    points: np.ndarray,
    axes: Sequence[float],
    bores: Sequence[Sequence[float]],
    *,
    uncertainty_band: float,
    exact_contains: Callable[[float, float, float], Any],
    exact_contains_many: Callable[[np.ndarray], Any] | None = None,
) -> tuple[np.ndarray, dict]:
    """Return a candidate mask with exact-classifier fallback near bore lips.

    A point is uncertain only when it lies within ``uncertainty_band`` of both
    the parent ellipsoid surface and at least one cylindrical bore wall. This
    is a screening rule, not an error bound: the caller must validate the
    resulting mask against a finer independently generated reference.
    """
    coordinates, semiaxes, bore_rows, band = _validated_inputs(
        points, axes, bores, uncertainty_band
    )
    mask = analytic_bore_mask(coordinates, semiaxes, bore_rows)
    uncertain = np.zeros(len(coordinates), dtype=bool)
    if len(bore_rows):
        near_parent = _approximate_ellipsoid_distance(coordinates, semiaxes) <= band
        for x_center, y_center, radius in bore_rows:
            radial_distance = np.hypot(coordinates[:, 0] - x_center,
                                       coordinates[:, 1] - y_center)
            near_bore = np.abs(radial_distance - radius) <= band
            uncertain |= near_parent & near_bore

    indices = np.flatnonzero(uncertain)
    if len(indices) and exact_contains_many is not None:
        fallback = np.asarray(exact_contains_many(coordinates[indices]))
        if fallback.shape != (len(indices),):
            raise ValueError("exact_contains_many must return one classification per fallback point")
        mask[indices] = fallback.astype(bool, copy=False)
        fallback_mode = "batched"
    else:
        for index in indices:
            x, y, z = coordinates[index]
            mask[index] = bool(exact_contains(float(x), float(y), float(z)))
        fallback_mode = "scalar"
    audit = {
        "method": "analytic-ellipsoid-cylinder-with-exact-boundary-band-fallback",
        "accuracy_status": "experimental-requires-independent-refinement-audit",
        "uncertainty_band": band,
        "point_count": int(len(coordinates)),
        "fallback_query_count": int(len(indices)),
        "fallback_fraction": float(len(indices) / max(len(coordinates), 1)),
        "fallback_mode": fallback_mode,
        "mask_true_count": int(np.count_nonzero(mask)),
    }
    return mask, audit
