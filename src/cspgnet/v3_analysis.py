"""Exploratory two-seed scene-paired comparisons for four fixed contrasts."""

from __future__ import annotations

import numpy as np


PRIMARY = (
    ("id_test", "geo-matched"),
    ("id_test", "geo-strong-adapted"),
    ("shape_ood", "geo-matched"),
    ("shape_ood", "geo-strong-adapted"),
)


def hierarchical_paired_comparison(z: np.ndarray, geometry: np.ndarray,
                                   *, seed: int = 20260925,
                                   resamples: int = 10000) -> dict:
    """Compare matched [two seeds, scenes] errors, conditional on shape library."""
    z = np.asarray(z, dtype=np.float64)
    geometry = np.asarray(geometry, dtype=np.float64)
    if z.shape != geometry.shape or z.ndim != 2 or z.shape[0] != 2:
        raise ValueError("paired arrays must have identical [2 seeds, scenes] shape")
    if z.shape[1] < 2 or not np.isfinite(z).all() or not np.isfinite(geometry).all():
        raise ValueError("nonfinite or insufficient scene-level errors")
    difference = z - geometry
    per_seed = difference.mean(axis=1)
    rng = np.random.default_rng(seed)
    draws = np.empty(resamples, dtype=np.float64)
    for index in range(resamples):
        chosen_seeds = rng.integers(0, 2, size=2)
        chosen_scenes = rng.integers(0, z.shape[1], size=z.shape[1])
        draws[index] = difference[np.ix_(chosen_seeds, chosen_scenes)].mean()
    # 98.75% simultaneous Bonferroni interval for four primary contrasts.
    alpha = 0.05 / len(PRIMARY)
    interval = np.quantile(draws, (alpha / 2, 1 - alpha / 2))
    return {
        "scenes": int(z.shape[1]),
        "seed_mean_differences": per_seed.tolist(),
        "mean_difference": float(difference.mean()),
        "median_paired_scene_difference": float(np.median(difference.mean(axis=0))),
        "bootstrap_simultaneous_98_75_ci": interval.tolist(),
        "bootstrap_directional_tail_fraction_not_p_value": float(
            min(1.0, 2.0 * min(np.mean(draws >= 0), np.mean(draws <= 0)))
        ),
        "favorable_seeds": int(np.sum(per_seed < 0)),
        "meets_exploratory_two_seed_screen": bool(interval[1] < 0 and np.sum(per_seed < 0) == 2),
        "statistical_scope": "exploratory; two seeds cannot establish the former 4/5 seed criterion",
        "uncertainty_scope": "conditional on the finite sampled base-shape libraries",
    }


def four_primary_contrasts(error_arrays: dict[tuple[str, str], np.ndarray]) -> dict:
    expected = {(split, model) for split in ("id_test", "shape_ood")
                for model in ("z", "geo-matched", "geo-strong-adapted")}
    if set(error_arrays) != expected:
        raise ValueError(f"expected all preregistered arrays, got {set(error_arrays)}")
    results = {}
    for split, geometry in PRIMARY:
        name = f"{split}: z - {geometry}"
        results[name] = hierarchical_paired_comparison(
            error_arrays[(split, "z")], error_arrays[(split, geometry)],
        )
    return results
