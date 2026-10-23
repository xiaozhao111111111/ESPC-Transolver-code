"""Family-agnostic acceptance policy and final manifest for the v6 geometry pool.

This module performs no CAD/FEM/BEM work and does not generate candidates. It
encodes only the user-approved exact genus quotas, the strict independent-flux
gate, and the distinction between terminal numerical failure and operational
or implementation interruptions.
"""
from __future__ import annotations

from collections import Counter
import math
import re
from typing import Any


SCHEMA = "structural-smooth-v6-validated-library-v1"
GENUS_COUNTS = {0: 200, 1: 150, 2: 100, 3: 50}
STRICT_FLUX_LIMIT = 0.03
TERMINAL_NUMERICAL_REASONS = {
    "independent_flux_projection_over_3_percent",
    "local_operator_mesh_sequence_not_converged",
    "registered_modal_accuracy_gate_failed",
    "registered_solver_accuracy_gate_failed",
}
OPERATIONAL_STOP_CLASSES = {"implementation_error", "resource_blocked", "interrupted", "provenance_error"}


def _hash(value: Any, label: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-fA-F]{64}", value) is None:
        raise ValueError(f"{label} must be a SHA-256 hexadecimal digest")
    return value.lower()


def _finite_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{label} must be finite numeric data")
    return float(value)


def classify_attempt(report: dict[str, Any]) -> dict[str, Any]:
    """Classify a single candidate receipt without silently losing evidence.

    Return actions are policy decisions only; callers persist the report and
    history before changing the active candidate queue.
    """
    if not isinstance(report, dict):
        raise ValueError("attempt report must be an object")
    status = report.get("status")
    genus = report.get("genus")
    if type(genus) is not int or genus not in GENUS_COUNTS:
        raise ValueError("attempt report requires genus 0, 1, 2, or 3")

    if status == "pre_cad_rejection":
        if report.get("cad_mesh_started") is not False or not isinstance(report.get("reason"), str):
            raise ValueError("pre-CAD rejection must have a reason and confirm no mesh/solver started")
        return {"action": "resample_same_genus_pre_cad", "genus": genus,
                "counts_as_terminal_numerical_failure": False}

    failure_class = report.get("failure_class")
    if failure_class in OPERATIONAL_STOP_CLASSES:
        return {"action": "halt_version_for_review", "genus": genus,
                "reason": failure_class, "counts_as_terminal_numerical_failure": False}

    if status == "passed":
        if report.get("registered_sequence_complete") is not True:
            raise ValueError("a passed operator must complete the registered mesh/modal sequence")
        if report.get("topology_passed") is not True or report.get("local_solver_passed") is not True:
            raise ValueError("passed operator receipt is missing topology or local solver success")
        error = _finite_number(report.get("independent_flux_projection_error"),
                               "independent_flux_projection_error")
        if not 0 <= error < STRICT_FLUX_LIMIT:
            raise ValueError("passed status contradicts the strict independent-flux error < 3% gate")
        return {"action": "accept_into_genus_quota", "genus": genus,
                "independent_flux_projection_error": error,
                "counts_as_terminal_numerical_failure": False}

    if status == "numerical_terminal_fail":
        if report.get("registered_sequence_complete") is not True:
            raise ValueError("coarse-only failure is not terminal; finish the registered sequence")
        reason = report.get("terminal_reason")
        if reason not in TERMINAL_NUMERICAL_REASONS:
            raise ValueError("terminal reason is not on the frozen numerical-failure allowlist")
        error = report.get("independent_flux_projection_error")
        if error is not None:
            error = _finite_number(error, "independent_flux_projection_error")
        if (reason == "independent_flux_projection_over_3_percent"
                and (error is None or error < STRICT_FLUX_LIMIT)):
            raise ValueError("projection failure classification contradicts its measured error")
        return {"action": "exclude_from_active_pool_and_replace_same_genus",
                "genus": genus, "terminal_reason": reason,
                "independent_flux_projection_error": error,
                "counts_as_terminal_numerical_failure": True,
                "preserve_historical_receipt": True}

    if status in {"running", "queued", "interrupted"}:
        return {"action": "resume_or_halt_same_candidate", "genus": genus,
                "counts_as_terminal_numerical_failure": False}
    raise ValueError(f"unrecognized/unsafe candidate status: {status!r}")


def build_validated_library_manifest(
    accepted_entries: list[dict[str, Any]], *, source_sha256: str,
    provenance_sha256: str, attempt_history_sha256: str,
) -> dict[str, Any]:
    """Create the only library schema accepted by the v6 layout planner."""
    _hash(source_sha256, "source_sha256")
    _hash(provenance_sha256, "provenance_sha256")
    _hash(attempt_history_sha256, "attempt_history_sha256")
    if not isinstance(accepted_entries, list) or len(accepted_entries) != 500:
        raise ValueError("v6 release requires exactly 500 accepted structures")
    ids: set[str] = set()
    seen_hashes: dict[str, set[str]] = {
        "structure_sha256": set(), "operator_sha256": set(), "report_sha256": set(),
    }
    counts: Counter[int] = Counter()
    rows: list[dict[str, Any]] = []
    for entry in accepted_entries:
        if not isinstance(entry, dict):
            raise ValueError("accepted library entry must be an object")
        identifier = entry.get("id")
        if not isinstance(identifier, str) or not identifier or identifier in ids:
            raise ValueError("accepted library IDs must be nonempty and unique")
        ids.add(identifier)
        genus = entry.get("genus")
        if type(genus) is not int or genus not in GENUS_COUNTS:
            raise ValueError(f"invalid genus for {identifier}")
        topology = entry.get("topology")
        local = entry.get("local_operator")
        if not isinstance(topology, dict) or topology.get("passed") is not True or topology.get("genus") != genus:
            raise ValueError(f"topology audit is missing/failed for {identifier}")
        if not isinstance(local, dict) or local.get("passed") is not True:
            raise ValueError(f"local operator report is missing/failed for {identifier}")
        if local.get("registered_sequence_complete") is not True:
            raise ValueError(f"registered FEM/modal sequence incomplete for {identifier}")
        flux_error = _finite_number(local.get("independent_flux_projection_error"),
                                    f"{identifier}.independent_flux_projection_error")
        if not 0 <= flux_error < STRICT_FLUX_LIMIT:
            raise ValueError(f"independent flux projection error is not strictly below 3% for {identifier}")
        volume = _finite_number(entry.get("physical_volume"), f"{identifier}.physical_volume")
        radius = _finite_number(entry.get("bounding_radius"), f"{identifier}.bounding_radius")
        if volume <= 0 or not 0 < radius < 0.9:
            raise ValueError(f"invalid volume/bounding radius for {identifier}")
        row = dict(entry)
        for field, hashes in seen_hashes.items():
            digest = _hash(row.get(field), f"{identifier}.{field}")
            if digest in hashes:
                raise ValueError(f"duplicate {field} in accepted library")
            hashes.add(digest)
            row[field] = digest
        counts[genus] += 1
        rows.append(row)
    if counts != Counter(GENUS_COUNTS):
        raise ValueError(f"accepted genus counts must be exactly {GENUS_COUNTS}; got {dict(counts)}")
    return {
        "schema": SCHEMA,
        "status": "all-500-local-passed-not-released",
        "production_released": False,
        "new_labeled_training_samples": 0,
        "expected_count": 500,
        "accepted_count": 500,
        "all_local_operators_validated": True,
        "all_independent_flux_projection_checks_passed": True,
        "genus_counts": {str(k): v for k, v in GENUS_COUNTS.items()},
        "source_sha256": source_sha256.lower(),
        "provenance_sha256": provenance_sha256.lower(),
        "attempt_history_sha256": attempt_history_sha256.lower(),
        "entries": rows,
    }
