"""Aggregate the five-seed formal Neumann ablations and select lambda_q."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path


LAMBDA_DIRS = {0.0: "lambda_0", 0.1: "lambda_0_1", 1.0: "lambda_1"}
METRICS = (
    "total_rel_l2",
    "scattered_rel_l2",
    "q64_rel_l2",
    "amplitude_rel_l2",
    "phase_mae_deg",
)


def _mean_std(values: list[float]) -> dict:
    if not values:
        return {"mean": None, "std": None, "values": []}
    return {
        "mean": statistics.mean(values),
        "std": statistics.stdev(values) if len(values) > 1 else 0.0,
        "values": values,
    }


def _read_five_runs(root: Path, condition: str) -> dict[int, dict]:
    runs = {}
    for seed in range(5):
        path = root / f"seed_{seed}" / "final_metrics.json"
        if not path.is_file():
            raise FileNotFoundError(f"missing completed run for {condition}, seed={seed}: {path}")
        runs[seed] = json.loads(path.read_text(encoding="utf-8"))
    return runs


def _summarize_runs(runs: dict[int, dict]) -> dict:
    def metrics_for(split: str) -> dict:
        rows = []
        for seed, result in sorted(runs.items()):
            item = result.get(split)
            if item is None:
                continue
            rows.append(item)
        if not rows:
            return {}
        return {
            metric: _mean_std([float(row[metric]["mean"]) for row in rows])
            for metric in METRICS
            if metric in rows[0]
        }

    return {
        "seed_count": len(runs),
        "best_val": metrics_for("best_val"),
        "test": metrics_for("test"),
        "count_ood_n8": metrics_for("count_ood_n8"),
    }


def summarize(root: Path, zero_z_root: Path | None, near_tie_tolerance: float) -> dict:
    if not math.isfinite(near_tie_tolerance) or near_tie_tolerance < 0:
        raise ValueError("near-tie tolerance must be a finite non-negative fraction")
    groups = {}
    raw_runs = {}
    for value, directory in LAMBDA_DIRS.items():
        runs = _read_five_runs(root / directory, f"lambda_q={value}")
        raw_runs[value] = runs
        groups[str(value)] = _summarize_runs(runs)

    validation_scores = {
        value: summary["best_val"]["total_rel_l2"]["mean"]
        for value, summary in groups.items()
    }
    best_score = min(validation_scores.values())
    eligible = [
        value
        for value, score in validation_scores.items()
        if score <= best_score * (1.0 + near_tie_tolerance)
    ]
    selected = min(
        eligible,
        key=lambda value: groups[value]["best_val"]["q64_rel_l2"]["mean"],
    )
    report = {
        "selection_rule": (
            "minimize mean across seeds of best-checkpoint validation complex relative L2; "
            "among runs within the configured relative near-tie tolerance, minimize validation q64 relative L2"
        ),
        "near_tie_relative_tolerance": near_tie_tolerance,
        "lambda_q": groups,
        "selection": {
            "lambda_q": float(selected),
            "best_validation_total_rel_l2": best_score,
            "eligible_near_ties": [float(value) for value in eligible],
            "validation_scores": {key: value for key, value in validation_scores.items()},
        },
    }
    if zero_z_root is not None:
        zero_runs = _read_five_runs(zero_z_root, "zero-Z-input")
        report["zero_z_input"] = {
            "matched_lambda_q": float(selected),
            **_summarize_runs(zero_runs),
        }
        report["comparison_at_selected_lambda"] = {
            "full_z_test": groups[selected]["test"],
            "zero_z_test": report["zero_z_input"]["test"],
        }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, help="full-Z root with lambda_*/seed_*/final_metrics.json")
    parser.add_argument("--report", required=True)
    parser.add_argument("--near-tie-relative-tolerance", type=float, default=0.01)
    parser.add_argument("--zero-z-root", default="")
    args = parser.parse_args()
    report = summarize(
        Path(args.root),
        Path(args.zero_z_root) if args.zero_z_root else None,
        args.near_tie_relative_tolerance,
    )
    output = Path(args.report)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(
        json.dumps(
            {
                "selected_lambda_q": report["selection"]["lambda_q"],
                "near_tie_relative_tolerance": report["near_tie_relative_tolerance"],
                "report": str(output),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
