#!/usr/bin/env python3
"""Audit DriftSentry target-label and raw-teacher-label consistency."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


LABELS = ("goal", "reasoning", "environment", "integration", "memory", "reward", "normal")
PLANNED_PERCENT = {
    "goal": 18.0,
    "reasoning": 25.0,
    "environment": 12.0,
    "integration": 15.0,
    "memory": 10.0,
    "reward": 10.0,
    "normal": 10.0,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, action="append", required=True, help="DriftSentry JSONL file; repeatable")
    parser.add_argument("--output", type=Path, help="Optional JSON output path")
    return parser.parse_args()


def raw_teacher_tier(metadata: dict[str, Any]) -> str | None:
    raw = metadata.get("raw_teacher_text")
    if not isinstance(raw, str):
        return None
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return None
    tier = payload.get("label", {}).get("tier")
    return tier if tier in LABELS else None


def audit(paths: list[Path]) -> dict[str, Any]:
    all_target: Counter[str] = Counter()
    valid_target: Counter[str] = Counter()
    teacher: Counter[str] = Counter()
    confusion: Counter[tuple[str, str]] = Counter()
    excluded: Counter[str] = Counter()

    for path in paths:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                metadata = row.get("metadata", {})
                target = metadata.get("target_tier")
                all_target[str(target)] += 1
                observed = raw_teacher_tier(metadata)
                if target not in LABELS or observed is None:
                    excluded["invalid_target" if target not in LABELS else "missing_or_invalid_raw_teacher_tier"] += 1
                    continue
                valid_target[target] += 1
                teacher[observed] += 1
                confusion[(target, observed)] += 1

    valid_rows = sum(confusion.values())
    if valid_rows == 0:
        raise ValueError("no rows had both a valid target tier and a valid raw teacher tier")
    agreements = sum(confusion[(label, label)] for label in LABELS)
    observed_agreement = agreements / valid_rows
    expected_agreement = sum(valid_target[label] * teacher[label] for label in LABELS) / (valid_rows**2)
    cohen_kappa = (observed_agreement - expected_agreement) / (1.0 - expected_agreement)

    total_rows = sum(all_target.values())
    distribution = {
        label: {
            "rows": all_target[label],
            "observed_percent": 100.0 * all_target[label] / total_rows,
            "planned_percent": PLANNED_PERCENT[label],
            "deviation_percentage_points": 100.0 * all_target[label] / total_rows - PLANNED_PERCENT[label],
        }
        for label in LABELS
    }

    return {
        "schema_version": "judge6.label-consistency-audit.v1",
        "inputs": [str(path) for path in paths],
        "method": (
            "Cohen's kappa between the independently assigned target_tier and the tier in "
            "metadata.raw_teacher_text.label, before any target-tier overwrite"
        ),
        "labels": list(LABELS),
        "rows_total": total_rows,
        "rows_valid_for_kappa": valid_rows,
        "rows_excluded_from_kappa": sum(excluded.values()),
        "exclusion_counts": dict(sorted(excluded.items())),
        "agreements": agreements,
        "disagreements": valid_rows - agreements,
        "observed_agreement": observed_agreement,
        "expected_agreement": expected_agreement,
        "cohen_kappa": cohen_kappa,
        "target_counts_all_rows": {label: all_target[label] for label in LABELS},
        "target_counts_valid_rows": {label: valid_target[label] for label in LABELS},
        "raw_teacher_counts_valid_rows": {label: teacher[label] for label in LABELS},
        "confusion": {
            target: {observed: confusion[(target, observed)] for observed in LABELS}
            for target in LABELS
        },
        "planned_distribution_percent": PLANNED_PERCENT,
        "distribution_audit": distribution,
        "maximum_absolute_distribution_deviation_percentage_points": max(
            abs(item["deviation_percentage_points"]) for item in distribution.values()
        ),
    }


def main() -> None:
    args = parse_args()
    result = audit(args.input)
    rendered = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
