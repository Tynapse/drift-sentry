#!/usr/bin/env python3
"""Build path- and endpoint-free public metrics/provenance from a private run."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--private-metrics", required=True, type=Path)
    predictions = result.add_mutually_exclusive_group(required=True)
    predictions.add_argument("--predictions", type=Path)
    predictions.add_argument("--predictions-sha256")
    result.add_argument("--dataset", required=True, type=Path)
    result.add_argument(
        "--evaluation-input-sha256",
        required=True,
        help="SHA-256 of the exact JSONL used to generate the supplied predictions.",
    )
    result.add_argument("--public-metrics", required=True, type=Path)
    result.add_argument("--public-provenance", required=True, type=Path)
    result.add_argument("--model-sha256", required=True)
    result.add_argument("--adapter-sha256", required=True)
    result.add_argument("--base-revision", required=True)
    result.add_argument("--replicas", type=int, required=True)
    result.add_argument("--concurrency", type=int, required=True)
    result.add_argument("--max-tokens", type=int, required=True)
    result.add_argument("--unchanged-message-rows", type=int, default=0)
    result.add_argument("--changed-message-rows", type=int, default=0)
    result.add_argument("--changed-rows-reinferred", type=int, default=0)
    result.add_argument("--accelerator", default="NVIDIA H200")
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.public_metrics.exists() or args.public_provenance.exists():
        raise SystemExit("refusing to overwrite public metrics or provenance")
    for name in ("evaluation_input_sha256", "model_sha256", "adapter_sha256"):
        require_sha256(getattr(args, name), name)
    require_hex(args.base_revision, "base_revision", 40)
    predictions_sha256 = sha256_file(args.predictions) if args.predictions else args.predictions_sha256
    require_sha256(predictions_sha256, "predictions_sha256")
    private = json.loads(args.private_metrics.read_text(encoding="utf-8"))
    fields = (
        "samples",
        "accuracy",
        "macro_f1_6tier_excluding_normal",
        "macro_f1_7way",
        "high_risk_exact_recall",
        "parse_failure_rate",
        "request_failure_count",
        "gold_counts",
        "prediction_counts",
        "per_label",
        "confusion_matrix",
    )
    metrics = {
        "schema_version": "drift-sentry.public-evaluation-metrics.v1",
        "model": "Tynapse/drift-sentry-4b-v1",
        "benchmark": "Tynapse/drift-sentry-bench-50k-v1",
        "dataset_sha256": sha256_file(args.dataset),
        "predictions_sha256": predictions_sha256,
        "conditions": {
            "accelerator": args.accelerator,
            "original_full_run_replicas": args.replicas,
            "concurrency": args.concurrency,
            "temperature": 0,
            "max_tokens": args.max_tokens,
            "response_format": "json_object",
            "publication_transform": {
                "evaluation_input_sha256": args.evaluation_input_sha256,
                "published_file_sha256": sha256_file(args.dataset),
                "unchanged_message_rows": args.unchanged_message_rows,
                "changed_message_rows": args.changed_message_rows,
                "changed_rows_reinferred": args.changed_rows_reinferred,
                "gold_labels_changed": 0,
                "metric_scope": (
                    "exact for the pre-publication evaluation input; not a fresh end-to-end inference "
                    "run for changed public prompts"
                ),
            },
        },
        "metrics": {field: private[field] for field in fields},
    }
    write_json(args.public_metrics, metrics)
    provenance = {
        "schema_version": "drift-sentry.public-evaluation-provenance.v1",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "model": {
            "repo_id": "Tynapse/drift-sentry-4b-v1",
            "served_name": "drift-sentry-4b-v1",
            "base_model": "Qwen/Qwen3.5-4B",
            "base_revision": args.base_revision,
            "merged_model_sha256": args.model_sha256,
            "adapter_sha256": args.adapter_sha256,
        },
        "dataset": {
            "repo_id": "Tynapse/drift-sentry-bench-50k-v1",
            "logical_file": "data/test.jsonl",
            "rows": private["samples"],
            "sha256": sha256_file(args.dataset),
        },
        "evaluation": metrics["conditions"],
        "artifacts": {
            "predictions_sha256": predictions_sha256,
            "public_metrics_sha256": sha256_file(args.public_metrics),
        },
        "redaction": {
            "removed": ["absolute filesystem paths", "loopback endpoint URLs", "host and GPU identifiers"],
            "internal_evidence_retained": True,
        },
    }
    write_json(args.public_provenance, provenance)
    print(json.dumps({"metrics": metrics["metrics"], "provenance": provenance}, indent=2, sort_keys=True))
    return 0


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_sha256(value: str | None, name: str) -> None:
    require_hex(value, name, 64)


def require_hex(value: str | None, name: str, length: int) -> None:
    if value is None or len(value) != length or any(character not in "0123456789abcdef" for character in value):
        raise SystemExit(f"{name} must be {length} lowercase hexadecimal characters")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
