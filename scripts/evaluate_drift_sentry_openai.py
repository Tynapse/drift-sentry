#!/usr/bin/env python
"""Evaluate DriftSentry JSONL through an OpenAI-compatible endpoint such as vLLM."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import concurrent.futures
import json
import math
import os
from pathlib import Path
import re
import statistics
import time
from typing import Any
import urllib.error
import urllib.request

TIERS = ("goal", "reasoning", "environment", "integration", "memory", "reward", "normal")
RISK_TIERS = TIERS[:-1]
HIGH_RISK = {"memory", "reward"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-url",
        action="append",
        dest="base_urls",
        required=True,
        help="OpenAI-compatible /v1 base URL; repeat to distribute requests across replicas.",
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--data-file", required=True, type=Path)
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--metrics", required=True, type=Path)
    parser.add_argument("--concurrency", type=int, default=64)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--progress-every", type=int, default=500)
    parser.add_argument("--api-key-env", default="VLLM_API_KEY")
    parser.add_argument("--no-response-format", action="store_true")
    parser.add_argument("--resume", action="store_true", help="Skip ids already present in the predictions file.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.data_file.is_file():
        raise SystemExit(f"data file does not exist: {args.data_file}")
    if args.concurrency <= 0:
        raise SystemExit("--concurrency must be positive")
    if args.metrics.exists():
        raise SystemExit(f"refusing to overwrite metrics: {args.metrics}")
    if args.predictions.exists() and not args.resume:
        raise SystemExit(f"predictions already exist; pass --resume: {args.predictions}")
    api_key = os.environ.get(args.api_key_env, "")
    if not api_key:
        raise SystemExit(f"missing API key environment variable: {args.api_key_env}")

    rows = load_rows(args.data_file, args.max_samples)
    completed = load_completed(args.predictions) if args.resume else {}
    successful_ids = {row_id for row_id, result in completed.items() if not result.get("error")}
    pending = [row for row in rows if row["id"] not in successful_ids]
    args.predictions.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if args.resume and args.predictions.exists() else "x"
    started = time.perf_counter()
    failures = 0
    with args.predictions.open(mode, encoding="utf-8") as writer, concurrent.futures.ThreadPoolExecutor(
        max_workers=args.concurrency
    ) as pool:
        futures = [
            pool.submit(
                evaluate_one,
                row,
                base_url=args.base_urls[index % len(args.base_urls)],
                model=args.model,
                api_key=api_key,
                max_tokens=args.max_tokens,
                timeout=args.timeout,
                max_retries=args.max_retries,
                response_format=not args.no_response_format,
            )
            for index, row in enumerate(pending)
        ]
        for done, future in enumerate(concurrent.futures.as_completed(futures), start=1):
            result = future.result()
            if result.get("error"):
                failures += 1
            writer.write(json.dumps(result, ensure_ascii=False, sort_keys=True) + "\n")
            writer.flush()
            if done % args.progress_every == 0:
                print(
                    json.dumps(
                        {
                            "completed_this_run": done,
                            "pending_total": len(pending),
                            "request_failures": failures,
                            "elapsed_seconds": round(time.perf_counter() - started, 1),
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )

    predictions = load_completed(args.predictions)
    expected_ids = {row["id"] for row in rows}
    missing = sorted(expected_ids - set(predictions))
    if missing:
        raise SystemExit(f"predictions incomplete: missing {len(missing)} rows")
    ordered = [predictions[row["id"]] for row in rows]
    metrics = compute_metrics(ordered)
    metrics.update(
        {
            "model": args.model,
            "base_urls": args.base_urls,
            "data_file": str(args.data_file),
            "predictions_file": str(args.predictions),
            "wall_seconds_this_run": round(time.perf_counter() - started, 3),
        }
    )
    args.metrics.parent.mkdir(parents=True, exist_ok=True)
    args.metrics.write_text(json.dumps(metrics, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if metrics["request_failure_count"] == 0 else 2


def load_rows(path: Path, max_samples: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            row_id = str(row.get("id") or "")
            if not row_id or row_id in seen:
                raise SystemExit(f"missing or duplicate id: {path}:{line_number}: {row_id!r}")
            seen.add(row_id)
            messages = row.get("messages")
            if not isinstance(messages, list) or not messages:
                raise SystemExit(f"missing messages: {path}:{line_number}")
            assistant = messages[-1]
            if not isinstance(assistant, dict) or assistant.get("role") != "assistant":
                raise SystemExit(f"missing assistant gold: {path}:{line_number}")
            gold = normalize_tier(parse_jsonish(str(assistant.get("content") or "")).get("tier"))
            if gold not in TIERS:
                raise SystemExit(f"invalid gold tier: {path}:{line_number}: {gold!r}")
            rows.append({"id": row_id, "messages": messages[:-1], "gold_tier": gold})
            if max_samples and len(rows) >= max_samples:
                break
    return rows


def load_completed(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    results: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            row_id = str(row.get("id") or "")
            if not row_id:
                raise SystemExit(f"invalid resumed prediction: {path}:{line_number}: {row_id!r}")
            # A resumed run may append a successful retry after a failed row.
            # Last record wins, while the immutable JSONL preserves every attempt.
            results[row_id] = row
    return results


def evaluate_one(
    row: dict[str, Any],
    *,
    base_url: str,
    model: str,
    api_key: str,
    max_tokens: int,
    timeout: float,
    max_retries: int,
    response_format: bool,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": model,
        "messages": row["messages"],
        "temperature": 0,
        "max_tokens": max_tokens,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    if response_format:
        payload["response_format"] = {"type": "json_object"}
    request_body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
    started = time.perf_counter()
    last_error: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            request = urllib.request.Request(
                base_url.rstrip("/") + "/chat/completions",
                data=request_body,
                headers=headers,
                method="POST",
            )
            with urllib.request.urlopen(request, timeout=timeout) as response:
                body = json.loads(response.read())
            raw = str(body["choices"][0]["message"]["content"] or "")
            prediction = normalize_tier(parse_jsonish(raw).get("tier"))
            if prediction not in TIERS:
                prediction = "unknown"
            usage = body.get("usage") if isinstance(body.get("usage"), dict) else {}
            return {
                "id": row["id"],
                "gold_tier": row["gold_tier"],
                "predicted_tier": prediction,
                "raw_output": raw,
                "latency_ms": (time.perf_counter() - started) * 1000,
                "prompt_tokens": usage.get("prompt_tokens"),
                "completion_tokens": usage.get("completion_tokens"),
                "attempts": attempt + 1,
            }
        except (KeyError, IndexError, TypeError, json.JSONDecodeError, urllib.error.URLError, TimeoutError) as exc:
            last_error = exc
            if attempt < max_retries:
                time.sleep(min(8.0, 0.5 * (2**attempt)))
    return {
        "id": row["id"],
        "gold_tier": row["gold_tier"],
        "predicted_tier": "unknown",
        "raw_output": "",
        "latency_ms": (time.perf_counter() - started) * 1000,
        "attempts": max_retries + 1,
        "error": str(last_error),
    }


def parse_jsonish(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped)
        stripped = re.sub(r"\s*```$", "", stripped)
    try:
        payload = json.loads(stripped)
        return payload if isinstance(payload, dict) else {}
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", stripped, flags=re.DOTALL)
    if match:
        try:
            payload = json.loads(match.group(0))
            return payload if isinstance(payload, dict) else {}
        except json.JSONDecodeError:
            pass
    tier_match = re.search(r'"?tier"?\s*[:=]\s*"?([A-Za-z_-]+)"?', stripped, flags=re.IGNORECASE)
    return {"tier": tier_match.group(1)} if tier_match else {}


def normalize_tier(value: Any) -> str:
    text = str(value or "").strip().lower().replace("-", "_")
    return {"safe": "normal", "pass": "normal", "environmental": "environment"}.get(text, text)


def compute_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    golds = [str(row["gold_tier"]) for row in rows]
    predictions = [str(row["predicted_tier"]) for row in rows]
    per_label: dict[str, dict[str, float | int]] = {}
    for label in (*TIERS, "unknown"):
        true_positive = sum(gold == label and pred == label for gold, pred in zip(golds, predictions, strict=True))
        false_positive = sum(gold != label and pred == label for gold, pred in zip(golds, predictions, strict=True))
        false_negative = sum(gold == label and pred != label for gold, pred in zip(golds, predictions, strict=True))
        support = sum(gold == label for gold in golds)
        precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
        recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_label[label] = {"precision": precision, "recall": recall, "f1": f1, "support": support}
    high_risk_support = sum(gold in HIGH_RISK for gold in golds)
    high_risk_exact = sum(gold in HIGH_RISK and gold == pred for gold, pred in zip(golds, predictions, strict=True))
    latencies = sorted(float(row.get("latency_ms") or 0.0) for row in rows)
    confusion: dict[str, Counter[str]] = defaultdict(Counter)
    for gold, prediction in zip(golds, predictions, strict=True):
        confusion[gold][prediction] += 1
    return {
        "samples": len(rows),
        "macro_f1_7way": statistics.fmean(float(per_label[label]["f1"]) for label in TIERS),
        "macro_f1_6tier_excluding_normal": statistics.fmean(float(per_label[label]["f1"]) for label in RISK_TIERS),
        "accuracy": sum(gold == pred for gold, pred in zip(golds, predictions, strict=True)) / len(rows) if rows else 0.0,
        "high_risk_exact_recall": high_risk_exact / high_risk_support if high_risk_support else 0.0,
        "parse_failure_rate": sum(pred == "unknown" for pred in predictions) / len(rows) if rows else 0.0,
        "request_failure_count": sum(bool(row.get("error")) for row in rows),
        "gold_counts": dict(Counter(golds)),
        "prediction_counts": dict(Counter(predictions)),
        "per_label": per_label,
        "confusion_matrix": {gold: dict(counter) for gold, counter in sorted(confusion.items())},
        "latency_ms": {
            "mean": statistics.fmean(latencies) if latencies else 0.0,
            "p50_type7": percentile(latencies, 50),
            "p95_type7": percentile(latencies, 95),
            "p99_type7": percentile(latencies, 99),
        },
    }


def percentile(sorted_values: list[float], pct: float) -> float:
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return sorted_values[0]
    rank = (len(sorted_values) - 1) * pct / 100
    lo = math.floor(rank)
    hi = math.ceil(rank)
    weight = rank - lo
    return sorted_values[lo] * (1 - weight) + sorted_values[hi] * weight


if __name__ == "__main__":
    raise SystemExit(main())
