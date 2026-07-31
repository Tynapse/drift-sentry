#!/usr/bin/env python
"""Measure sequential OpenAI-compatible latency at an exact prompt-token count."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import socket
import statistics
import time
from typing import Any
import urllib.error
import urllib.request


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--sample-jsonl", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--hardware-label", required=True, help="Measured serving hardware recorded in the evidence JSON.")
    parser.add_argument("--requests", type=int, default=1000)
    parser.add_argument("--warmups", type=int, default=20)
    parser.add_argument("--target-prompt-tokens", type=int, default=2048)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--force-exact-output-tokens", action="store_true")
    parser.add_argument("--timeout-seconds", type=float, default=180.0)
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument("--api-key-env", default="VLLM_API_KEY")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output_path = Path(args.output)
    if output_path.exists():
        raise SystemExit(f"refusing to overwrite output: {output_path}")
    api_key = os.environ.get(args.api_key_env, "")
    if not api_key:
        raise SystemExit(f"missing API key environment variable: {args.api_key_env}")
    try:
        from transformers import AutoTokenizer
    except ModuleNotFoundError as exc:
        raise SystemExit("transformers is required; install tynapse-drift-sentry[train]") from exc

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True, local_files_only=True)
    sample_id, source_messages = load_messages(Path(args.sample_jsonl))
    prompt, padded_messages, prompt_tokens = build_exact_prompt(tokenizer, source_messages, args.target_prompt_tokens)
    prompt_sha256 = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    endpoint = args.base_url.rstrip("/") + "/completions"
    body: dict[str, Any] = {
        "model": args.model,
        "prompt": prompt,
        "max_tokens": args.max_tokens,
        "temperature": 0.0,
        "seed": 0,
    }
    expected_completion_tokens = args.max_tokens if args.force_exact_output_tokens else None
    if args.force_exact_output_tokens:
        body["min_tokens"] = args.max_tokens
    started_at = datetime.now(timezone.utc)
    cold = run_phase(
        name="cold",
        count=1,
        url=endpoint,
        body=body,
        api_key=api_key,
        timeout_seconds=args.timeout_seconds,
        target_prompt_tokens=prompt_tokens,
        expected_completion_tokens=expected_completion_tokens,
        progress_every=0,
    )
    warmup = run_phase(
        name="warmup",
        count=args.warmups,
        url=endpoint,
        body=body,
        api_key=api_key,
        timeout_seconds=args.timeout_seconds,
        target_prompt_tokens=prompt_tokens,
        expected_completion_tokens=expected_completion_tokens,
        progress_every=args.progress_every,
    )
    measured = run_phase(
        name="measured",
        count=args.requests,
        url=endpoint,
        body=body,
        api_key=api_key,
        timeout_seconds=args.timeout_seconds,
        target_prompt_tokens=prompt_tokens,
        expected_completion_tokens=expected_completion_tokens,
        progress_every=args.progress_every,
    )
    finished_at = datetime.now(timezone.utc)
    result = {
        "schema_version": "nipa.latency.v1",
        "started_at_utc": started_at.isoformat(),
        "finished_at_utc": finished_at.isoformat(),
        "host": socket.gethostname(),
        "hardware_label": args.hardware_label,
        "endpoint": endpoint,
        "model": args.model,
        "tokenizer_path": args.tokenizer,
        "sample_jsonl": args.sample_jsonl,
        "sample_id": sample_id,
        "message_roles": [message["role"] for message in padded_messages],
        "prompt_sha256": prompt_sha256,
        "target_prompt_tokens": args.target_prompt_tokens,
        "rendered_prompt_tokens": prompt_tokens,
        "max_output_tokens": args.max_tokens,
        "forced_exact_output_tokens": args.force_exact_output_tokens,
        "batch_size": 1,
        "concurrency": 1,
        "sampling": {"temperature": 0.0, "seed": 0},
        "percentile_method": "Hyndman-Fan type 7 linear interpolation",
        "cold": cold,
        "warmup": warmup,
        "measured": measured,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output_path), "measured": measured.get("latency_ms"), "failed": measured["failed"]}))
    return 0 if measured["failed"] == 0 and measured["succeeded"] == args.requests else 1


def load_messages(path: Path) -> tuple[str, list[dict[str, str]]]:
    with path.open(encoding="utf-8") as handle:
        first_line = handle.readline()
    row = json.loads(first_line)
    messages = row.get("messages")
    if not isinstance(messages, list) or not messages:
        raise ValueError("first JSONL row has no messages list")
    normalized = [
        {"role": str(message["role"]), "content": str(message["content"])}
        for message in messages
        if isinstance(message, dict) and "role" in message and "content" in message
    ]
    if normalized and normalized[-1]["role"] == "assistant":
        normalized.pop()
    if not normalized or not any(message["role"] == "user" for message in normalized):
        raise ValueError("sample must contain a user message before the assistant target")
    return str(row.get("id") or ""), normalized


def render_prompt(tokenizer: Any, messages: list[dict[str, str]]) -> tuple[str, int]:
    kwargs = {"tokenize": False, "add_generation_prompt": True}
    try:
        prompt = tokenizer.apply_chat_template(messages, enable_thinking=False, **kwargs)
    except TypeError:
        prompt = tokenizer.apply_chat_template(messages, **kwargs)
    ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    return prompt, len(ids)


def build_exact_prompt(
    tokenizer: Any, source_messages: list[dict[str, str]], target_tokens: int
) -> tuple[str, list[dict[str, str]], int]:
    user_index = max(index for index, message in enumerate(source_messages) if message["role"] == "user")
    base_messages = [dict(message) for message in source_messages]
    original = base_messages[user_index]["content"]
    _, base_tokens = render_prompt(tokenizer, base_messages)
    if base_tokens > target_tokens:
        raise ValueError(f"sample prompt already has {base_tokens} tokens, above target {target_tokens}")

    def candidate(repeats: int) -> tuple[str, list[dict[str, str]], int]:
        messages = [dict(message) for message in base_messages]
        messages[user_index]["content"] = original + " test" * repeats
        prompt, token_count = render_prompt(tokenizer, messages)
        return prompt, messages, token_count

    low = 0
    high = max(1, target_tokens - base_tokens)
    while candidate(high)[2] < target_tokens:
        high *= 2
    while low <= high:
        middle = (low + high) // 2
        prompt, messages, token_count = candidate(middle)
        if token_count == target_tokens:
            return prompt, messages, token_count
        if token_count < target_tokens:
            low = middle + 1
        else:
            high = middle - 1
    for repeats in range(max(0, high - 16), low + 17):
        prompt, messages, token_count = candidate(repeats)
        if token_count == target_tokens:
            return prompt, messages, token_count
    raise ValueError(f"could not construct an exact {target_tokens}-token prompt")


def post_completion(url: str, body: dict[str, Any], api_key: str, timeout_seconds: float) -> tuple[float, dict[str, Any]]:
    request = urllib.request.Request(
        url,
        data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
        method="POST",
    )
    started = time.perf_counter_ns()
    with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
        payload = json.loads(response.read())
    return (time.perf_counter_ns() - started) / 1_000_000, payload


def output_text(payload: dict[str, Any]) -> str:
    choices = payload.get("choices") or []
    if not choices:
        return ""
    first = choices[0]
    return str(first.get("text") or first.get("message", {}).get("content") or "")


def run_phase(
    *,
    name: str,
    count: int,
    url: str,
    body: dict[str, Any],
    api_key: str,
    timeout_seconds: float,
    target_prompt_tokens: int,
    expected_completion_tokens: int | None,
    progress_every: int,
) -> dict[str, Any]:
    latencies_ms: list[float] = []
    prompt_tokens: list[int] = []
    completion_tokens: list[int] = []
    output_sha256: list[str] = []
    failures: list[dict[str, Any]] = []
    for index in range(count):
        try:
            elapsed_ms, payload = post_completion(url, body, api_key, timeout_seconds)
            usage = payload.get("usage") or {}
            observed_prompt_tokens = int(usage.get("prompt_tokens") or 0)
            if observed_prompt_tokens != target_prompt_tokens:
                raise ValueError(f"server reported {observed_prompt_tokens} prompt tokens; expected {target_prompt_tokens}")
            observed_completion_tokens = int(usage.get("completion_tokens") or 0)
            if expected_completion_tokens is not None and observed_completion_tokens != expected_completion_tokens:
                raise ValueError(
                    f"server reported {observed_completion_tokens} completion tokens; expected {expected_completion_tokens}"
                )
            text = output_text(payload)
            latencies_ms.append(elapsed_ms)
            prompt_tokens.append(observed_prompt_tokens)
            completion_tokens.append(observed_completion_tokens)
            output_sha256.append(hashlib.sha256(text.encode("utf-8")).hexdigest())
        except (urllib.error.URLError, TimeoutError, ValueError, json.JSONDecodeError) as error:
            failures.append({"index": index, "error_type": type(error).__name__, "error": str(error)})
        if progress_every and (index + 1) % progress_every == 0:
            print(f"{name}: {index + 1}/{count} ok={len(latencies_ms)} failed={len(failures)}", flush=True)
    summary: dict[str, Any] = {
        "attempted": count,
        "succeeded": len(latencies_ms),
        "failed": len(failures),
        "failures": failures,
        "latencies_ms": latencies_ms,
        "prompt_tokens": {"expected": target_prompt_tokens, "observed_unique": sorted(set(prompt_tokens))},
        "completion_tokens": {
            "sum": sum(completion_tokens),
            "min": min(completion_tokens) if completion_tokens else None,
            "max": max(completion_tokens) if completion_tokens else None,
            "mean": statistics.fmean(completion_tokens) if completion_tokens else None,
            "values": completion_tokens,
        },
        "output_sha256": output_sha256,
    }
    if latencies_ms:
        summary["latency_ms"] = {
            "min": min(latencies_ms),
            "mean": statistics.fmean(latencies_ms),
            "p50_type7": percentile_type7(latencies_ms, 0.50),
            "p95_type7": percentile_type7(latencies_ms, 0.95),
            "p99_type7": percentile_type7(latencies_ms, 0.99),
            "max": max(latencies_ms),
        }
    return summary


def percentile_type7(values: list[float], percentile: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("no values")
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * percentile
    low = math.floor(rank)
    high = math.ceil(rank)
    weight = rank - low
    return ordered[low] * (1.0 - weight) + ordered[high] * weight


if __name__ == "__main__":
    raise SystemExit(main())
