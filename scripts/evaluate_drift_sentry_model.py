#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import re
import statistics
import time
from collections import Counter
from pathlib import Path
from typing import Any

import torch
from sklearn.metrics import classification_report, f1_score, precision_recall_fscore_support
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from peft import PeftModel
except ModuleNotFoundError:  # pragma: no cover
    PeftModel = None  # type: ignore[assignment]


TIERS = ["goal", "reasoning", "environment", "integration", "memory", "reward", "normal"]
RISK_TIERS = [tier for tier in TIERS if tier != "normal"]
HIGH_RISK_TIERS = {"memory", "reward"}


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = load_rows(Path(args.data_file), args.max_samples)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        revision=args.model_revision or None,
        trust_remote_code=args.trust_remote_code,
    )
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        revision=args.model_revision or None,
        torch_dtype=torch.bfloat16 if args.dtype == "bf16" else torch.float16,
        device_map="auto",
        trust_remote_code=args.trust_remote_code,
    )
    if args.adapter:
        if PeftModel is None:
            raise SystemExit("peft is required when --adapter is set")
        model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()

    run_name = args.run_name or sanitize_name(args.adapter or args.model)
    predictions_path = output_dir / f"{run_name}.predictions.jsonl"
    metrics_path = output_dir / f"{run_name}.metrics.json"
    metrics = evaluate(
        model=model,
        tokenizer=tokenizer,
        rows=rows,
        predictions_path=predictions_path,
        batch_size=args.batch_size,
        max_input_tokens=args.max_input_tokens,
        max_new_tokens=args.max_new_tokens,
    )
    metrics.update(
        {
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "model": args.model,
            "model_revision": args.model_revision,
            "adapter": args.adapter,
            "data_file": args.data_file,
            "predictions_path": str(predictions_path),
        }
    )
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate any chat model on DriftSentry JSONL test data.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-revision", default="", help="Exact Hugging Face base/merged-model revision.")
    parser.add_argument("--adapter", default="")
    parser.add_argument("--data-file", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--run-name", default="")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-input-tokens", type=int, default=4096)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16")
    parser.add_argument("--trust-remote-code", action="store_true", default=True)
    return parser.parse_args()


def load_rows(path: Path, max_samples: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            assistant = row["messages"][-1]
            gold_payload = parse_jsonish(assistant["content"])
            gold = normalize_tier(gold_payload.get("tier") or row.get("metadata", {}).get("target_tier"))
            if gold not in TIERS:
                raise ValueError(f"invalid gold tier for {row.get('id')}: {gold!r}")
            rows.append({"id": row.get("id", ""), "messages": row["messages"][:-1], "gold_tier": gold})
            if max_samples and len(rows) >= max_samples:
                break
    return rows


def evaluate(
    *,
    model: Any,
    tokenizer: Any,
    rows: list[dict[str, Any]],
    predictions_path: Path,
    batch_size: int,
    max_input_tokens: int,
    max_new_tokens: int,
) -> dict[str, Any]:
    golds: list[str] = []
    preds: list[str] = []
    batch_elapsed: list[float] = []
    with predictions_path.open("w", encoding="utf-8") as writer:
        for start in range(0, len(rows), batch_size):
            batch = rows[start : start + batch_size]
            prompts = [build_prompt(tokenizer, row["messages"]) for row in batch]
            encoded = tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_input_tokens,
            )
            encoded = {key: value.to(model.device) for key, value in encoded.items()}
            started = time.perf_counter()
            with torch.inference_mode():
                generated = model.generate(
                    **encoded,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            batch_elapsed.append(time.perf_counter() - started)
            new_tokens = generated[:, encoded["input_ids"].shape[1] :]
            decoded = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
            for row, raw in zip(batch, decoded, strict=True):
                payload = parse_jsonish(raw)
                pred = normalize_tier(payload.get("tier"))
                if pred not in TIERS:
                    pred = "unknown"
                gold = row["gold_tier"]
                golds.append(gold)
                preds.append(pred)
                writer.write(
                    json.dumps(
                        {
                            "id": row["id"],
                            "gold_tier": gold,
                            "predicted_tier": pred,
                            "raw_output": raw,
                            "parsed_output": payload,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
    return compute_metrics(golds, preds, batch_elapsed, batch_size)


def compute_metrics(golds: list[str], preds: list[str], batch_elapsed: list[float], batch_size: int) -> dict[str, Any]:
    labels_with_unknown = TIERS + ["unknown"]
    confusion = {gold: {pred: 0 for pred in labels_with_unknown} for gold in TIERS}
    for gold, pred in zip(golds, preds, strict=True):
        confusion[gold][pred] += 1
    per_label_p, per_label_r, per_label_f1, per_label_support = precision_recall_fscore_support(
        golds,
        preds,
        labels=labels_with_unknown,
        zero_division=0.0,
    )
    high_risk_gold = sum(1 for gold in golds if gold in HIGH_RISK_TIERS)
    high_risk_exact = sum(1 for gold, pred in zip(golds, preds, strict=True) if gold in HIGH_RISK_TIERS and pred == gold)
    high_risk_family = sum(
        1 for gold, pred in zip(golds, preds, strict=True) if gold in HIGH_RISK_TIERS and pred in HIGH_RISK_TIERS
    )
    elapsed_sum = sum(batch_elapsed)
    sorted_elapsed = sorted(batch_elapsed)
    return {
        "samples": len(golds),
        "macro_f1_7way": f1_score(golds, preds, labels=TIERS, average="macro", zero_division=0.0),
        "macro_f1_6tier_excluding_normal": f1_score(golds, preds, labels=RISK_TIERS, average="macro", zero_division=0.0),
        "accuracy": sum(g == p for g, p in zip(golds, preds, strict=True)) / len(golds) if golds else 0.0,
        "high_risk_exact_recall": high_risk_exact / high_risk_gold if high_risk_gold else 0.0,
        "high_risk_family_recall": high_risk_family / high_risk_gold if high_risk_gold else 0.0,
        "parse_failure_rate": sum(pred == "unknown" for pred in preds) / len(preds) if preds else 0.0,
        "gold_counts": dict(Counter(golds)),
        "prediction_counts": dict(Counter(preds)),
        "confusion_matrix_gold_by_prediction": confusion,
        "per_label": {
            label: {
                "precision": float(per_label_p[index]),
                "recall": float(per_label_r[index]),
                "f1": float(per_label_f1[index]),
                "support": int(per_label_support[index]),
            }
            for index, label in enumerate(labels_with_unknown)
        },
        "latency": {
            "batch_size": batch_size,
            "batch_seconds_mean": statistics.fmean(batch_elapsed) if batch_elapsed else 0.0,
            "batch_seconds_p50": percentile(sorted_elapsed, 50),
            "batch_seconds_p95": percentile(sorted_elapsed, 95),
            "samples_per_second": len(golds) / elapsed_sum if elapsed_sum else 0.0,
        },
        "classification_report": classification_report(
            golds,
            preds,
            labels=labels_with_unknown,
            zero_division=0.0,
            output_dict=True,
        ),
    }


def build_prompt(tokenizer: Any, messages: list[dict[str, str]]) -> str:
    try:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    except TypeError:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


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
    if tier_match:
        return {"tier": tier_match.group(1)}
    for tier in TIERS:
        if re.search(rf"\b{re.escape(tier)}\b", stripped, flags=re.IGNORECASE):
            return {"tier": tier}
    return {}


def normalize_tier(value: Any) -> str:
    text = str(value or "").strip().lower().replace("-", "_")
    return {"safe": "normal", "pass": "normal", "environmental": "environment"}.get(text, text)


def percentile(sorted_values: list[float], pct: float) -> float:
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return sorted_values[0]
    rank = (len(sorted_values) - 1) * pct / 100
    lo = int(rank)
    hi = min(lo + 1, len(sorted_values) - 1)
    weight = rank - lo
    return sorted_values[lo] * (1 - weight) + sorted_values[hi] * weight


def sanitize_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")


if __name__ == "__main__":
    raise SystemExit(main())
