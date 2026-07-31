#!/usr/bin/env python
"""Finalize a strict, deterministic DriftSentry public benchmark from generated JSONL.

The command is fail-closed: it writes the benchmark only when every requested
class quota can be filled after label, PII, exact-duplicate, and SimHash checks.
Reference files are read-only and are used only as deduplication surfaces.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
import tempfile
from typing import Any, Iterable

TIERS = ("goal", "reasoning", "environment", "integration", "memory", "reward", "normal")
DEFAULT_COUNTS = {
    "goal": 7143,
    "reasoning": 7143,
    "environment": 7143,
    "integration": 7143,
    "memory": 7143,
    "reward": 7143,
    "normal": 7142,
}
PRIVATE_METADATA_KEYS = {"teacher_base_url", "raw_teacher_text"}
SPACE_RE = re.compile(r"\s+")
NUMBER_RE = re.compile(r"\d+")
TOKEN_RE = re.compile(r"[a-z0-9가-힣_]+", re.IGNORECASE)
PII_PATTERNS = {
    "email": re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE),
    "korean_phone": re.compile(r"(?<!\d)01[016789][- ]?\d{3,4}[- ]?\d{4}(?!\d)"),
    "resident_id": re.compile(r"(?<!\d)\d{6}[- ]?[1-8]\d{6}(?!\d)"),
    "aws_access_key": re.compile(r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b"),
    "secret_token": re.compile(r"(?<![A-Za-z0-9])(?:sk|hf)[_-][A-Za-z0-9_-]{16,}(?![A-Za-z0-9_-])"),
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", action="append", required=True, type=Path, help="Generated JSONL; repeatable.")
    parser.add_argument("--reference", action="append", default=[], type=Path, help="Existing split for dedup; repeatable.")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument(
        "--target-count",
        action="append",
        default=[],
        metavar="TIER=COUNT",
        help="Override a class quota. Defaults sum to exactly 50,000.",
    )
    parser.add_argument(
        "--simhash-max-distance",
        type=int,
        default=3,
        help="Reject normalized prompts within this 64-bit SimHash Hamming distance; set -1 to disable.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not -1 <= args.simhash_max_distance <= 3:
        raise SystemExit("--simhash-max-distance must be -1 (disabled) or between 0 and 3")
    target_counts = parse_target_counts(args.target_count)
    ensure_new_output(args.output, args.manifest)
    for path in [*args.candidate, *args.reference]:
        if not path.is_file():
            raise SystemExit(f"input does not exist: {path}")

    reference_exact: set[str] = set()
    semantic_index: dict[tuple[int, int], list[int]] = defaultdict(list)
    reference_rows = 0
    for path in args.reference:
        for row in read_jsonl(path):
            prompt = normalized_prompt(row)
            if not prompt:
                continue
            reference_rows += 1
            reference_exact.add(sha256_text(prompt))
            add_simhash(semantic_index, simhash64(prompt))

    candidates: list[tuple[str, dict[str, Any]]] = []
    candidate_rows = 0
    for path in args.candidate:
        for row in read_jsonl(path):
            candidate_rows += 1
            rank = sha256_text(f"{row.get('id', '')}\n{normalized_prompt(row)}")
            candidates.append((rank, row))
    candidates.sort(key=lambda item: item[0])

    selected: list[dict[str, Any]] = []
    selected_counts: Counter[str] = Counter()
    rejected: Counter[str] = Counter()
    selected_exact: set[str] = set()
    selected_semantic: dict[tuple[int, int], list[int]] = defaultdict(list)

    for _rank, row in candidates:
        try:
            tier = strict_tier(row)
        except ValueError:
            rejected["label_invalid_or_mismatch"] += 1
            continue
        if tier not in target_counts or selected_counts[tier] >= target_counts[tier]:
            rejected["quota_filled"] += 1
            continue
        prompt = normalized_prompt(row)
        if not prompt:
            rejected["empty_prompt"] += 1
            continue
        public_row = sanitize_row(row)
        pii_kind = find_pii(json.dumps(public_row, ensure_ascii=False, sort_keys=True))
        if pii_kind:
            rejected[f"pii:{pii_kind}"] += 1
            continue
        exact = sha256_text(prompt)
        if exact in reference_exact:
            rejected["exact_reference_duplicate"] += 1
            continue
        if exact in selected_exact:
            rejected["exact_candidate_duplicate"] += 1
            continue
        fingerprint = simhash64(prompt)
        if args.simhash_max_distance >= 0 and near_duplicate(
            semantic_index, fingerprint, args.simhash_max_distance
        ):
            rejected["semantic_reference_duplicate"] += 1
            continue
        if args.simhash_max_distance >= 0 and near_duplicate(
            selected_semantic, fingerprint, args.simhash_max_distance
        ):
            rejected["semantic_candidate_duplicate"] += 1
            continue
        selected.append(public_row)
        selected_counts[tier] += 1
        selected_exact.add(exact)
        add_simhash(selected_semantic, fingerprint)
        if selected_counts == Counter(target_counts):
            break

    shortages = {tier: count - selected_counts[tier] for tier, count in target_counts.items() if selected_counts[tier] < count}
    if shortages:
        raise SystemExit(
            "insufficient accepted rows after strict filtering: "
            + json.dumps({"selected": dict(selected_counts), "shortages": shortages, "rejected": dict(rejected)}, sort_keys=True)
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=args.output.parent, delete=False) as handle:
        temporary_output = Path(handle.name)
        for row in selected:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    temporary_output.replace(args.output)

    manifest = {
        "schema_version": "judge6.public-benchmark-manifest.v1",
        "row_count": len(selected),
        "class_counts": {tier: selected_counts[tier] for tier in TIERS},
        "target_counts": target_counts,
        "candidate_rows": candidate_rows,
        "reference_rows_indexed": reference_rows,
        "candidate_files": file_descriptors(args.candidate),
        "reference_files": file_descriptors(args.reference),
        "filters": {
            "strict_teacher_target_match": True,
            "pii_patterns": sorted(PII_PATTERNS),
            "normalized_exact_hash": "sha256",
            "semantic_fingerprint": "64-bit token SimHash with four 16-bit exact bands",
            "simhash_max_distance": args.simhash_max_distance,
            "rejected": dict(sorted(rejected.items())),
        },
        "selection": "ascending sha256(id + normalized prompt), per-tier fixed quota",
        "output_file": args.output.name,
        "output_sha256": sha256_file(args.output),
    }
    args.manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def parse_target_counts(values: list[str]) -> dict[str, int]:
    if not values:
        return dict(DEFAULT_COUNTS)
    counts: dict[str, int] = {}
    for value in values:
        tier, separator, raw_count = value.partition("=")
        if not separator or tier not in TIERS:
            raise SystemExit(f"invalid --target-count: {value}")
        try:
            count = int(raw_count)
        except ValueError as exc:
            raise SystemExit(f"invalid --target-count: {value}") from exc
        if count < 0:
            raise SystemExit(f"negative --target-count: {value}")
        counts[tier] = count
    missing = sorted(set(TIERS) - set(counts))
    if missing:
        raise SystemExit(f"missing --target-count tiers: {', '.join(missing)}")
    return counts


def ensure_new_output(*paths: Path) -> None:
    existing = [str(path) for path in paths if path.exists()]
    if existing:
        raise SystemExit("refusing to overwrite existing output: " + ", ".join(existing))


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"invalid JSONL: {path}:{line_number}: {exc}") from exc
            if not isinstance(row, dict):
                raise SystemExit(f"JSONL row is not an object: {path}:{line_number}")
            yield row


def strict_tier(row: dict[str, Any]) -> str:
    messages = row.get("messages")
    if not isinstance(messages, list) or not messages:
        raise ValueError("missing messages")
    assistant = messages[-1]
    if not isinstance(assistant, dict) or assistant.get("role") != "assistant":
        raise ValueError("missing assistant label")
    payload = json.loads(str(assistant.get("content") or ""))
    tier = str(payload.get("tier") or "").strip().lower()
    metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    target = str(metadata.get("target_tier") or "").strip().lower()
    teacher = str(metadata.get("teacher_tier") or tier).strip().lower()
    if tier not in TIERS or target != tier or teacher != tier:
        raise ValueError("tier mismatch")
    return tier


def normalized_prompt(row: dict[str, Any]) -> str:
    messages = row.get("messages")
    if not isinstance(messages, list):
        return ""
    parts: list[str] = []
    for message in messages:
        if not isinstance(message, dict) or message.get("role") == "assistant":
            continue
        parts.append(f"{message.get('role', '')}:{message.get('content', '')}")
    text = "\n".join(parts).lower()
    text = NUMBER_RE.sub("<num>", text)
    return SPACE_RE.sub(" ", text).strip()


def sanitize_row(row: dict[str, Any]) -> dict[str, Any]:
    public = json.loads(json.dumps(row, ensure_ascii=False))
    metadata = public.get("metadata")
    if isinstance(metadata, dict):
        for key in PRIVATE_METADATA_KEYS:
            metadata.pop(key, None)
        metadata["split"] = "test"
        metadata["benchmark_name"] = "drift-sentry-bench-50k-v1"
    return public


def find_pii(text: str) -> str | None:
    for name, pattern in PII_PATTERNS.items():
        if pattern.search(text):
            return name
    return None


def simhash64(text: str) -> int:
    tokens = TOKEN_RE.findall(text)
    features = tokens + [f"{tokens[index]}::{tokens[index + 1]}" for index in range(len(tokens) - 1)]
    weights = [0] * 64
    for feature in set(features):
        digest = int.from_bytes(hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest(), "big")
        for bit in range(64):
            weights[bit] += 1 if (digest >> bit) & 1 else -1
    fingerprint = 0
    for bit, weight in enumerate(weights):
        if weight >= 0:
            fingerprint |= 1 << bit
    return fingerprint


def add_simhash(index: dict[tuple[int, int], list[int]], fingerprint: int) -> None:
    for band in range(4):
        index[(band, (fingerprint >> (band * 16)) & 0xFFFF)].append(fingerprint)


def near_duplicate(index: dict[tuple[int, int], list[int]], fingerprint: int, max_distance: int) -> bool:
    checked: set[int] = set()
    for band in range(4):
        key = (band, (fingerprint >> (band * 16)) & 0xFFFF)
        for candidate in index.get(key, []):
            if candidate in checked:
                continue
            checked.add(candidate)
            if (fingerprint ^ candidate).bit_count() <= max_distance:
                return True
    return False


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_descriptors(paths: list[Path]) -> list[dict[str, Any]]:
    return [{"name": f"{path.parent.name}/{path.name}", "sha256": sha256_file(path)} for path in paths]


if __name__ == "__main__":
    raise SystemExit(main())
