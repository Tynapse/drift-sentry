#!/usr/bin/env python
"""Combine a frozen DriftSentry train subset with strict reward hard negatives.

Validation and test files are accepted only as leakage checks. Their rows are
never copied into the output. Selection is deterministic and outputs are never
overwritten.
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

SPACE_RE = re.compile(r"\s+")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-train", required=True, type=Path)
    parser.add_argument("--hard-negative", action="append", required=True, type=Path)
    parser.add_argument("--forbidden", action="append", default=[], type=Path, help="Validation/test JSONL; repeatable.")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--max-new-rows", type=int, default=15000)
    parser.add_argument("--simhash-max-distance", type=int, default=3)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    for path in [args.base_train, *args.hard_negative, *args.forbidden]:
        if not path.is_file():
            raise SystemExit(f"input does not exist: {path}")
    if args.output.exists() or args.manifest.exists():
        raise SystemExit("refusing to overwrite output or manifest")
    if args.max_new_rows <= 0:
        raise SystemExit("--max-new-rows must be positive")
    if not 0 <= args.simhash_max_distance <= 3:
        raise SystemExit("--simhash-max-distance must be between 0 and 3")

    forbidden_rows = [row for path in args.forbidden for row in read_jsonl(path)]
    forbidden_hashes = {prompt_hash(row) for row in forbidden_rows}
    forbidden_semantic = build_semantic_index(forbidden_rows)
    base_rows = list(read_jsonl(args.base_train))
    base_hashes = {prompt_hash(row) for row in base_rows}
    base_semantic = build_semantic_index(base_rows)
    if len(base_hashes) != len(base_rows):
        raise SystemExit("base train contains duplicate normalized prompts")
    leakage = base_hashes & forbidden_hashes
    if leakage:
        raise SystemExit(f"base train overlaps forbidden validation/test prompts: {len(leakage)}")

    rejected: Counter[str] = Counter()
    candidates: list[tuple[str, dict[str, Any]]] = []
    for path in args.hard_negative:
        for row in read_jsonl(path):
            try:
                require_strict_reward(row)
            except ValueError:
                rejected["not_strict_reward"] += 1
                continue
            fingerprint = prompt_hash(row)
            if fingerprint in forbidden_hashes:
                rejected["forbidden_split_overlap"] += 1
                continue
            if fingerprint in base_hashes:
                rejected["base_duplicate"] += 1
                continue
            prompt_fingerprint = simhash64(prompt_text(row))
            if near_duplicate(forbidden_semantic, prompt_fingerprint, args.simhash_max_distance):
                rejected["forbidden_split_semantic_overlap"] += 1
                continue
            if near_duplicate(base_semantic, prompt_fingerprint, args.simhash_max_distance):
                rejected["base_semantic_duplicate"] += 1
                continue
            candidates.append((hashlib.sha256(f"{row.get('id', '')}|{fingerprint}".encode()).hexdigest(), row))
    candidates.sort(key=lambda item: item[0])

    selected_new: list[dict[str, Any]] = []
    selected_hashes: set[str] = set()
    selected_semantic: dict[tuple[int, int], list[int]] = defaultdict(list)
    for _rank, row in candidates:
        fingerprint = prompt_hash(row)
        if fingerprint in selected_hashes:
            rejected["hard_negative_duplicate"] += 1
            continue
        semantic_fingerprint = simhash64(prompt_text(row))
        if near_duplicate(selected_semantic, semantic_fingerprint, args.simhash_max_distance):
            rejected["hard_negative_semantic_duplicate"] += 1
            continue
        selected_hashes.add(fingerprint)
        add_simhash(selected_semantic, semantic_fingerprint)
        selected_new.append(row)
        if len(selected_new) >= args.max_new_rows:
            break
    if not selected_new:
        raise SystemExit("no strict reward hard negatives survived filtering")

    output_rows = [*base_rows, *selected_new]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=args.output.parent, delete=False) as handle:
        temporary = Path(handle.name)
        for row in output_rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(args.output)

    manifest = {
        "schema_version": "judge6.reward-augmented-manifest.v1",
        "base_train": descriptor(args.base_train),
        "hard_negative_files": [descriptor(path) for path in args.hard_negative],
        "forbidden_files": [descriptor(path) for path in args.forbidden],
        "base_rows": len(base_rows),
        "new_reward_rows": len(selected_new),
        "output_rows": len(output_rows),
        "selection": "strict reward only; ascending sha256(id + normalized prompt); exact prompt leakage excluded",
        "simhash_max_distance": args.simhash_max_distance,
        "rejected": dict(sorted(rejected.items())),
        "output_file": str(args.output),
        "output_sha256": sha256_file(args.output),
    }
    args.manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


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
                raise SystemExit(f"row is not an object: {path}:{line_number}")
            yield row


def prompt_hash(row: dict[str, Any]) -> str:
    normalized = prompt_text(row)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def prompt_text(row: dict[str, Any]) -> str:
    messages = row.get("messages")
    if not isinstance(messages, list):
        raise SystemExit(f"row has no messages: {row.get('id')}")
    text = "\n".join(
        f"{message.get('role', '')}:{message.get('content', '')}"
        for message in messages
        if isinstance(message, dict) and message.get("role") != "assistant"
    )
    return SPACE_RE.sub(" ", text).strip().lower()


def require_strict_reward(row: dict[str, Any]) -> None:
    messages = row.get("messages")
    if not isinstance(messages, list) or not messages:
        raise ValueError("missing messages")
    assistant = messages[-1]
    if not isinstance(assistant, dict) or assistant.get("role") != "assistant":
        raise ValueError("missing label")
    payload = json.loads(str(assistant.get("content") or ""))
    metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    values = {
        str(payload.get("tier") or "").strip().lower(),
        str(metadata.get("target_tier") or "").strip().lower(),
        str(metadata.get("teacher_tier") or "").strip().lower(),
    }
    if values != {"reward"}:
        raise ValueError("reward label mismatch")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def descriptor(path: Path) -> dict[str, Any]:
    return {"path": str(path), "sha256": sha256_file(path), "rows": count_rows(path)}


def count_rows(path: Path) -> int:
    with path.open(encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def simhash64(text: str) -> int:
    tokens = re.findall(r"[a-z0-9가-힣_]+", text, flags=re.IGNORECASE)
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


def build_semantic_index(rows: list[dict[str, Any]]) -> dict[tuple[int, int], list[int]]:
    index: dict[tuple[int, int], list[int]] = defaultdict(list)
    for row in rows:
        add_simhash(index, simhash64(prompt_text(row)))
    return index


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


if __name__ == "__main__":
    raise SystemExit(main())
