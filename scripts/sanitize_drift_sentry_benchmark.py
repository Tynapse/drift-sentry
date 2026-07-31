#!/usr/bin/env python3
"""Create a deterministic public-safe copy of a DriftSentry JSONL benchmark.

The transform replaces credential- and PII-shaped strings with stable,
nonfunctional placeholders. It never writes raw matches to the audit report.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import tempfile
from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from typing import Any

PATTERNS = {
    "email": re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE),
    "korean_phone": re.compile(r"(?<!\d)01[016789][- ]?\d{3,4}[- ]?\d{4}(?!\d)"),
    "resident_id": re.compile(r"(?<!\d)\d{6}[- ]?[1-8]\d{6}(?!\d)"),
    "aws_access_key": re.compile(r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b"),
    "secret_token": re.compile(r"(?<![A-Za-z0-9])(?:sk|hf)[_-][A-Za-z0-9_-]{16,}(?![A-Za-z0-9_-])"),
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--audit", required=True, type=Path)
    parser.add_argument("--input-label", help="Logical input name for the audit; defaults to the input basename.")
    parser.add_argument("--output-label", help="Logical output name for the audit; defaults to the output basename.")
    parser.add_argument("--teacher-model", default="Qwen/Qwen3.6-27B")
    parser.add_argument("--teacher-revision", required=True)
    parser.add_argument("--benchmark-name", default="drift-sentry-bench-50k-v1")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.output.exists() or args.audit.exists():
        raise SystemExit("refusing to overwrite output or audit file")
    if not args.input.is_file():
        raise SystemExit(f"input does not exist: {args.input}")

    match_counts: Counter[str] = Counter()
    content_modified_rows = 0
    metadata_modified_rows = 0
    modified_rows_total = 0
    row_count = 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.audit.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=args.output.parent, delete=False) as handle:
        temporary_output = Path(handle.name)
        for row in read_jsonl(args.input):
            row_count += 1
            transformed, content_changed, metadata_changed = transform_row(
                row,
                match_counts,
                teacher_model=args.teacher_model,
                teacher_revision=args.teacher_revision,
                benchmark_name=args.benchmark_name,
            )
            content_modified_rows += int(content_changed)
            metadata_modified_rows += int(metadata_changed)
            modified_rows_total += int(content_changed or metadata_changed)
            handle.write(json.dumps(transformed, ensure_ascii=False, sort_keys=True) + "\n")
    temporary_output.replace(args.output)

    residual_counts = scan_file(args.output)
    if any(residual_counts.values()):
        args.output.unlink(missing_ok=True)
        raise SystemExit(f"sanitized output still contains restricted patterns: {dict(residual_counts)}")

    audit = {
        "schema_version": "drift-sentry.public-content-audit.v1",
        "method": "deterministic regex replacement; raw matched strings are never recorded",
        "input_file": args.input_label or args.input.name,
        "input_sha256": sha256_file(args.input),
        "output_file": args.output_label or args.output.name,
        "output_sha256": sha256_file(args.output),
        "rows": row_count,
        "rows_content_modified": content_modified_rows,
        "rows_metadata_modified": metadata_modified_rows,
        "rows_modified_total": modified_rows_total,
        "replacements": dict(sorted(match_counts.items())),
        "residual_matches": dict(sorted(residual_counts.items())),
        "teacher_model": args.teacher_model,
        "teacher_revision": args.teacher_revision,
        "benchmark_name": args.benchmark_name,
        "placeholder_format": "<SYNTHETIC_KIND_SHA256PREFIX>",
    }
    args.audit.write_text(json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise SystemExit(f"row is not an object: {path}:{line_number}")
            yield row


def transform_row(
    row: dict[str, Any],
    counts: Counter[str],
    *,
    teacher_model: str,
    teacher_revision: str,
    benchmark_name: str,
) -> tuple[dict[str, Any], bool, bool]:
    public = json.loads(json.dumps(row, ensure_ascii=False))
    content_changed = False
    metadata_changed = False
    messages = public.get("messages")
    if isinstance(messages, list):
        for message in messages:
            if not isinstance(message, dict) or not isinstance(message.get("content"), str):
                continue
            content, message_content_changed = replace_restricted(message["content"], counts)
            message["content"] = content
            content_changed = content_changed or message_content_changed
    metadata = public.get("metadata")
    if isinstance(metadata, dict):
        if metadata.get("teacher_model") != teacher_model:
            metadata["teacher_model"] = teacher_model
            metadata_changed = True
        if metadata.get("teacher_revision") != teacher_revision:
            metadata["teacher_revision"] = teacher_revision
            metadata_changed = True
        if metadata.get("benchmark_name") != benchmark_name:
            metadata["benchmark_name"] = benchmark_name
            metadata_changed = True
    return public, content_changed, metadata_changed


def replace_restricted(text: str, counts: Counter[str]) -> tuple[str, bool]:
    changed = False
    for kind, pattern in PATTERNS.items():
        def replacement(match: re.Match[str], *, pattern_kind: str = kind) -> str:
            nonlocal changed
            changed = True
            counts[pattern_kind] += 1
            digest = hashlib.sha256(f"{pattern_kind}\0{match.group(0)}".encode()).hexdigest()[:12]
            return f"<SYNTHETIC_{pattern_kind.upper()}_{digest}>"

        text = pattern.sub(replacement, text)
    return text, changed


def scan_file(path: Path) -> Counter[str]:
    counts: Counter[str] = Counter()
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            for kind, pattern in PATTERNS.items():
                counts[kind] += len(pattern.findall(line))
    return counts


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
