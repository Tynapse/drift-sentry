from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
import json
from pathlib import Path
from typing import Any


def load_conversation_rows(
    *,
    data_path: str | Path | None = None,
    repo_id: str | None = None,
    config_name: str | None = None,
    split: str | None = None,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    """Load OpenAI-style conversation rows from JSONL or Hugging Face Datasets."""

    if data_path:
        rows = _load_jsonl(Path(data_path))
    else:
        if not repo_id or not split:
            raise ValueError("data_path or both repo_id and split are required")
        try:
            from datasets import load_dataset  # type: ignore[import-untyped]
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError("datasets is required for Hugging Face dataset loading") from exc
        rows = [dict(row) for row in load_dataset(repo_id, config_name, split=split)]
    if limit is not None and limit > 0:
        return rows[:limit]
    return rows


def rows_to_conversations(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [row_to_conversation(row) for row in rows]


def row_to_conversation(row: Mapping[str, Any]) -> dict[str, Any]:
    messages_value = row.get("messages")
    if isinstance(messages_value, str):
        messages_value = json.loads(messages_value)
    if not isinstance(messages_value, list):
        raise ValueError("conversation row must contain a messages list")
    messages = [_normalize_message(message) for message in messages_value]
    if len(messages) < 2 or messages[-1]["role"] != "assistant":
        raise ValueError("conversation must end with a gold assistant message")
    if not any(message["role"] == "user" for message in messages[:-1]):
        raise ValueError("conversation prompt must contain a user message")
    result: dict[str, Any] = {
        "id": str(row.get("id") or ""),
        "messages": messages,
    }
    if isinstance(row.get("metadata"), Mapping):
        result["metadata"] = dict(row["metadata"])
    return result


def prompt_messages(messages: Sequence[Mapping[str, Any]]) -> list[dict[str, str]]:
    normalized = [_normalize_message(message) for message in messages]
    if not normalized or normalized[-1]["role"] != "assistant":
        raise ValueError("conversation must end with a gold assistant message")
    return normalized[:-1]


def _normalize_message(message: Any) -> dict[str, str]:
    if not isinstance(message, Mapping):
        raise ValueError("each message must be an object")
    role = str(message.get("role") or "").strip()
    content = message.get("content")
    if role not in {"system", "user", "assistant"}:
        raise ValueError(f"unsupported message role: {role or '<empty>'}")
    if not isinstance(content, str):
        raise ValueError("message content must be a string")
    return {"role": role, "content": content}


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number} JSONL row must be an object")
            rows.append(row)
    return rows
