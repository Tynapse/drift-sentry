from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


def byteify(value: Any) -> Any:
    """Convert bytes nested inside JSON-like data into UTF-8 strings."""
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, Mapping):
        return {byteify(key): byteify(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, str):
        return [byteify(item) for item in value]
    return value
