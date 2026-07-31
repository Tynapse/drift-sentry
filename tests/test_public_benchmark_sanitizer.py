from __future__ import annotations

import importlib.util
import unittest
from collections import Counter
from pathlib import Path

SCRIPT = Path(__file__).parents[1] / "scripts" / "sanitize_drift_sentry_benchmark.py"
SPEC = importlib.util.spec_from_file_location("benchmark_sanitizer", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)

FINALIZER_SCRIPT = Path(__file__).parents[1] / "scripts" / "finalize_drift_sentry_benchmark.py"
FINALIZER_SPEC = importlib.util.spec_from_file_location("benchmark_finalizer", FINALIZER_SCRIPT)
assert FINALIZER_SPEC and FINALIZER_SPEC.loader
FINALIZER = importlib.util.module_from_spec(FINALIZER_SPEC)
FINALIZER_SPEC.loader.exec_module(FINALIZER)


class PublicBenchmarkSanitizerTest(unittest.TestCase):
    def test_replaces_hyphen_and_underscore_secret_tokens(self) -> None:
        counts: Counter[str] = Counter()
        hyphen_secret = "sk-" + "abcdefghijklmnop"
        underscore_secret = "hf_" + "abcdefghijklmnop"
        text, changed = MODULE.replace_restricted(
            f"use {hyphen_secret} and {underscore_secret}",
            counts,
        )
        self.assertTrue(changed)
        self.assertNotIn(hyphen_secret, text)
        self.assertNotIn(underscore_secret, text)
        self.assertEqual(counts["secret_token"], 2)
        self.assertFalse(any(pattern.search(text) for pattern in MODULE.PATTERNS.values()))

    def test_placeholder_is_deterministic(self) -> None:
        secret = "sk-" + "abcdefghijklmnop"
        first, _ = MODULE.replace_restricted(secret, Counter())
        second, _ = MODULE.replace_restricted(secret, Counter())
        self.assertEqual(first, second)

    def test_finalizer_rejects_hyphen_secret_token(self) -> None:
        self.assertEqual(FINALIZER.find_pii("sk-" + "abcdefghijklmnop"), "secret_token")
        self.assertEqual(FINALIZER.find_pii("sk-" + "abcdefghijklmno-"), "secret_token")

    def test_transform_normalizes_public_provenance(self) -> None:
        row = {
            "messages": [{"role": "user", "content": "safe text"}],
            "metadata": {
                "benchmark_name": "legacy-name",
                "teacher_model": "legacy-model",
                "teacher_revision": "legacy-revision",
            },
        }
        transformed, content_changed, metadata_changed = MODULE.transform_row(
            row,
            Counter(),
            teacher_model="Qwen/Qwen3.6-27B",
            teacher_revision="a" * 40,
            benchmark_name="drift-sentry-bench-50k-v1",
        )
        self.assertFalse(content_changed)
        self.assertTrue(metadata_changed)
        self.assertEqual(transformed["metadata"]["benchmark_name"], "drift-sentry-bench-50k-v1")
        self.assertEqual(transformed["metadata"]["teacher_revision"], "a" * 40)


if __name__ == "__main__":
    unittest.main()
