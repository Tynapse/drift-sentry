from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts import evaluate_drift_sentry_model as evaluator
from scripts import generate_semantic_synthetic_6tier as generator


class LabelIntegrityTest(unittest.TestCase):
    def test_generation_retries_then_accepts_or_excludes(self) -> None:
        target = {
            "sample_index": 0,
            "split": "train",
            "target_tier": "goal",
            "seed": 42,
            "archetype": generator.build_archetype_catalog(["goal"])[0],
        }
        for tiers in (("goal",), ("reward", "goal"), ("reward",) * 3, (None,) * 3, ("invalid",) * 3):
            replies = [
                json.dumps(
                    {
                        **{field: f"example {index}: {field}" for field in generator.CASE_FIELDS},
                        "label": {} if tier is None else {"tier": tier},
                    }
                )
                for index, tier in enumerate(tiers)
            ]
            with self.subTest(tiers=tiers), patch.object(generator.time, "sleep"):
                with patch.object(generator, "call_teacher", side_effect=replies) as teacher:
                    result = generator.generate_one(
                        target=target,
                        taxonomy_brief="test taxonomy",
                        base_url="http://localhost/v1",
                        model="test",
                        api_key="test",
                        timeout=1,
                        max_retries=0,
                        max_tokens=512,
                        temperature=0,
                        enable_thinking=False,
                        response_format_json=True,
                        case_retries=2,
                        labels=evaluator.TIERS,
                        include_private_metadata=True,
                    )
                self.assertEqual(teacher.call_count, len(tiers))
                self.assertEqual(result["ok"], tiers[-1] == "goal")
                if result["ok"]:
                    record = result["record"]
                    self.assertEqual(record["metadata"]["raw_teacher_text"], replies[-1])
                    self.assertEqual(json.loads(record["messages"][-1]["content"])["tier"], "goal")
                else:
                    self.assertNotIn("record", result)
                    expected = "teacher_tier_mismatch" if tiers[-1] == "reward" else "teacher_tier_missing_or_invalid"
                    self.assertEqual(result["failure_reason"], expected)

    def test_evaluation_requires_assistant_gold(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "test.jsonl"
            for role, label in (
                ("assistant", {"tier": "normal"}),
                ("assistant", {}),
                ("assistant", {"tier": "unknown"}),
                ("user", {"tier": "normal"}),
            ):
                with self.subTest(role=role, label=label):
                    path.write_text(
                        json.dumps(
                            {
                                "id": "case",
                                "metadata": {"target_tier": "reward"},
                                "messages": [{"role": role, "content": json.dumps(label)}],
                            }
                        )
                        + "\n",
                        encoding="utf-8",
                    )
                    if role == "assistant" and label.get("tier") == "normal":
                        self.assertEqual(evaluator.load_rows(path, 0)[0]["gold_tier"], "normal")
                    else:
                        with self.assertRaises(ValueError):
                            evaluator.load_rows(path, 0)


if __name__ == "__main__":
    unittest.main()
