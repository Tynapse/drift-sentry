from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from drift_sentry.data.conversations import load_conversation_rows, prompt_messages, row_to_conversation


class ConversationDataTest(unittest.TestCase):
    def test_loads_and_normalizes_jsonl(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "train.jsonl"
            path.write_text(
                json.dumps(
                    {
                        "id": "row-1",
                        "messages": [
                            {"role": "system", "content": "policy"},
                            {"role": "user", "content": "candidate response"},
                            {"role": "assistant", "content": '{"tier":"normal"}'},
                        ],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            rows = load_conversation_rows(data_path=path)
            conversation = row_to_conversation(rows[0])
            self.assertEqual(conversation["id"], "row-1")
            self.assertEqual(len(prompt_messages(conversation["messages"])), 2)

    def test_rejects_missing_gold_assistant_message(self) -> None:
        with self.assertRaisesRegex(ValueError, "gold assistant"):
            row_to_conversation({"messages": [{"role": "user", "content": "candidate"}]})


if __name__ == "__main__":
    unittest.main()
