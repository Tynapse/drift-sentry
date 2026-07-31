from __future__ import annotations

from pathlib import Path
import re
import unittest


ROOT = Path(__file__).parents[1]
TEXT_SUFFIXES = {".md", ".py", ".toml", ".yaml", ".yml"}


class PublicSurfaceTest(unittest.TestCase):
    def test_product_specific_prompt_modules_are_absent(self) -> None:
        forbidden = [
            ROOT / "drift_sentry" / "data" / ("kb" + "_judge.py"),
            ROOT / "drift_sentry" / "data" / "transforms",
        ]
        present = [
            str(path.relative_to(ROOT))
            for path in forbidden
            if path.is_file() or (path.is_dir() and any(path.rglob("*.py")))
        ]
        self.assertEqual(present, [])

    def test_no_private_paths_or_credential_literals(self) -> None:
        patterns = {
            "private_home": re.compile("/home/" + "ubuntu|/Users/" + "myungsub"),
            "aws_access_key": re.compile(r"(?<![A-Z0-9])(?:AKIA|ASIA)[A-Z0-9]{16}(?![A-Z0-9])"),
            "secret_token": re.compile(
                r"(?<![A-Za-z0-9])(?:s" + "k|h" + r"f)[_-][A-Za-z0-9_-]{16,}(?![A-Za-z0-9_-])"
            ),
            "private_key": re.compile("-----BEGIN " + "PRIVATE KEY-----"),
        }
        findings: list[str] = []
        for path in ROOT.rglob("*"):
            if not path.is_file() or path.suffix not in TEXT_SUFFIXES or any(part == ".git" for part in path.parts):
                continue
            text = path.read_text(encoding="utf-8")
            for name, pattern in patterns.items():
                if pattern.search(text):
                    findings.append(f"{path.relative_to(ROOT)}:{name}")
        self.assertEqual(findings, [])

    def test_documentation_uses_final_artifact_urls(self) -> None:
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        self.assertIn("https://huggingface.co/Tynapse/drift-sentry-4b-v1", readme)
        self.assertIn("https://huggingface.co/datasets/Tynapse/drift-sentry-bench-50k-v1", readme)
        self.assertIn("https://github.com/Tynapse/drift-sentry", readme)


if __name__ == "__main__":
    unittest.main()
