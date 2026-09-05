"""Runtime guards tested without loading AI models or touching the GPU."""
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import generate
from runtime_lock import generation_lock


class RuntimeTests(unittest.TestCase):
    def test_missing_timing_runtime_fails_before_model_load(self):
        with tempfile.TemporaryDirectory() as folder:
            text = Path(folder) / "script.txt"
            text.write_text("Hello world.", encoding="utf-8")
            with patch.object(sys, "argv", ["generate.py", "--timing", "--text-file", str(text)]), \
                 patch.object(generate.shutil, "which", return_value=None):
                with self.assertRaisesRegex(RuntimeError, "uv is required"):
                    generate.main()
            self.assertEqual(list(Path(folder).iterdir()), [text])

    def test_cross_process_lock_rejects_concurrency_and_releases(self):
        with tempfile.TemporaryDirectory() as folder:
            lock = Path(folder) / "generation.lock"
            command = [sys.executable, "-c",
                       "from pathlib import Path\nfrom runtime_lock import generation_lock\nimport sys\nwith generation_lock(Path(sys.argv[1])):\n print('acquired')",
                       str(lock)]
            with generation_lock(lock):
                result = subprocess.run(command, capture_output=True, text=True)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("already busy", result.stderr)
            result = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("acquired", result.stdout)


if __name__ == "__main__":
    unittest.main()
