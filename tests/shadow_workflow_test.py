"""Exercise the actual workflow's state restore without GitHub or credentials."""
import base64
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from urllib.error import HTTPError

import yaml


class StateRestoreTests(unittest.TestCase):
    def _restore(self, response_for):
        path = Path(__file__).resolve().parents[1] / ".github/workflows/news-shadow.yml"
        workflow = yaml.safe_load(path.read_text())
        steps = workflow["jobs"]["metadata-acquisition"]["steps"]
        shell = next(step["run"] for step in steps if step.get("id") == "meta")
        marker = 'python3 - "$RUNNER_TEMP/news-shadow-metadata/discovery-state" <<\'PY\'\n'
        source = shell.split(marker, 1)[1].split("\nPY", 1)[0]

        def open_request(request, timeout):
            value = response_for(request.full_url)
            if isinstance(value, int):
                raise HTTPError(request.full_url, value, "synthetic failure", {}, None)
            return io.StringIO(json.dumps(value))

        with tempfile.TemporaryDirectory() as directory, \
                patch.dict(os.environ, {"SHADOW_INTERNAL_REPOSITORY": "owner/repo",
                    "INTERNAL_GITHUB_TOKEN": "synthetic", "GITHUB_SHA": "a" * 40}, clear=True), \
                patch("sys.argv", ["-", directory]), \
                patch("urllib.request.urlopen", side_effect=open_request):
            exec(compile(source, str(path), "exec"), {})
            return {p.name: p.read_text() for p in Path(directory).iterdir()}

    def test_only_verified_first_run_absence_starts_empty(self):
        self.assertEqual(self._restore(lambda url: {} if "/commits/" in url else 404), {})

    def test_contents_access_failure_cannot_be_misread_as_first_run(self):
        with self.assertRaisesRegex(SystemExit, "HTTP 404"):
            self._restore(lambda url: 404)

    def test_existing_branch_state_read_failures_are_fatal(self):
        for status in (403, 404, 429, 500):
            with self.subTest(status=status), self.assertRaisesRegex(SystemExit, f"HTTP {status}"):
                self._restore(lambda url: status if "/contents/" in url else {})

    def test_existing_state_bytes_are_restored(self):
        index = json.dumps({"schema_version": "news-shadow-index/v1", "experiments": {}})
        events = json.dumps({"type": "discovery"}) + "\n"
        def respond(url):
            if "/contents/" not in url:
                return {}
            raw = index if "index.json" in url else events
            return {"content": base64.b64encode(raw.encode()).decode()}
        self.assertEqual(self._restore(respond), {"index.json": index, "events.jsonl": events})

    def test_invalid_state_is_not_silently_replaced(self):
        with self.assertRaises(ValueError):
            self._restore(lambda url: {"content": "invalid"} if "/contents/" in url else {})


if __name__ == "__main__":
    unittest.main()
