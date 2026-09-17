"""Execute the hosted acquisition function with synthetic, offline GitHub data."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
import zipfile

import yaml

from shadow.contracts import seal_bundle, verify_bundle


class SealedAcquisitionTests(unittest.TestCase):
    def _acquire(self, *, digest_matches):
        repository = Path(__file__).resolve().parents[1]
        workflow = yaml.safe_load(
            (repository / ".github/workflows/news-shadow.yml").read_text()
        )
        steps = workflow["jobs"]["metadata-acquisition"]["steps"]
        shell = next(step["run"] for step in steps if step.get("id") == "meta")
        start = shell.index("try_sealed_bundle() {")
        function = shell[start:shell.index("\nif [[", start)]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            bundle = root / "original"
            manifest = seal_bundle(bundle, {
                "source": {"run_id": 123, "run_attempt": 1},
                "report_date": "2026-09-17",
                "capabilities": {"filter_replay": False},
            })
            archive = root / "bundle.zip"
            with zipfile.ZipFile(archive, "w") as zipped:
                zipped.write(bundle / "manifest.json", "manifest.json")
            digest = hashlib.sha256(archive.read_bytes()).hexdigest()
            (root / "artifacts.json").write_text(json.dumps({"artifacts": [{
                "id": 456, "name": "news-shadow-bundle-1", "expired": False,
                "created_at": "2026-09-17T08:00:00Z",
                "size_in_bytes": archive.stat().st_size,
                "digest": "sha256:" + (digest if digest_matches else "0" * 64),
            }]}))
            (root / "attempt.json").write_text(json.dumps({
                "run_attempt": 1, "run_started_at": "2026-09-17T07:00:00Z",
                "updated_at": "2026-09-17T09:00:00Z",
            }))
            (root / "news-shadow-metadata").mkdir()
            # Match the workflow's conditional call exactly: Bash disables
            # errexit inside functions invoked by `if !`, even with set -e.
            stub = '''gh() {
  case "$*" in
    *"/artifacts?per_page=100"*) cat "$FIXTURE/artifacts.json" ;;
    *"/attempts/1"*) cat "$FIXTURE/attempt.json" ;;
    *"/456/zip"*) cat "$FIXTURE/bundle.zip" ;;
    *) return 2 ;;
  esac
}
'''
            script = "set -euo pipefail\n" + stub + function + '''
if ! try_sealed_bundle 123 1 "$FIXTURE/output"; then
  echo rejected
else
  echo accepted
fi
'''
            result = subprocess.run(
                ["bash", "-c", script], cwd=repository,
                env={**os.environ, "RUNNER_TEMP": str(root), "FIXTURE": str(root)},
                capture_output=True, text=True, timeout=30,
            )
            output = root / "output"
            if digest_matches:
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn("accepted", result.stdout)
                self.assertEqual(
                    verify_bundle(output, capability=None)["bundle_sha256"],
                    manifest["bundle_sha256"],
                )
            else:
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn("rejected", result.stdout)
                self.assertIn("digest mismatch", result.stderr)
                self.assertFalse(output.exists(), "Invalid archive reached extraction")

    def test_mismatched_digest_is_rejected_before_extraction(self):
        self._acquire(digest_matches=False)

    def test_valid_digest_reaches_verified_extraction(self):
        self._acquire(digest_matches=True)


if __name__ == "__main__":
    unittest.main()
