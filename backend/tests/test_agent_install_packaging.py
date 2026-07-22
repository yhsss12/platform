import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.services.agent_package_manager import AgentPackageManager


class TestAgentInstallPackaging(unittest.TestCase):
    def test_package_manager_resolve_and_sha256(self):
        with tempfile.TemporaryDirectory() as td:
            tmp_path = Path(td)
            pkg = tmp_path / "agent.tar.gz"
            pkg.write_bytes(b"hello-agent")

            import hashlib

            sha = hashlib.sha256(pkg.read_bytes()).hexdigest()
            manifest_dir = tmp_path / "agent_packages"
            manifest_dir.mkdir()
            (manifest_dir / "agent.tar.gz").write_bytes(pkg.read_bytes())
            m = {
                "latest": "0.1.0",
                "packages": [
                    {
                        "version": "0.1.0",
                        "os": "linux",
                        "arch": "x86_64",
                        "path": "agent.tar.gz",
                        "sha256": sha,
                    }
                ],
            }
            (manifest_dir / "manifest.json").write_text(json.dumps(m), encoding="utf-8")

            mgr = AgentPackageManager(manifest_path=str(manifest_dir / "manifest.json"))
            ref = mgr.resolve(os_name="linux", arch="x86_64", version=None)
            self.assertEqual(ref.version, "0.1.0")
            self.assertEqual(ref.sha256, sha)
            self.assertTrue(Path(ref.file_path).exists())

    def test_package_manager_sha256_mismatch(self):
        with tempfile.TemporaryDirectory() as td:
            tmp_path = Path(td)
            manifest_dir = tmp_path / "agent_packages"
            manifest_dir.mkdir()
            (manifest_dir / "agent.tar.gz").write_bytes(b"hello-agent")
            m = {
                "latest": "0.1.0",
                "packages": [
                    {
                        "version": "0.1.0",
                        "os": "linux",
                        "arch": "x86_64",
                        "path": "agent.tar.gz",
                        "sha256": "0" * 64,
                    }
                ],
            }
            (manifest_dir / "manifest.json").write_text(json.dumps(m), encoding="utf-8")
            mgr = AgentPackageManager(manifest_path=str(manifest_dir / "manifest.json"))
            with self.assertRaises(ValueError):
                mgr.resolve(os_name="linux", arch="x86_64", version=None)


if __name__ == "__main__":
    unittest.main()
