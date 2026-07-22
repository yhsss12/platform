import io
import os
import sys
import tarfile
import tempfile
from pathlib import Path


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def _read_ar_members(path: Path) -> dict[str, bytes]:
    data = path.read_bytes()
    assert data.startswith(b"!<arch>\n")
    pos = 8
    out: dict[str, bytes] = {}
    while pos + 60 <= len(data):
        hdr = data[pos : pos + 60]
        pos += 60
        name = hdr[0:16].decode("utf-8", "replace").strip()
        if name.endswith("/"):
            name = name[:-1]
        size = int(hdr[48:58].decode("ascii", "ignore").strip() or "0")
        body = data[pos : pos + size]
        pos += size
        if size % 2 == 1:
            pos += 1
        if name:
            out[name] = body
    return out


def test_build_deb_from_agent_tarball_minimal():
    from app.services.agent_deb_builder import build_deb_from_agent_tarball

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        agent_dir = tmp / "agent"
        agent_dir.mkdir()
        (agent_dir / "requirements.txt").write_text("", encoding="utf-8")
        (agent_dir / "agent_main.py").write_text("print('hello')\n", encoding="utf-8")

        tar_path = tmp / "agent-linux-x86_64-0.1.0.tar.gz"
        with tarfile.open(tar_path, mode="w:gz") as tf:
            tf.add(agent_dir, arcname="agent")

        deb_path = tmp / "eai-agent_0.1.0_x86_64.deb"
        res = build_deb_from_agent_tarball(
            tar_gz_path=str(tar_path),
            output_deb_path=str(deb_path),
            version="0.1.0",
            arch="x86_64",
        )

        assert Path(res.file_path).is_file()
        members = _read_ar_members(deb_path)
        assert "debian-binary" in members
        assert members["debian-binary"] == b"2.0\n"
        assert "control.tar.gz" in members
        assert "data.tar.gz" in members

        with tarfile.open(fileobj=io.BytesIO(members["control.tar.gz"]), mode="r:gz") as tfc:
            control = tfc.extractfile("control")
            assert control is not None
            txt = control.read().decode("utf-8", "replace")
            assert "Package: eai-agent" in txt
            assert "Version: 0.1.0" in txt

        with tarfile.open(fileobj=io.BytesIO(members["data.tar.gz"]), mode="r:gz") as tfd:
            names = set(tfd.getnames())
            assert "./opt/eai-agent/agent_main.py" in names
            assert "./usr/bin/eai-agent-run" in names
            assert "./lib/systemd/system/eai-agent.service" in names
