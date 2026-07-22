from __future__ import annotations

from pathlib import Path

import pytest

from app.services.isaac_lab import video_compat as vc


def test_probe_mp4_codec_detects_mp4v(tmp_path: Path):
    path = tmp_path / "preview.mp4"
    path.write_bytes(b"\x00" * 40 + b"mdat" + b"\x00" * 100 + b"mp4v" + b"\x00" * 20)
    probe = vc.probe_mp4_codec(path)
    assert probe["codec"] == "mp4v"
    assert probe["browserCompatible"] is False


def test_probe_mp4_codec_detects_h264(tmp_path: Path):
    path = tmp_path / "preview.mp4"
    path.write_bytes(b"ftyp" + b"\x00" * 20 + b"moov" + b"\x00" * 10 + b"avc1" + b"mdat")
    probe = vc.probe_mp4_codec(path)
    assert probe["codec"] == "h264"
    assert probe["browserCompatible"] is True


def test_ensure_browser_playable_returns_source_when_compatible(tmp_path: Path):
    path = tmp_path / "preview.mp4"
    path.write_bytes(b"ftyp" + b"\x00" * 20 + b"moov" + b"\x00" * 10 + b"avc1" + b"mdat")
    playable, note = vc.ensure_browser_playable_mp4(path)
    assert playable == path
    assert note is None


def test_ensure_browser_playable_transcodes_mp4v(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    source = tmp_path / "preview.mp4"
    source.write_bytes(b"\x00" * 40 + b"mdat" + b"\x00" * 100 + b"mp4v")

    dest = vc.browser_cached_video_path(source)

    def fake_transcode(src: Path, out: Path, *, timeout: int = 600):
        assert src == source
        out.write_bytes(b"ftyp" + b"\x00" * 20 + b"moov" + b"avc1" + b"mdat")
        return True, None

    monkeypatch.setattr(vc, "transcode_to_browser_mp4", fake_transcode)

    playable, note = vc.ensure_browser_playable_mp4(source)
    assert playable == dest
    assert note == "transcoded"
