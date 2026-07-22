"""Tests for safe binary/metadata file handling in resource center paths."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

from app.services.model_asset_checkpoint_resolver import _load_checkpoint_payload  # noqa: E402
from app.services.safe_file_io import (  # noqa: E402
    is_probably_binary,
    safe_read_json,
    safe_read_text,
)


def test_is_probably_binary_for_checkpoint_and_parquet(tmp_path: Path):
    assert is_probably_binary(tmp_path / "model_final.pt") is True
    assert is_probably_binary(tmp_path / "dataset.npz") is True
    assert is_probably_binary(tmp_path / "file-000.parquet") is True
    assert is_probably_binary(tmp_path / "train_config.json") is False


def test_safe_read_json_skips_binary_pt(tmp_path: Path):
    binary_pt = tmp_path / "model_final.pt"
    binary_pt.write_bytes(b"\x80" * 128)
    assert safe_read_json(binary_pt) is None
    assert safe_read_text(binary_pt) is None


def test_load_checkpoint_payload_reads_json_smoke_pt(tmp_path: Path):
    smoke_pt = tmp_path / "model_final.pt"
    smoke_pt.write_text(
        json.dumps(
            {
                "format": "pi0_lerobot_smoke_v1",
                "backend": "pi0",
                "state_dim": 9,
                "action_dim": 8,
            }
        ),
        encoding="utf-8",
    )
    payload = _load_checkpoint_payload(smoke_pt)
    assert payload.get("backend") == "pi0"
    assert payload.get("state_dim") == 9


def test_load_checkpoint_payload_does_not_raise_on_binary_torch_pt(tmp_path: Path):
    binary_pt = tmp_path / "model_final.pt"
    binary_pt.write_bytes(b"\x80" * 128)
    payload = _load_checkpoint_payload(binary_pt)
    assert isinstance(payload, dict)


def test_safe_read_json_reads_manifest(tmp_path: Path):
    manifest = tmp_path / "model_manifest.json"
    manifest.write_text(json.dumps({"modelType": "pi0", "actionDim": 8}), encoding="utf-8")
    data = safe_read_json(manifest)
    assert data is not None
    assert data.get("modelType") == "pi0"
