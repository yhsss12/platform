"""Tests for partial failure handling in resource overview API."""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

from app.services.resource_definition_service import get_resource_overview_counts  # noqa: E402


def test_resource_overview_survives_model_asset_scan_failure():
    with patch(
        "app.services.model_asset_db_service.list_model_assets_from_db",
        side_effect=UnicodeDecodeError("utf-8", b"\x80", 0, 1, "invalid start byte"),
    ):
        result = get_resource_overview_counts()

    assert result["modelAssets"] is None
    assert result["partialFailure"] is True
    assert any(w.get("category") == "modelAssets" for w in result.get("warnings") or [])
    assert isinstance(result.get("metrics"), int)
    assert isinstance(result.get("scenes"), int)


def test_resource_overview_returns_counts_when_model_assets_ok():
    with patch(
        "app.services.model_asset_db_service.list_model_assets_from_db",
        return_value=[{"id": "model__a"}, {"id": "model__b"}],
    ):
        result = get_resource_overview_counts()

    assert result["modelAssets"] == 2
    assert result.get("partialFailure") is False
