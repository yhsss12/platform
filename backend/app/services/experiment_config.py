from __future__ import annotations

import threading
from pathlib import Path
from typing import Dict, Tuple

import yaml

from app.schemas.experiment import (
    ExperimentMethodConfig,
    ExperimentMethodResponse,
    ExperimentMethodUpdateRequest,
)

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_CONFIG_PATH = _PROJECT_ROOT / "config" / "experiment_method.yaml"

_DEFAULT_METHODS: Dict[str, ExperimentMethodConfig] = {
    "proposed": ExperimentMethodConfig(
        name="proposed",
        method_code="P",
        description="Complete proposed method with decoupling, dual-path preview, and explicit recovery.",
        decoupling=True,
        dual_path_preview=True,
        recovery=True,
        preview_route="split",
        relay_mode="minimal",
        preview_mode_lock="auto",
        browser_recovery_enabled=True,
    ),
    "baseline_b1": ExperimentMethodConfig(
        name="baseline_b1",
        method_code="B1",
        description="Relay-coupled baseline with MJPEG-only preview through the platform relay path.",
        decoupling=False,
        dual_path_preview=False,
        recovery=False,
        preview_route="relay",
        relay_mode="coupled",
        preview_mode_lock="mjpeg",
        browser_recovery_enabled=False,
    ),
    "baseline_b2": ExperimentMethodConfig(
        name="baseline_b2",
        method_code="B2",
        description="Single-path preview baseline using WebRTC only without fallback.",
        decoupling=True,
        dual_path_preview=False,
        recovery=False,
        preview_route="single_path",
        relay_mode="minimal",
        preview_mode_lock="webrtc",
        browser_recovery_enabled=False,
    ),
    "baseline_b3": ExperimentMethodConfig(
        name="baseline_b3",
        method_code="B3",
        description="Dual-path baseline without explicit browser-side recovery.",
        decoupling=True,
        dual_path_preview=True,
        recovery=False,
        preview_route="split",
        relay_mode="minimal",
        preview_mode_lock="auto",
        browser_recovery_enabled=False,
    ),
}

_ALIASES = {
    "p": "proposed",
    "proposed": "proposed",
    "b1": "baseline_b1",
    "baseline_b1": "baseline_b1",
    "relay-coupled": "baseline_b1",
    "b2": "baseline_b2",
    "baseline_b2": "baseline_b2",
    "single-path": "baseline_b2",
    "b3": "baseline_b3",
    "baseline_b3": "baseline_b3",
    "no-recovery": "baseline_b3",
}


def _canonical_name(name: str) -> str:
    key = (name or "").strip().lower()
    return _ALIASES.get(key, "proposed")


class ExperimentConfigService:
    def __init__(self, path: Path = _CONFIG_PATH) -> None:
        self._path = path
        self._lock = threading.Lock()
        self._cached_mtime_ns: int | None = None
        self._cached_model: ExperimentMethodConfig | None = None

    def _materialize(self, name: str, overrides: Dict[str, object] | None = None) -> ExperimentMethodConfig:
        canonical = _canonical_name(name)
        base = _DEFAULT_METHODS[canonical].model_dump()
        for key, value in (overrides or {}).items():
            if value is None or key == "name":
                continue
            if key in base:
                base[key] = value
        base["name"] = canonical
        if "browser_recovery_enabled" not in (overrides or {}):
            base["browser_recovery_enabled"] = bool(base.get("recovery", False))
        return ExperimentMethodConfig.model_validate(base)

    def _read_file(self) -> Tuple[ExperimentMethodConfig, str]:
        if not self._path.exists():
            model = _DEFAULT_METHODS["proposed"]
            return model, "default"

        raw = yaml.safe_load(self._path.read_text(encoding="utf-8")) or {}
        section = raw.get("experiment_method") if isinstance(raw, dict) else raw
        if not isinstance(section, dict):
            model = _DEFAULT_METHODS["proposed"]
            return model, "default"
        name = str(section.get("name") or "proposed")
        model = self._materialize(name, section)
        return model, str(self._path)

    def load(self) -> ExperimentMethodResponse:
        with self._lock:
            try:
                st = self._path.stat()
                if (
                    self._cached_model is not None
                    and self._cached_mtime_ns is not None
                    and st.st_mtime_ns == self._cached_mtime_ns
                ):
                    return ExperimentMethodResponse(
                        experiment_method=self._cached_model,
                        source=str(self._path),
                    )
            except FileNotFoundError:
                pass

            model, source = self._read_file()
            if source == "default":
                self._path.parent.mkdir(parents=True, exist_ok=True)
                body = {"experiment_method": model.model_dump()}
                self._path.write_text(
                    yaml.safe_dump(body, allow_unicode=True, sort_keys=False),
                    encoding="utf-8",
                )
                source = str(self._path)
            try:
                self._cached_mtime_ns = self._path.stat().st_mtime_ns
            except FileNotFoundError:
                self._cached_mtime_ns = None
            self._cached_model = model
            return ExperimentMethodResponse(experiment_method=model, source=source)

    def save(self, payload: ExperimentMethodUpdateRequest) -> ExperimentMethodResponse:
        overrides = payload.model_dump(exclude_none=True)
        model = self._materialize(payload.name, overrides)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        body = {"experiment_method": model.model_dump()}
        self._path.write_text(
            yaml.safe_dump(body, allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )
        with self._lock:
            self._cached_model = model
            try:
                self._cached_mtime_ns = self._path.stat().st_mtime_ns
            except FileNotFoundError:
                self._cached_mtime_ns = None
        return ExperimentMethodResponse(experiment_method=model, source=str(self._path))


_experiment_config_service = ExperimentConfigService()


def get_experiment_config_service() -> ExperimentConfigService:
    return _experiment_config_service
