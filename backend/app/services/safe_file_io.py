"""Safe text/metadata file reads for resource scanning and asset enrichment."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

TEXT_EXTENSIONS = frozenset({".json", ".yaml", ".yml", ".toml", ".txt", ".csv", ".log", ".md"})
BINARY_EXTENSIONS = frozenset(
    {
        ".pt",
        ".pth",
        ".ckpt",
        ".npz",
        ".npy",
        ".hdf5",
        ".h5",
        ".parquet",
        ".mp4",
        ".png",
        ".jpg",
        ".jpeg",
        ".webp",
        ".gif",
        ".zip",
        ".tar",
        ".gz",
        ".bin",
        ".safetensors",
    }
)
METADATA_JSON_NAMES = frozenset(
    {
        "model_manifest.json",
        "train_config.json",
        "status.json",
        "dataset.manifest.json",
        "metadata.json",
        "stats.json",
        "generation_report.json",
        "model_assets_registry.json",
        "adapter_ready.json",
        "platform_eval_ready.json",
    }
)


def file_suffix(path: Path | str) -> str:
    return Path(path).suffix.lower()


def is_probably_binary(path: Path | str) -> bool:
    suffix = file_suffix(path)
    if suffix in BINARY_EXTENSIONS:
        return True
    if suffix in TEXT_EXTENSIONS:
        return False
    name = Path(path).name.lower()
    if name in METADATA_JSON_NAMES:
        return False
    if name.endswith(".json") or name.endswith(".yaml") or name.endswith(".yml"):
        return False
    return suffix not in TEXT_EXTENSIONS and suffix != ""


def is_metadata_file(path: Path | str) -> bool:
    suffix = file_suffix(path)
    if suffix in TEXT_EXTENSIONS:
        return True
    return Path(path).name.lower() in METADATA_JSON_NAMES


def safe_read_text(path: Path | str, *, errors: str = "strict") -> Optional[str]:
    candidate = Path(path)
    if not candidate.is_file():
        return None
    if is_probably_binary(candidate) and candidate.suffix.lower() not in {".pt", ".pth", ".ckpt"}:
        logger.warning("Skip binary file text read: %s", candidate)
        return None
    try:
        return candidate.read_text(encoding="utf-8", errors=errors)
    except UnicodeDecodeError as exc:
        logger.warning("Skip non-utf8 file: %s: %s", candidate, exc)
        return None
    except OSError as exc:
        logger.warning("Skip unreadable file: %s: %s", candidate, exc)
        return None


def safe_read_json(path: Path | str) -> Optional[dict[str, Any]]:
    candidate = Path(path)
    if not candidate.is_file():
        return None
    if candidate.suffix.lower() != ".json" and candidate.name.lower() not in METADATA_JSON_NAMES:
        logger.warning("Skip non-json metadata file: %s", candidate)
        return None
    text = safe_read_text(candidate)
    if not text:
        return None
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        logger.warning("Skip invalid json file: %s: %s", candidate, exc)
        return None
    return data if isinstance(data, dict) else None
