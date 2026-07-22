"""Canonical training backend / model-type identifiers for validation."""

from __future__ import annotations

from typing import Any, Optional

# Maps canonical backend -> accepted raw aliases (lowercase, normalized).
_BACKEND_ALIASES: dict[str, frozenset[str]] = {
    "diffusion_policy": frozenset(
        {
            "diffusion_policy",
            "diffusion",
            "diffusionpolicy",
            "diffusion_policy_adapter",
            "diffusion-policy",
            "dp",
        }
    ),
    "robomimic_bc": frozenset(
        {
            "robomimic_bc",
            "robomimic",
            "robomimicbc",
            "bc",
            "robomimic_bc_adapter",
        }
    ),
    "isaac_robomimic_bc": frozenset(
        {
            "isaac_robomimic_bc",
            "isaac_robomimic",
            "isaacrobomimicbc",
        }
    ),
    "torch_bc": frozenset(
        {
            "torch_bc",
            "torchbc",
            "bc_pytorch",
            "bc_torch",
        }
    ),
    "act": frozenset({"act", "act_adapter"}),
    "pi0": frozenset({"pi0", "openpi", "pi_0"}),
}

_ALIAS_TO_CANONICAL: dict[str, str] = {}
for canonical, aliases in _BACKEND_ALIASES.items():
    _ALIAS_TO_CANONICAL[canonical] = canonical
    for alias in aliases:
        _ALIAS_TO_CANONICAL[alias] = canonical

# Human-readable labels -> canonical
_DISPLAY_LABEL_ALIASES: dict[str, str] = {
    "diffusion policy": "diffusion_policy",
    "robomimic bc": "robomimic_bc",
    "isaac robomimic bc": "isaac_robomimic_bc",
    "bc (pytorch)": "torch_bc",
    "bc pytorch": "torch_bc",
}


def _normalize_backend_token(value: str) -> str:
    text = str(value or "").strip().lower()
    if not text:
        return ""
    text = text.replace("-", "_")
    text = "_".join(text.split())
    return text


def canonicalize_training_backend(value: Optional[str]) -> str:
    """Normalize framework/modelType/trainingBackend strings to a canonical backend id."""
    raw = str(value or "").strip()
    if not raw:
        return ""

    display_key = raw.lower().strip()
    if display_key in _DISPLAY_LABEL_ALIASES:
        return _DISPLAY_LABEL_ALIASES[display_key]

    token = _normalize_backend_token(raw)
    if token in _ALIAS_TO_CANONICAL:
        return _ALIAS_TO_CANONICAL[token]

    compact = token.replace("_", "")
    if compact in _ALIAS_TO_CANONICAL:
        return _ALIAS_TO_CANONICAL[compact]

    if "diffusion" in token or token == "dp":
        return "diffusion_policy"
    if token.startswith("robomimic"):
        return "robomimic_bc"
    if token.startswith("isaac") and "robomimic" in token:
        return "isaac_robomimic_bc"
    if token in {"act"}:
        return "act"
    if token in {"pi0", "openpi"}:
        return "pi0"
    if "torch" in token and "bc" in token:
        return "torch_bc"

    return token


def training_backends_compatible(left: Optional[str], right: Optional[str]) -> bool:
    canonical_left = canonicalize_training_backend(left)
    canonical_right = canonicalize_training_backend(right)
    if not canonical_left or not canonical_right:
        return False
    return canonical_left == canonical_right


def resolve_asset_training_backend(asset: dict[str, Any]) -> str:
    for key in (
        "backendType",
        "trainingBackend",
        "framework",
        "modelType",
        "baseAlgorithm",
        "modelTypeId",
        "adapterId",
    ):
        canonical = canonicalize_training_backend(str(asset.get(key) or ""))
        if canonical:
            return canonical
    return ""
