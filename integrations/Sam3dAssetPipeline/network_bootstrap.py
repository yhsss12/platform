"""HF mirror + GitHub URL rewrite applied before each pipeline stage."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Callable

PIPELINE_DIR = Path(__file__).resolve().parent
PIPELINE_GITCONFIG = PIPELINE_DIR / ".gitconfig.pipeline"
DEFAULT_HF_ENDPOINT = "https://hf-mirror.com"
GITHUB_INSTEADOF_KEY = "url.git@github.com:.insteadOf"
GITHUB_INSTEADOF_VALUE = "https://github.com/"


def _write_pipeline_gitconfig() -> Path:
    if not PIPELINE_GITCONFIG.is_file():
        PIPELINE_GITCONFIG.write_text(
            '[url "git@github.com:"]\n\tinsteadOf = https://github.com/\n',
            encoding="utf-8",
        )
    return PIPELINE_GITCONFIG.resolve()


def apply_network_bootstrap(
    *,
    hf_endpoint: str | None = None,
    enable_git_github_ssh_rewrite: bool = True,
    use_global_git_config: bool = False,
    log_fn: Callable[[str], None] | None = None,
) -> dict[str, str]:
    """Equivalent to exporting HF_ENDPOINT and git url.insteadOf before segment/reconstruct."""
    endpoint = (
        hf_endpoint
        or os.environ.get("SAM3D_HF_ENDPOINT")
        or os.environ.get("HF_ENDPOINT")
        or DEFAULT_HF_ENDPOINT
    ).strip()
    os.environ["HF_ENDPOINT"] = endpoint
    os.environ.setdefault("SAM3D_HF_ENDPOINT", endpoint)

    applied: dict[str, str] = {"HF_ENDPOINT": endpoint}

    if enable_git_github_ssh_rewrite:
        if use_global_git_config:
            subprocess.run(
                ["git", "config", "--global", GITHUB_INSTEADOF_KEY, GITHUB_INSTEADOF_VALUE],
                check=False,
                capture_output=True,
                text=True,
            )
            if log_fn:
                log_fn("[network_bootstrap] git config --global url.git@github.com:.insteadOf https://github.com/")
        else:
            gitconfig = _write_pipeline_gitconfig()
            os.environ["GIT_CONFIG_GLOBAL"] = str(gitconfig)
            applied["GIT_CONFIG_GLOBAL"] = str(gitconfig)
            if log_fn:
                log_fn(f"[network_bootstrap] GIT_CONFIG_GLOBAL={gitconfig}")

    if log_fn:
        log_fn(f"[network_bootstrap] HF_ENDPOINT={endpoint}")

    return applied


def subprocess_env(extra: dict[str, str] | None = None) -> dict[str, str]:
    apply_network_bootstrap()
    env = os.environ.copy()
    if extra:
        env.update(extra)
    return env
