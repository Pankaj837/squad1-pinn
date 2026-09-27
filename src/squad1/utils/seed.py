"""Reproducibility helpers: one seeding entry point + provenance record for every output artefact."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import random
import subprocess
import sys
from dataclasses import asdict, is_dataclass
from typing import Any

import numpy as np
import torch


def seed_everything(seed: int, deterministic: bool = False) -> int:
    """Seed python, numpy and torch (CPU+CUDA); ``deterministic=True`` also requests deterministic torch kernels."""
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise ValueError(f"seed must be a non-negative int, got {seed!r}")
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():  # pragma: no cover - GPU only
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.benchmark = False
    return seed


def _jsonable(obj: Any) -> Any:
    if is_dataclass(obj) and not isinstance(obj, type):
        return _jsonable(asdict(obj))
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in sorted(obj.items(), key=lambda kv: str(kv[0]))}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    return obj


def config_hash(config: Any) -> str:
    """Stable short hash of any (dataclass / dict / list) configuration."""
    payload = json.dumps(_jsonable(config), sort_keys=True, default=str)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def _git_commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, timeout=3, check=False
        )
        return out.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def provenance(config: Any = None, seed: int | None = None) -> dict[str, Any]:
    """Record needed to reproduce a result: seed, config hash, library versions, device, git commit."""
    import scipy

    from squad1 import __version__

    return {
        "squad1_version": __version__,
        "seed": seed,
        "config_hash": config_hash(config) if config is not None else None,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "cuda": torch.version.cuda,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "git_commit": _git_commit(),
    }
