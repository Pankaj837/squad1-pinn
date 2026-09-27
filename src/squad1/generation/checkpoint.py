"""Generator checkpoints that are safe to load (``torch.load(weights_only=True)``: tensors + plain values only)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from squad1.errors import ContractError
from squad1.generation.diffusion import DiffusionScheduler
from squad1.generation.dit import DiT

FORMAT = "squad1_generator_v1"


def save_generator(
    path: str | Path, model: DiT, scheduler: DiffusionScheduler, meta: dict[str, Any] | None = None
) -> Path:
    """Write ``{format, config, timesteps, state, meta}`` (``meta``: only str/int/float/bool/list/dict values)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format": FORMAT,
            "config": dict(model.config),
            "timesteps": scheduler.timesteps,
            "state": model.state_dict(),
            "meta": dict(meta or {}),
        },
        p,
    )
    return p


def load_generator(
    path: str | Path, device: str | torch.device = "cpu"
) -> tuple[DiT, DiffusionScheduler, dict[str, Any]]:
    blob = torch.load(Path(path), map_location=device, weights_only=True)
    if not isinstance(blob, dict) or blob.get("format") != FORMAT:
        raise ContractError(f"{path} is not a {FORMAT} checkpoint")
    model = DiT(**blob["config"])
    model.load_state_dict(blob["state"])
    model.to(device).eval()
    return model, DiffusionScheduler(int(blob["timesteps"])), dict(blob["meta"])
