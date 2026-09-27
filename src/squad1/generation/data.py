"""Training-data generators that lie on the physics manifold (model-space output)."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from squad1.contracts.channels import ChannelNormalizer, get_domain
from squad1.errors import ContractError
from squad1.physics.darcy import DarcyBC, solve_pressure
from squad1.physics.domains import PHYSICS_CLASSES


def darcy_fields(n: int, size: int, seed: int = 0, contrast: float = 0.8, bc: DarcyBC | None = None) -> torch.Tensor:
    """Physical-space ``(n, 2, S, S)``: smooth log-normal permeability + its exact pressure solution."""
    if size < 4:
        raise ContractError("size must be >= 4")
    g = torch.Generator().manual_seed(seed)
    z = torch.randn(n, 1, size, size, generator=g, dtype=torch.float64)
    for _ in range(2):
        z = F.avg_pool2d(F.pad(z, (1, 1, 1, 1), mode="replicate"), 3, 1)
    z = z / z.flatten(1).std(dim=1).view(-1, 1, 1, 1).clamp_min(1e-9)
    sigma = contrast * (0.4 + 0.6 * torch.rand(n, 1, 1, 1, generator=g, dtype=torch.float64))
    spec = get_domain("darcy")
    k = torch.exp(sigma * z)[:, 0].clamp(spec.channels[0].lo * 5, spec.channels[0].hi / 5)
    p = solve_pressure(k, 1.0 / (size - 1), bc or DarcyBC())
    return torch.stack([k, p], dim=1).float()


def make_dataset(domain: str, n: int, size: int, seed: int = 0) -> tuple[torch.Tensor, float]:
    """Return ``(model_space_tensor (n, C, S, S), h)`` for any supported domain."""
    if domain in ("darcy", "cooling_plate"):
        phys = darcy_fields(n, size, seed)
    elif domain in PHYSICS_CLASSES:
        phys = PHYSICS_CLASSES[domain]().sample(n, size, seed)  # type: ignore[attr-defined]
    else:
        raise ContractError(f"no data generator for domain {domain!r}")
    x = ChannelNormalizer.for_domain(domain).to_model(phys.double()).float()
    if x.abs().max() > 1.0 + 1e-4:
        raise ContractError(
            f"generated data for {domain!r} exceeds the registered range (max |x| = {float(x.abs().max()):.3f})"
        )
    return x.clamp(-1.0, 1.0), 1.0 / (size - 1)


def cond_field_stub(n: int, channels: int, size: int, seed: int = 0) -> torch.Tensor:
    """Deterministic smooth spatial conditioning field for tests/demos ``(n, channels, S, S)``."""
    rng = np.random.default_rng(seed)
    g = np.linspace(0, 1, size)
    X, Y = np.meshgrid(g, g, indexing="ij")
    out = [
        [
            np.sin((c + 1) * np.pi * X + rng.uniform(0, 6)) * np.cos((c + 1) * np.pi * Y + rng.uniform(0, 6))
            for c in range(channels)
        ]
        for _ in range(n)
    ]
    return torch.tensor(np.array(out), dtype=torch.float32)
