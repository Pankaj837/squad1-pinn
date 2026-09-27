"""Does a *trained* generator respect its conditioning, and does physics guidance help? (small, CPU, ~1 min)."""

import numpy as np
import pytest
import torch

from squad1.contracts import ChannelNormalizer
from squad1.generation import DiffusionScheduler, DiT, TrainConfig, darcy_fields, sample, train_diffusion
from squad1.physics import DarcyPhysics

S = 16
H = 1 / (S - 1)


@pytest.fixture(scope="module")
def trained():
    norm = ChannelNormalizer.for_domain("darcy")
    xs, cs = [], []
    for i, c in enumerate(np.linspace(0.2, 1.4, 8)):
        xs.append(darcy_fields(24, S, seed=100 + i, contrast=float(c)))
        cs.append(torch.full((24,), float(c)))
    data = norm.to_model(torch.cat(xs).double()).float().clamp(-1, 1)
    cond = ((torch.cat(cs) - 0.8) / 0.4).unsqueeze(1)
    torch.manual_seed(0)
    model = DiT(S, 2, 2, hidden_size=64, depth=3, num_heads=4, cond_dim=1)
    sched = DiffusionScheduler(100)
    ema, hist = train_diffusion(
        model, sched, data, TrainConfig(steps=600, batch_size=32, seed=0, log_every=100), cond_vec=cond
    )
    return ema, sched, norm, hist


@pytest.mark.slow
def test_trained_generator_follows_the_conditioning(trained):
    ema, sched, norm, hist = trained
    assert hist["loss"][-1] < 0.4 * hist["loss"][0]

    def logk_std(cond_value, cfg):
        x = sample(ema, sched, 32, cond_vec=torch.full((32, 1), cond_value), steps=20, cfg_scale=cfg, seed=1)
        return float(torch.log(norm.to_physical(x.double())[:, 0]).flatten(1).std(dim=1).mean())

    low, high = (0.3 - 0.8) / 0.4, (1.3 - 0.8) / 0.4
    assert logk_std(high, 0.0) > 2.0 * logk_std(low, 0.0)  # conditioned contrast is respected
    assert logk_std(high, 2.0) / logk_std(low, 2.0) > logk_std(high, 0.0) / logk_std(low, 0.0)  # CFG sharpens it


@pytest.mark.slow
def test_physics_guidance_lowers_the_raw_residual(trained):
    ema, sched, norm, _ = trained
    phys = DarcyPhysics()

    def resid(scale):
        x = sample(
            ema,
            sched,
            16,
            cond_vec=torch.zeros(16, 1),
            steps=20,
            seed=2,
            physics=phys,
            normalizer=norm,
            h=H,
            physics_scale=scale,
        )
        return float(phys.residual_rms(norm.to_physical(x.double()), H).mean())

    assert resid(0.2) < 0.3 * resid(0.0)
