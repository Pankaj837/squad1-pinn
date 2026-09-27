import pytest
import torch

from squad1.contracts import CandidateDesign, ChannelNormalizer
from squad1.errors import ContractError, NonFiniteError
from squad1.generation import (
    DiffusionScheduler,
    DiT,
    TrainConfig,
    cond_field_stub,
    darcy_fields,
    generate_candidates,
    make_dataset,
    quality_check,
    sample,
    train_diffusion,
)
from squad1.physics import DarcyPhysics
from squad1.physics.darcy import solve_pressure


# ------------------------------------------------------------------ scheduler
def test_scheduler_alpha_bar_monotone_and_bounds():
    s = DiffusionScheduler(100)
    assert (s.alpha_bar[1:] < s.alpha_bar[:-1]).all()
    assert 0.99 < s.alpha_bar[0] <= 1.0 and s.alpha_bar[-1] < 1e-3


def test_q_sample_and_predict_x0_are_inverse():
    s = DiffusionScheduler(100)
    x0 = torch.randn(4, 2, 8, 8)
    noise = torch.randn_like(x0)
    t = torch.tensor([3, 30, 60, 90])
    xt = s.q_sample(x0, t, noise)
    assert torch.allclose(s.predict_x0(xt, t, noise), x0, atol=1e-3 * (1 / s.alpha_bar[90].sqrt()))


def test_ddim_full_denoise_recovers_x0_with_oracle_eps():
    s = DiffusionScheduler(50)
    x0 = torch.randn(2, 1, 4, 4)
    noise = torch.randn_like(x0)
    seq = s.timestep_sequence(10)
    x = s.q_sample(x0, torch.full((2,), seq[0]), noise)
    for i, t in enumerate(seq):
        tt = torch.full((2,), t)
        eps = (x - s.alpha_bar[t].sqrt() * x0) / (1 - s.alpha_bar[t]).sqrt()  # oracle
        x = s.ddim_step(x, t, seq[i + 1] if i + 1 < len(seq) else -1, eps)
    assert torch.allclose(x, x0, atol=1e-4)


def test_timestep_sequence_validation():
    s = DiffusionScheduler(20)
    assert s.timestep_sequence(5)[0] == 19 and s.timestep_sequence(5)[-1] == 0
    with pytest.raises(ContractError):
        s.timestep_sequence(0)
    with pytest.raises(ContractError):
        DiffusionScheduler(1)


# ------------------------------------------------------------------------ DiT
def make_dit(**kw):
    args = dict(img_size=16, patch_size=4, in_channels=2, hidden_size=32, depth=2, num_heads=4)
    args.update(kw)
    return DiT(**args)


def test_dit_shapes_and_zero_init_output():
    m = make_dit()
    x = torch.randn(3, 2, 16, 16)
    y = m(x, torch.tensor([1, 5, 9]))
    assert y.shape == x.shape
    assert y.abs().max() == 0  # adaLN-Zero: identity/zero at init


def test_dit_conditioning_paths():
    m = make_dit(cond_dim=6, field_channels=3)
    for p in m.parameters():
        torch.nn.init.normal_(p, std=0.05) if p.ndim > 1 else None
    x = torch.randn(2, 2, 16, 16)
    t = torch.tensor([3, 7])
    cf = torch.randn(2, 3, 16, 16)
    a = m(x, t, torch.zeros(2, 6), cf)
    b = m(x, t, torch.ones(2, 6), cf)
    n = m(x, t, None, cf)
    d = m(x, t, torch.ones(2, 6), cf, drop_cond=torch.tensor([True, True]))
    assert (a - b).abs().max() > 1e-6
    assert torch.allclose(n, d, atol=1e-6)  # dropped == null conditioning
    with pytest.raises(ContractError):
        m(x, t, torch.ones(2, 5), cf)
    with pytest.raises(ContractError):
        m(x, t, torch.ones(2, 6), None)


def test_dit_input_validation():
    m = make_dit()
    with pytest.raises(ContractError):
        m(torch.randn(1, 3, 16, 16), torch.tensor([1]))
    with pytest.raises(ContractError):
        m(torch.randn(1, 2, 16, 16), torch.tensor([1, 2]))
    with pytest.raises(ContractError):
        m(torch.randn(1, 2, 16, 16), torch.tensor([1]), cond_vec=torch.zeros(1, 4))
    with pytest.raises(ContractError):
        m(torch.randn(1, 2, 16, 16), torch.tensor([1]), cond_field=torch.zeros(1, 1, 16, 16))
    with pytest.raises(ContractError):
        DiT(15, 4, 2)
    with pytest.raises(ContractError):
        DiT(16, 4, 2, hidden_size=30, num_heads=4)


def test_dit_grads_flow():
    m = make_dit(cond_dim=3)
    x = torch.randn(2, 2, 16, 16)
    for p in m.parameters():
        if p.ndim > 1:
            torch.nn.init.normal_(p, std=0.05)
    m(x, torch.tensor([2, 4]), torch.randn(2, 3), drop_cond=torch.tensor([True, False])).square().mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters() if p.requires_grad)


# ------------------------------------------------------------------ data + sampler
def test_darcy_data_is_on_manifold_and_in_range():
    x, h = make_dataset("darcy", 4, 16, seed=0)
    assert x.shape == (4, 2, 16, 16) and x.abs().max() <= 1.0
    phys = ChannelNormalizer.for_domain("darcy").to_physical(x.double())
    assert DarcyPhysics().residual_rms(phys, h).max() < 1e-4  # float32 + clamp round-trip floor


def test_make_dataset_all_domains_and_errors():
    from squad1.physics.domains import PHYSICS_CLASSES

    for d in PHYSICS_CLASSES:
        x, _ = make_dataset(d, 2, 12, seed=1)
        assert x.abs().max() <= 1.0
    with pytest.raises(ContractError):
        make_dataset("nope", 1, 8)
    with pytest.raises(ContractError):
        darcy_fields(1, 2)


def test_sampler_determinism_range_and_cfg():
    s = DiffusionScheduler(50)
    m = make_dit(cond_dim=4)
    for p in m.parameters():
        if p.ndim > 1:
            torch.nn.init.normal_(p, std=0.05)
    cv = torch.randn(3, 4)
    a = sample(m, s, 3, cond_vec=cv, steps=5, seed=7)
    b = sample(m, s, 3, cond_vec=cv, steps=5, seed=7)
    c = sample(m, s, 3, cond_vec=cv, steps=5, seed=8)
    g = sample(m, s, 3, cond_vec=cv, steps=5, seed=7, cfg_scale=2.0)
    assert torch.equal(a, b) and not torch.equal(a, c) and not torch.equal(a, g)
    assert a.abs().max() <= 1.0 and torch.isfinite(a).all()
    with pytest.raises(ContractError):
        sample(m, s, 3, steps=5, cfg_scale=1.0)
    with pytest.raises(ContractError):
        sample(m, s, 3, cond_vec=cv[:2], steps=5)
    with pytest.raises(ContractError):
        sample(m, s, 2, steps=5, physics_scale=1.0)


def test_generate_candidates_contract():
    s = DiffusionScheduler(20)
    m = make_dit()
    c = generate_candidates(m, s, "darcy", 2, 1 / 15, steps=3, seed=1)
    assert isinstance(c, CandidateDesign) and c.representation == "model" and c.generation["seed"] == 1
    with pytest.raises(ContractError):
        generate_candidates(m, s, "navier_stokes", 2, 1 / 15, steps=3)


def test_physics_guidance_runs_and_changes_samples():
    s = DiffusionScheduler(20)
    m = make_dit()
    for p in m.parameters():
        if p.ndim > 1:
            torch.nn.init.normal_(p, std=0.05)
    norm = ChannelNormalizer.for_domain("darcy")
    a = sample(m, s, 2, steps=4, seed=3)
    b = sample(m, s, 2, steps=4, seed=3, physics=DarcyPhysics(), normalizer=norm, h=1 / 15, physics_scale=0.05)
    assert torch.isfinite(b).all() and not torch.equal(a, b)


# ----------------------------------------------------------------- training loop
def test_training_reduces_loss_and_validates_inputs():
    torch.manual_seed(0)
    data, _ = make_dataset("laplace_heat", 16, 8, seed=0)
    s = DiffusionScheduler(50)
    m = DiT(8, 2, 1, hidden_size=32, depth=2, num_heads=4)
    ema, hist = train_diffusion(m, s, data, TrainConfig(steps=120, batch_size=16, lr=3e-3, log_every=40, seed=0))
    assert hist["loss"][-1] < hist["loss"][0]
    assert all(torch.isfinite(p).all() for p in ema.parameters())
    with pytest.raises(ContractError):
        train_diffusion(m, s, data[:, :, :4, :4], TrainConfig(steps=1))
    bad = data.clone()
    bad[0, 0, 0, 0] = float("nan")
    with pytest.raises(NonFiniteError):
        train_diffusion(m, s, bad, TrainConfig(steps=1))
    with pytest.raises(ContractError):
        train_diffusion(m, s, data, TrainConfig(steps=1), cond_vec=torch.zeros(3, 2))


def test_training_is_reproducible():
    data, _ = make_dataset("laplace_heat", 8, 8, seed=0)
    s = DiffusionScheduler(20)
    out = []
    for _ in range(2):
        torch.manual_seed(0)
        m = DiT(8, 2, 1, hidden_size=16, depth=1, num_heads=2)
        ema, hist = train_diffusion(m, s, data, TrainConfig(steps=20, batch_size=8, log_every=10, seed=5))
        out.append((hist["loss"], next(ema.parameters()).clone()))
    assert out[0][0] == out[1][0] and torch.equal(out[0][1], out[1][1])


# ----------------------------------------------------------------------- quality
def test_quality_check_flags_bad_samples():
    x, h = make_dataset("darcy", 3, 12, seed=0)
    x = x.clone()
    x[1] = 5.0  # out of range
    c = CandidateDesign("darcy", x, "model", h)
    r = quality_check(c, DarcyPhysics(), max_residual_rms=1.0)
    assert r["ok"] == [True, False, True] and r["n_ok"] == 2 and "residual_rms" in r


def test_quality_check_reports_physics_errors_instead_of_raising():
    from squad1.physics import get_physics

    c = CandidateDesign("navier_stokes", torch.rand(2, 3, 8, 8) * 2 - 1, "model", 0.1)
    r = quality_check(c, get_physics("navier_stokes"))
    assert r["batch"] == 2 and "residual_rms" in r


def test_cond_field_stub_and_solver_consistency():
    f = cond_field_stub(2, 3, 8, seed=1)
    assert f.shape == (2, 3, 8, 8) and torch.equal(f, cond_field_stub(2, 3, 8, seed=1))
    k = torch.ones(1, 8, 8, dtype=torch.float64)
    assert solve_pressure(k, 1 / 7).shape == (1, 8, 8)


def test_checkpoint_roundtrip_is_exact_and_safe(tmp_path):
    from squad1.generation import load_generator, save_generator

    torch.manual_seed(0)
    m = make_dit(cond_dim=4, field_channels=8)
    for p in m.parameters():
        if p.ndim > 1:
            torch.nn.init.normal_(p, std=0.05)
    s = DiffusionScheduler(30)
    path = save_generator(tmp_path / "ck" / "g.pt", m, s, {"note": "unit", "seed": 3})
    m2, s2, meta = load_generator(path)
    assert meta == {"note": "unit", "seed": 3} and s2.timesteps == 30 and m2.config == m.config
    x = torch.randn(2, 2, 16, 16)
    t = torch.tensor([3, 9])
    cv, cf = torch.randn(2, 4), torch.randn(2, 8, 16, 16)
    assert torch.equal(m.eval()(x, t, cv, cf), m2(x, t, cv, cf))
    torch.save({"format": "other"}, tmp_path / "bad.pt")
    with pytest.raises(ContractError):
        load_generator(tmp_path / "bad.pt")
