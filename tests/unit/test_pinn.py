import math

import pytest
import torch

from squad1.errors import ContractError, PhysicsError, UnsafeExpressionError
from squad1.pinn import MLP, Heat1D, Poisson2D, TrainerConfig, compile_rule, rad_sample, relative_l2, train

# ------------------------------------------------------------------------ safe rules
V = {"P_escape", "impact", "cost", "budget"}


def test_rule_values():
    f = compile_rule("max(0, cost - budget) + P_escape * impact", V)
    out = f(
        cost=torch.tensor([5.0, 1.0]),
        budget=torch.tensor([3.0, 3.0]),
        P_escape=torch.tensor(0.5),
        impact=torch.tensor(0.2),
    )
    assert torch.isclose(out, torch.tensor(1.0 + 0.1))
    g = compile_rule("-sqrt(abs(cost)) ** 2 + 2.5", V)
    assert torch.isclose(g(cost=4.0), torch.tensor(-4.0 + 2.5))


@pytest.mark.parametrize(
    "bad",
    [
        "__import__('os').system('echo x')",
        "().__class__.__bases__[0].__subclasses__()",
        "open('f','w')",
        "cost.__class__",
        "[x for x in range(3)]",
        "cost if 1 else 2",
        "lambda: 1",
        "exec('1')",
        "unknown_var * 2",
        "'abc'",
        "True",
        "cost[0]",
        "max(cost)",
        "sqrt(cost, cost)",
        "sqrt(x=cost)",
        "cost @ cost",
        "cost < 1",
        "",
        "   ",
        "(",
        "1 + " * 200 + "1",
    ],
)
def test_injection_and_garbage_rejected(bad):
    with pytest.raises(UnsafeExpressionError):
        compile_rule(bad, V)


def test_missing_variable_at_call_time():
    with pytest.raises(TypeError):
        compile_rule("cost + budget", V)(cost=1.0)


# ------------------------------------------------------------------------- MLP
def test_mlp_shapes_normalisation_and_fourier():
    m = MLP(2, 1, hidden=16, layers=3, lo=(0.0, -1.0), hi=(1.0, 1.0), fourier_features=8, seed=1)
    assert m(torch.rand(5, 2)).shape == (5, 1)
    m2 = MLP(2, 1, hidden=16, layers=3, lo=(0.0, -1.0), hi=(1.0, 1.0), fourier_features=8, seed=1)
    assert torch.equal(m.B, m2.B)  # seeded Fourier matrix
    assert MLP(1, 2, activation="sin")(torch.rand(3, 1)).shape == (3, 2)
    with pytest.raises(ContractError):
        m(torch.rand(5, 3))
    with pytest.raises(ContractError):
        MLP(2, 1, activation="relu")
    with pytest.raises(ContractError):
        MLP(2, 1, lo=(0, 0), hi=(1, 0))
    with pytest.raises(ContractError):
        MLP(0, 1)


# --------------------------------------------------------------------- problems
def test_exact_solutions_satisfy_pde_ic_bc():
    class Exact(torch.nn.Module):
        def __init__(self, p):
            super().__init__()
            self.p = p

        def forward(self, x):
            return self.p.exact(x)

    gen = torch.Generator().manual_seed(0)
    for P in (Heat1D(0.05), Poisson2D()):
        x = P.sample_interior(200, gen).requires_grad_(True)
        assert P.residual(Exact(P), x).abs().max() < 1e-4
        xb, tb = P.boundary_points(100, gen)
        assert (P.exact(xb) - tb).abs().max() < 1e-5
        ic = P.initial_points(50, gen)
        if ic is not None:
            assert (P.exact(ic[0]) - ic[1]).abs().max() < 1e-5
    with pytest.raises(PhysicsError):
        Heat1D(-1)


def test_a_wrong_network_has_large_error_even_if_residual_is_zero():
    """The framework's original failure mode: u = 0 solves the heat PDE exactly but is not the solution."""
    zero = MLP(2, 1, lo=Heat1D.lo, hi=Heat1D.hi)
    for p in zero.parameters():
        torch.nn.init.zeros_(p)
    P = Heat1D()
    x = P.sample_interior(100, torch.Generator().manual_seed(0)).requires_grad_(True)
    assert P.residual(zero, x).abs().max() == 0
    assert relative_l2(zero, P) == pytest.approx(1.0, rel=1e-6)


def test_rad_sampling_concentrates_where_residual_is_large():
    class Peaked(Heat1D):
        def residual(self, model, x):
            return torch.exp(-200 * ((x[:, 0:1] - 0.5) ** 2 + (x[:, 1:2] - 0.0) ** 2))

    P = Peaked()
    pts = rad_sample(P, MLP(2, 1, lo=P.lo, hi=P.hi), 500, 20000, torch.Generator().manual_seed(0), k=2.0, c=0.0)
    near = ((pts[:, 0] - 0.5).abs() < 0.2) & (pts[:, 1].abs() < 0.2)
    assert near.float().mean() > 0.6  # uniform sampling would give ~ 0.16*0.5... = 8%


# ----------------------------------------------------------------------- training
def test_config_validation():
    with pytest.raises(ContractError):
        train(Heat1D(), cfg=TrainerConfig(adam_steps=-1))


def test_short_training_runs_reduces_loss_and_is_reproducible():
    cfg = TrainerConfig(
        adam_steps=60,
        lr=3e-3,
        n_interior=200,
        n_boundary=50,
        n_initial=50,
        log_every=20,
        balance_every=30,
        seed=3,
    )
    a = train(Heat1D(), cfg=cfg, hidden=16, layers=2)
    b = train(Heat1D(), cfg=cfg, hidden=16, layers=2)
    assert a.history["loss"][-1] < a.history["loss"][0]
    assert a.history["loss"] == b.history["loss"] and a.rel_l2 == b.rel_l2
    assert a.config_hash == b.config_hash and a.rel_l2 is not None and math.isfinite(a.rel_l2)
    assert a.weights["bc"] >= 1.0


def test_fixed_point_set_and_lbfgs_paths_and_rad():
    cfg = TrainerConfig(
        adam_steps=30,
        lbfgs_steps=2,
        resample=False,
        n_interior=100,
        n_boundary=30,
        n_initial=30,
        rad_every=10,
        rad_pool=500,
        log_every=10,
        balance_every=0,
    )
    r = train(Poisson2D(), cfg=cfg, hidden=16, layers=2)
    assert r.history["step"][-1] == 32 and math.isfinite(r.rel_l2)


@pytest.mark.slow
def test_heat_equation_reaches_low_error_against_exact_solution():
    cfg = TrainerConfig(
        adam_steps=2500,
        lbfgs_steps=15,
        lr=2e-3,
        n_interior=1000,
        n_boundary=200,
        n_initial=200,
        seed=0,
        log_every=500,
    )
    r = train(Heat1D(), cfg=cfg)
    assert r.rel_l2 < 0.06, f"rel L2 {r.rel_l2:.3f}"


@pytest.mark.slow
def test_poisson_with_fourier_features_low_error():
    cfg = TrainerConfig(
        adam_steps=2500,
        lbfgs_steps=15,
        lr=2e-3,
        n_interior=1000,
        n_boundary=200,
        seed=0,
        log_every=500,
        balance_every=200,
    )
    r = train(Poisson2D(), cfg=cfg, fourier_features=8, fourier_scale=1.0)
    assert r.rel_l2 < 0.06, f"rel L2 {r.rel_l2:.3f}"
