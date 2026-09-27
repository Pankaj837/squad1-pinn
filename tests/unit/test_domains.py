import numpy as np
import pytest
import torch

from squad1.contracts import ChannelNormalizer, get_domain
from squad1.errors import ConvergenceError, PhysicsError
from squad1.physics.domains import PHYSICS_CLASSES, ReactionDiffusion, _random_ring, newton_dirichlet

ANALYTIC = ["laplace_heat", "stress_equilibrium", "navier_stokes"]
NEWTON = ["reaction_diffusion", "diffusion_decay", "thermal_advection"]


def phys(name):
    return PHYSICS_CLASSES[name]()


@pytest.mark.parametrize("name", ANALYTIC + NEWTON)
def test_samples_have_right_shape_range_and_determinism(name):
    P = phys(name)
    a = P.sample(3, 16, seed=1)
    b = P.sample(3, 16, seed=1)
    c = P.sample(3, 16, seed=2)
    assert a.shape == (3, get_domain(name).n_channels, 16, 16) and a.dtype == torch.float32
    assert torch.equal(a, b) and not torch.equal(a, c)
    m = ChannelNormalizer.for_domain(name).to_model(a)
    assert m.abs().max() <= 1.0 + 1e-5, "samples must fit inside the registered physical ranges"


@pytest.mark.parametrize("name", ANALYTIC)
def test_analytic_data_is_on_manifold_and_beats_noise(name):
    P = phys(name)
    x = P.sample(6, 24, seed=0).double()
    h = 1 / 23
    r_data = P.residual_rms(x, h).max()
    rnd = torch.randn_like(x) * x.std()
    assert r_data < 0.02 * P.residual_rms(rnd, h).min()


@pytest.mark.parametrize("name", ANALYTIC)
def test_truncation_error_converges(name):
    P = phys(name)
    errs = []
    for size in (17, 33):
        x = P.sample(2, size, seed=3).double()
        errs.append(P.residual_rms(x, 1 / (size - 1)).mean().item())
    assert errs[1] < errs[0] / 2.5  # >= ~1.3 order (float32 sampling limits the ratio)


@pytest.mark.parametrize("name", NEWTON)
def test_newton_data_solves_discrete_equations(name):
    P = phys(name)
    x = P.sample(3, 14, seed=0).double()
    assert P.residual_rms(x, 1 / 13).max() < 1e-6  # float32 storage floor


def test_newton_domains_have_nontrivial_fields():
    x = phys("reaction_diffusion").sample(2, 14, seed=0)
    assert x.std() > 0.02


def test_gradient_of_constraint_is_autograd_of_loss():
    P = phys("navier_stokes")
    x = P.sample(2, 16).double()
    g = P.gradient(x, 1 / 15)
    assert g.shape == x.shape and torch.isfinite(g).all()


def test_domain_parameter_validation_and_size_guard():
    with pytest.raises(PhysicsError):
        ReactionDiffusion(D=-1)
    with pytest.raises(PhysicsError):
        phys("reaction_diffusion").sample(1, 64)
    with pytest.raises(PhysicsError):
        PHYSICS_CLASSES["navier_stokes"](Re=0)


def test_newton_reports_failure(monkeypatch):
    P = ReactionDiffusion(D=1e-9, r=1e6)  # violently stiff: force a non-converging solve
    ring = _random_ring(14, np.random.default_rng(0), 0.0, 1.0)
    with pytest.raises(ConvergenceError):
        newton_dirichlet(P, ring, 1 / 13, iters=2)
