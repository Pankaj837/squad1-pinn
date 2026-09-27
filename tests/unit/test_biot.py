import numpy as np
import pytest
import sympy as sp
import torch

from squad1.errors import PhysicsError
from squad1.physics.biot import BiotPhysics


@pytest.fixture(autouse=True)
def _f64(double):
    yield


MU, LAM, ALPHA = 1.3, 0.7, 0.8


def _mms(H):
    x, y = sp.symbols("x y")
    ux = sp.sin(sp.pi * x) * sp.sin(sp.pi * y)
    uy = sp.sin(2 * sp.pi * x) * sp.sin(sp.pi * y) * 0.5
    p = 1 - x
    div = sp.diff(ux, x) + sp.diff(uy, y)
    fx = MU * (sp.diff(ux, x, 2) + sp.diff(ux, y, 2)) + (LAM + MU) * sp.diff(div, x) - ALPHA * sp.diff(p, x)
    fy = MU * (sp.diff(uy, x, 2) + sp.diff(uy, y, 2)) + (LAM + MU) * sp.diff(div, y) - ALPHA * sp.diff(p, y)
    fs = [sp.lambdify((x, y), e, "numpy") for e in (ux, uy, p, fx, fy)]
    g = np.linspace(0, 1, H)
    X, Y = np.meshgrid(g, g, indexing="ij")
    ux_, uy_, p_, fx_, fy_ = (torch.tensor(np.broadcast_to(f(X, Y), X.shape).copy()) for f in fs)
    return ux_, uy_, p_, fx_, fy_, 1.0 / (H - 1)


def _loss(H):
    ux, uy, p, fx, fy, h = _mms(H)
    x = torch.stack([torch.ones_like(p), p, ux, uy])[None]
    phys = BiotPhysics(MU, LAM, ALPHA, body_force=(fx, fy))
    return phys.loss(x, h).item() ** 0.5, phys, x, h


def test_manufactured_solution_second_order():
    e1, *_ = _loss(17)
    e2, *_ = _loss(33)
    # constraints are h^2-scaled second differences: truncation error ~ h^2 * h^2 -> ratio ~ 16 when h halves
    assert e1 < 5e-3
    assert e1 / e2 > 8


def test_wrong_solution_has_large_residual():
    _, phys, x, h = _loss(17)
    bad = x.clone()
    bad[:, 2] += 0.3 * torch.sin(3 * torch.linspace(0, 6, 17))[None, :, None]
    assert phys.loss(bad, h).item() > 100 * phys.loss(x, h).item()


def test_components_and_gradient_consistent():
    _, phys, x, h = _loss(17)
    comp = phys.component_losses(x, h)
    assert set(comp) == {"darcy", "mech_x", "mech_y", "bc"}
    g = phys.gradient(x + 0.01, h)
    assert g.shape == x.shape and torch.isfinite(g).all()


def test_pressure_couples_into_momentum():
    _, phys, x, h = _loss(17)
    y = x.clone()
    y[:, 1] = y[:, 1] + 0.1 * torch.sin(4 * torch.linspace(0, 3, 17))[None, :, None]
    assert phys.component_losses(y, h)["mech_x"] > phys.component_losses(x, h)["mech_x"]


def test_parameter_validation():
    with pytest.raises(PhysicsError):
        BiotPhysics(mu=-1)
    with pytest.raises(PhysicsError):
        BiotPhysics(alpha=5)
    with pytest.raises(PhysicsError):
        BiotPhysics(weights=(1, -1, 1))
