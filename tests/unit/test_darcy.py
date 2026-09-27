import numpy as np
import pytest
import torch

from squad1.errors import ContractError, NonFiniteError, PhysicsError
from squad1.physics.darcy import DarcyBC, DarcyPhysics, darcy_residual, solve_pressure


@pytest.fixture(autouse=True)
def _f64(double):
    yield


H = 16
h = 1.0 / (H - 1)
xs = torch.linspace(0, 1, H, dtype=torch.float64)
X, Y = torch.meshgrid(xs, xs, indexing="ij")


def test_linear_pressure_uniform_k_zero_residual():
    r = darcy_residual(torch.ones(1, H, H), (1 - X)[None], h)
    assert r.abs().max() < 1e-8


def test_nonharmonic_pressure_analytic_residual():
    r = darcy_residual(torch.ones(1, H, H), (X**2)[None], h)  # div grad x^2 = 2
    assert torch.allclose(r, torch.full_like(r, 2.0), atol=1e-6)


def test_variable_k_exact_solution():
    k = (1 + X)[None]
    p = (1 - torch.log(1 + X) / np.log(2))[None]
    assert (darcy_residual(k, p, h) * h * h).abs().max() < 5e-4


@pytest.mark.parametrize("method", ["dense", "cg"])
def test_solver_gives_zero_residual(method):
    torch.manual_seed(0)
    k = torch.exp(0.7 * torch.randn(3, H, H))
    p = solve_pressure(k, h, method=method)
    phys = DarcyPhysics()
    x = torch.stack([k, p], 1)
    assert phys.loss(x, h).max() < 1e-16
    assert torch.allclose(p[:, 0], torch.ones(3, H)) and torch.allclose(p[:, -1], torch.zeros(3, H))


def test_dense_and_cg_agree_high_contrast():
    k = torch.full((1, H, H), 0.01)
    k[:, :, 5:8] = 100.0
    a = solve_pressure(k, h, method="dense")
    b = solve_pressure(k, h, method="cg", tol=1e-13)
    assert (a - b).abs().max() < 1e-7


def test_layout_changes_flow():
    ka = torch.full((1, H, H), 0.05)
    ka[:, :, 4:8] = 5.0  # channel along the flow direction... walls
    kb = torch.full((1, H, H), 0.05)
    kb[:, 4:8, :] = 5.0  # slab across the flow
    assert (solve_pressure(ka, h) - solve_pressure(kb, h)).abs().max() > 0.05


def test_gradient_matches_finite_difference_and_is_batched():
    torch.manual_seed(1)
    x = torch.rand(3, 2, H, H) + 0.5
    phys = DarcyPhysics()
    g = phys.gradient(x, h)
    assert g.shape == x.shape
    eps = 1e-6
    x2 = x.clone()
    x2[1, 1, 5, 5] += eps
    fd = (phys.loss(x2, h)[1] - phys.loss(x, h)[1]) / eps
    assert abs(fd - g[1, 1, 5, 5]) < 1e-4 * (1 + abs(fd))
    # per-sample independence: perturbing sample 1 must not change the loss of sample 0
    assert torch.equal(phys.loss(x2, h)[0], phys.loss(x, h)[0])


def test_solution_is_differentiable_wrt_k():
    k = torch.full((1, 8, 8), 1.0, requires_grad=True)
    p = solve_pressure(k, 1 / 7)
    p.sum().backward()
    assert k.grad is not None and torch.isfinite(k.grad).all()


def test_rejects_bad_inputs():
    phys = DarcyPhysics()
    with pytest.raises(ContractError):
        phys.check(torch.rand(4, 2))
    with pytest.raises(ContractError):
        phys.check(torch.rand(1, 3, H, H))
    bad = torch.ones(1, 2, H, H)
    bad[0, 0, 3, 3] = -1
    with pytest.raises(PhysicsError):
        phys.check(bad)
    bad = torch.ones(1, 2, H, H)
    bad[0, 1, 0, 0] = float("nan")
    with pytest.raises(NonFiniteError):
        phys.check(bad)
    with pytest.raises(PhysicsError):
        solve_pressure(torch.zeros(1, H, H), h)


def test_solve_dependent_and_custom_bc():
    phys = DarcyPhysics(DarcyBC(2.0, -1.0))
    k = torch.ones(1, H, H)
    x = phys.solve(k, h)
    assert phys.loss(x, h).max() < 1e-20
    assert torch.allclose(x[0, 1, 0], torch.full((H,), 2.0))
