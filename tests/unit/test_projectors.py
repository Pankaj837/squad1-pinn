import pytest
import torch

from squad1.errors import ContractError, PhysicsError
from squad1.physics import DarcyPhysics, get_physics
from squad1.physics.darcy import solve_pressure
from squad1.projection import GaussNewtonConfig, get_projector


@pytest.fixture(autouse=True)
def _f64(double):
    yield


H = 14
h = 1.0 / (H - 1)


def imperfect(B=3, noise=0.2, seed=0):
    g = torch.Generator().manual_seed(seed)
    k = torch.exp(0.6 * torch.randn(B, H, H, generator=g))
    k = torch.nn.functional.avg_pool2d(k[:, None], 3, 1, 1)[:, 0]
    p = solve_pressure(k, h)
    truth = torch.stack([k, p], 1)
    bad = truth.clone()
    bad[:, 1] += noise * torch.randn(B, H, H, generator=g)
    return bad, truth


def test_gauss_newton_dense_reaches_machine_zero():
    x0, _ = imperfect()
    phys = DarcyPhysics()
    x, info = get_projector("gauss_newton").project(x0, phys, h)
    assert phys.loss(x0, h).min() > 1e-8
    assert info["loss_after"].max() < 1e-22 and info["converged"].all() and not info["rejected"].any()
    assert info["linear_solver"] == "dense"


def test_gn_fixed_k_equals_exact_solve_and_joint_also_lands_on_manifold():
    x0, _ = imperfect()
    phys = DarcyPhysics()
    xs, _ = get_projector("solve").project(x0, phys, h)
    xf, _ = get_projector("gauss_newton", fixed_channels=(0,)).project(x0, phys, h)
    xj, ij = get_projector("gauss_newton").project(x0, phys, h)
    assert (xs - xf).abs().max() < 1e-7
    assert torch.equal(xf[:, 0], x0[:, 0])
    assert ij["loss_after"].max() < 1e-20
    assert (xj[:, 0] - x0[:, 0]).abs().max() > 1e-6  # joint mode is allowed to move k
    assert (xj - x0).flatten(1).norm(dim=1).max() < 1.2 * (xf - x0).flatten(1).norm(dim=1).max()


def test_cg_matches_dense():
    x0, _ = imperfect(B=2, noise=0.1)
    phys = DarcyPhysics()
    a, _ = get_projector("gauss_newton", linear_solver="dense").project(x0, phys, h)
    b, ib = get_projector("gauss_newton", linear_solver="cg", cg_tol=1e-13, cg_max_iter=5000).project(x0, phys, h)
    assert ib["linear_solver"] == "cg"
    assert ib["loss_after"].max() < 1e-16
    assert (a - b).abs().max() < 1e-5


def test_first_order_methods_decrease_loss_monotonically_but_stall():
    x0, truth = imperfect(noise=0.1)
    phys = DarcyPhysics()
    l0 = phys.loss(x0, h)
    for name in ("gradient", "residual_weighted"):
        x, info = get_projector(name, max_iters=150).project(x0, phys, h)
        assert (info["loss_after"] <= l0).all()
        assert info["loss_after"].max() < 0.2 * l0.max()
    gn, ign = get_projector("gauss_newton").project(x0, phys, h)
    assert ign["loss_after"].max() < 1e-6 * info["loss_after"].max()  # hard projection >> penalty descent


def test_rejection_reverts_to_input_with_reason():
    x0, _ = imperfect(noise=0.5)
    phys = DarcyPhysics()
    x, info = get_projector("gauss_newton", max_correction_rms=1e-6).project(x0, phys, h)
    assert info["rejected"].all() and not info["converged"].any()
    assert torch.equal(x, x0) and info["reject_reason"] == ["excessive_correction"] * 3
    assert torch.equal(info["loss_after"], info["loss_before"])


def test_batch_samples_are_independent():
    x0, _ = imperfect(B=3)
    phys = DarcyPhysics()
    full, _ = get_projector("gauss_newton").project(x0, phys, h)
    single, _ = get_projector("gauss_newton").project(x0[1:2], phys, h)
    assert (full[1] - single[0]).abs().max() < 1e-9


def test_positivity_is_kept():
    x0, _ = imperfect(noise=0.6)
    x0[:, 0, 4:6, 4:6] = 1e-6
    x, _ = get_projector("gauss_newton").project(x0, DarcyPhysics(), h)
    assert (x[:, 0] > 0).all()


def test_nonlinear_navier_stokes_projection():
    P = get_physics("navier_stokes")
    x = P.sample(2, 16, seed=0).double()
    hh = 1 / 15
    bad = x + 0.02 * torch.randn_like(x)
    out, info = get_projector("gauss_newton", max_iters=25).project(bad, P, hh)
    assert info["loss_after"].max() < 1e-4 * info["loss_before"].min()


def test_solve_projector_unsupported_and_scm_pinn_not_implemented():
    P = get_physics("laplace_heat")
    x = P.sample(1, 12).double()
    with pytest.raises(PhysicsError):
        get_projector("solve").project(x, P, 1 / 11)
    with pytest.raises(NotImplementedError):
        get_projector("scm_pinn")
    with pytest.raises(NotImplementedError):
        get_projector("pirf")
    with pytest.raises(PhysicsError):
        get_projector("nope")


def test_invalid_inputs_rejected_before_work():
    with pytest.raises(ContractError):
        get_projector("gauss_newton").project(torch.rand(2, 2), DarcyPhysics(), h)
    with pytest.raises(PhysicsError):
        get_projector("gauss_newton", linear_solver="bogus").project(imperfect()[0], DarcyPhysics(), h)


def test_gn_config_defaults_are_sane():
    c = GaussNewtonConfig()
    assert c.max_iters >= 5 and c.tol <= 1e-8 and c.damping > 0
