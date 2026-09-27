import pytest
import torch

from squad1.contracts import CandidateDesign
from squad1.errors import ContractError, NonFiniteError
from squad1.physics import DarcyPhysics, get_physics
from squad1.physics.darcy import solve_pressure
from squad1.projection import PCFMPipeline


def darcy_candidate(rep="model", B=2, H=14, noise=0.05, dtype=torch.float32, seed=0):
    g = torch.Generator().manual_seed(seed)
    k = torch.exp(0.5 * torch.randn(B, H, H, generator=g, dtype=torch.float64))
    k = torch.nn.functional.avg_pool2d(k[:, None], 3, 1, 1)[:, 0]
    h = 1 / (H - 1)
    p = solve_pressure(k, h) + noise * torch.randn(B, H, H, generator=g, dtype=torch.float64)
    phys = CandidateDesign("darcy", torch.stack([k, p], 1).to(dtype), "physical", h)
    return phys if rep == "physical" else phys.to_model()


@pytest.mark.parametrize("rep", ["model", "physical"])
def test_returns_input_representation_and_dtype(rep):
    c = darcy_candidate(rep)
    r = PCFMPipeline().project(c)
    assert r.projected.representation == rep
    assert r.projected.tensor.dtype == torch.float32 and r.projected.tensor.shape == c.tensor.shape
    assert r.projected.h == c.h and r.projected.generation["projected_with"] == "gauss_newton"
    assert r.all_converged and not r.any_rejected
    phys = r.projected.to_physical()
    assert DarcyPhysics().loss(phys.tensor.double(), c.h).max() < 1e-9  # float32 storage floor
    assert r.residual_rms_after.max() < 1e-6 * r.residual_rms_before.min()


def test_no_op_when_already_physical():
    H = 12
    k = torch.exp(0.4 * torch.randn(1, H, H, dtype=torch.float64))
    h = 1 / (H - 1)
    x = torch.stack([k, solve_pressure(k, h)], 1)
    c = CandidateDesign("darcy", x.float(), "physical", h).to_model()
    r = PCFMPipeline().project(c)
    assert (r.projected.tensor - c.tensor).abs().max() < 1e-3
    assert r.correction_rms.max() < 1e-3


def test_rejection_returns_bit_identical_input_in_model_space():
    c = darcy_candidate("model", noise=0.4)
    r = PCFMPipeline(max_correction_rms=1e-5).project(c)
    assert r.rejected.all() and torch.equal(r.projected.tensor, c.tensor)
    assert r.reject_reason == ["excessive_correction"] * 2
    assert not r.converged.any()


def test_correction_is_measured_in_model_space():
    c = darcy_candidate("model", noise=0.1)
    r = PCFMPipeline().project(c)
    manual = (r.projected.tensor.double() - c.tensor.double()).flatten(1).pow(2).mean(1).sqrt()
    assert torch.allclose(manual, r.correction_rms, atol=1e-5)


@pytest.mark.parametrize("method", ["solve", "gradient", "residual_weighted"])
def test_other_methods_run_through_pipeline(method):
    c = darcy_candidate("model", noise=0.05)
    r = PCFMPipeline(method=method, **({"max_iters": 60} if method != "solve" else {})).project(c)
    assert (r.loss_after <= r.loss_before).all()
    assert r.method == method


def test_bc_from_candidate_metadata_and_result_dict():
    c = darcy_candidate("physical").replace(boundary={"p_left": 1.0, "p_right": 0.0})
    d = PCFMPipeline().project(c).to_dict()
    assert d["method"] == "gauss_newton" and len(d["loss_after"]) == 2 and d["converged"] == [True, True]


def test_contract_violations():
    with pytest.raises(ContractError):
        PCFMPipeline().project(torch.rand(1, 2, 8, 8))
    with pytest.raises(ContractError):
        CandidateDesign("darcy", torch.rand(2, 2), "model", 0.1)
    with pytest.raises(ContractError):
        CandidateDesign("darcy", torch.rand(1, 3, 8, 8), "model", 0.1)
    with pytest.raises(ContractError):
        CandidateDesign("darcy", torch.rand(1, 2, 8, 8), "weird", 0.1)
    with pytest.raises(ContractError):
        CandidateDesign("darcy", torch.rand(1, 2, 8, 8), "model", -1.0)
    bad = torch.rand(1, 2, 8, 8)
    bad[0, 0, 0, 0] = float("nan")
    with pytest.raises(NonFiniteError):
        CandidateDesign("darcy", bad, "model", 0.1)
    with pytest.raises(ContractError):
        CandidateDesign("no_such_domain", torch.rand(1, 2, 8, 8), "model", 0.1)


def test_domain_library_through_pipeline():
    x = get_physics("stress_equilibrium").sample(2, 16, seed=1)
    c = CandidateDesign("stress_equilibrium", x + 0.02 * torch.randn_like(x), "physical", 1 / 15)
    r = PCFMPipeline(max_iters=20).project(c)
    assert r.residual_rms_after.max() < 1e-3 * r.residual_rms_before.min()


def test_projection_never_leaves_registered_ranges():
    g = torch.Generator().manual_seed(9)
    wild = torch.rand(3, 2, 12, 12, generator=g) * 2 - 1  # unstructured candidate
    r = PCFMPipeline(max_iters=30).project(CandidateDesign("darcy", wild, "model", 1 / 11))
    assert r.projected.tensor.abs().max() <= 1.0 + 1e-5
    assert (r.loss_after <= r.loss_before).all()
    phys = r.projected.to_physical().tensor
    assert phys[:, 0].min() >= 1e-2 * (1 - 1e-5) and phys[:, 0].max() <= 1e2 * (1 + 1e-5)


def test_projection_runs_in_model_space_and_uses_generator_units():
    from squad1.projection.pipeline import ModelSpacePhysics

    c = darcy_candidate("model", noise=0.05)
    mp = ModelSpacePhysics(DarcyPhysics())
    lo, hi = mp.bounds(c.tensor)
    assert (lo == -1).all() and (hi == 1).all() and mp.clamp_feasible(c.tensor * 5).abs().max() == 1
    # constraint of the wrapper == constraint of the wrapped physics on the converted tensor
    phys = c.to_physical().tensor
    assert torch.allclose(
        mp.constraint_vector(c.tensor, c.h), DarcyPhysics().constraint_vector(phys, c.h), rtol=1e-4, atol=1e-6
    )
    mp.check(c.tensor)
    with pytest.raises(ContractError):
        mp.check(torch.rand(2, 3, 8, 8))
    with pytest.raises(NonFiniteError):
        mp.check(torch.full((1, 2, 8, 8), float("nan")))
    exact = mp.solve_dependent(c.tensor.double(), c.h)
    assert mp.loss(exact, c.h).max() < 1e-16
    from squad1.physics import get_physics

    assert ModelSpacePhysics(get_physics("laplace_heat")).solve_dependent(torch.zeros(1, 1, 8, 8), 0.1) is None


def test_fallback_solve_rescues_unconverged_samples_and_is_reported():
    c = darcy_candidate("model", noise=0.3)
    stuck = PCFMPipeline(max_iters=1, fallback_solve=False).project(c)
    assert not stuck.converged.any()
    rescued = PCFMPipeline(max_iters=1).project(c)
    assert rescued.converged.all() and rescued.metadata["fallback_solve"] == [True, True]
    assert rescued.residual_rms_after.max() < 1e-6
    capped = PCFMPipeline(max_iters=1, max_correction_rms=1e-6).project(c)
    assert capped.rejected.all() and torch.equal(capped.projected.tensor, c.tensor)


def test_fallback_respects_correction_cap_after_solve():
    c = darcy_candidate("model", noise=0.3)
    ok = PCFMPipeline(max_iters=1).project(c)
    cap = float(ok.correction_rms.max()) * 0.5
    capped = PCFMPipeline(max_iters=1, max_correction_rms=cap).project(c)
    assert capped.rejected.any() and all(
        r == "excessive_correction" for r, j in zip(capped.reject_reason, capped.rejected.tolist()) if j
    )


def test_garbage_candidates_are_projected_or_flagged_never_crash():
    g = torch.Generator().manual_seed(2)
    for scale in (1.0, 5.0):
        wild = (torch.rand(2, 2, 12, 12, generator=g) * 2 - 1) * scale
        wild = wild.clamp(-1, 1)
        r = PCFMPipeline().project(CandidateDesign("darcy", wild, "model", 1 / 11))
        assert torch.isfinite(r.projected.tensor).all() and r.projected.tensor.abs().max() <= 1.0 + 1e-5
        assert (r.loss_after <= r.loss_before).all()


def test_fallback_is_disabled_when_channels_are_pinned():
    p = PCFMPipeline(max_iters=1, fixed_channels=(0,))
    assert p.fallback_solve is False
    assert PCFMPipeline().fallback_solve is True and PCFMPipeline(method="solve").fallback_solve is False


def test_candidate_accepts_numpy_scalar_h_and_darcy_rejects_degenerate_width():
    import numpy as np

    c = CandidateDesign("darcy", torch.rand(1, 2, 8, 8), "model", np.float32(0.25))
    assert isinstance(c.h, float) and c.h == 0.25
    with pytest.raises(ContractError):
        CandidateDesign("darcy", torch.rand(1, 2, 8, 8), "model", True)
    from squad1.errors import PhysicsError
    from squad1.physics.darcy import solve_pressure

    with pytest.raises(PhysicsError):
        solve_pressure(torch.ones(1, 8, 1, dtype=torch.float64), 0.1)
