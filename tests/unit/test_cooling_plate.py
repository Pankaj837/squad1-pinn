import dataclasses

import pytest
import torch
import trimesh

from squad1.applications import CoolingParams, CoolingPlate, LayoutConfig, optimize_layout, plate_to_stl
from squad1.contracts import CandidateDesign
from squad1.errors import ContractError, PhysicsError
from squad1.physics.darcy import DarcyBC, DarcyPhysics


@pytest.fixture(autouse=True)
def _f64(double):
    yield


H = 24
h = 1.0 / (H - 1)


def layout(n, w, axis=1, hi=50.0, lo=0.02):
    k = torch.full((1, H, H), lo)
    for i in range(n):
        c = int(round((i + 0.5) * H / n - w / 2))
        if axis == 1:
            k[:, :, c : c + w] = hi
        else:
            k[:, c : c + w, :] = hi
    return k


def test_uniform_open_plate_matches_plug_flow_energy_balance():
    P = CoolingParams()
    pl = CoolingPlate(P)
    r = pl.solve(torch.full((1, H, H), 50.0), h)
    u = P.q_pump / (H * P.plate_length_m / (H - 1))  # plate width = W cells of size L/(H-1)
    rise = (P.heat_flux_w_m2 / P.thickness_m) * P.plate_length_m / (P.rho_c * u)
    assert float(r["T"][0, -1].mean()) - P.t_in_c == pytest.approx(rise, rel=0.01)
    assert float(r["energy_balance_error"]) < 1e-6 and float(r["flow_residual"]) < 1e-12


def test_conduction_only_limit_matches_analytic_profile():
    P = dataclasses.replace(CoolingParams(), q_pump=1e-12)
    pl = CoolingPlate(P)
    r = pl.solve(torch.full((1, H, H), 0.02), h)
    L, S, kap = P.plate_length_m, P.heat_flux_w_m2 / P.thickness_m, P.kappa_solid
    x = torch.linspace(0, L, H)
    exact = P.t_in_c + S * (L * x - x**2 / 2) / kap
    assert float(r["T"][0, :, 0][-1]) == pytest.approx(
        float(exact[-1]), rel=0.06
    )  # last row is a full cell: O(h) boundary effect
    assert (r["T"][0, :, 3] - exact).abs().max() < 0.07 * (exact[-1] - P.t_in_c)


def test_energy_is_conserved_for_random_layouts():
    torch.manual_seed(0)
    pl = CoolingPlate()
    k = torch.exp(1.5 * torch.randn(3, H, H)).clamp(0.02, 50)
    assert pl.solve(k, h)["energy_balance_error"].max() < 1e-5


def test_metrics_depend_on_layout_not_only_area_fraction():
    pl = CoolingPlate()
    straight = layout(4, 2)
    perm = torch.randperm(H * H, generator=torch.Generator().manual_seed(0))
    shuffled = straight.reshape(-1)[perm].reshape(1, H, H)
    rs, rh = pl.report(straight, h)[0], pl.report(shuffled, h)[0]
    assert rs["relative_weight"] == pytest.approx(rh["relative_weight"])  # same channel area
    assert rh["pressure_drop_kPa"] > 20 * rs["pressure_drop_kPa"]  # ... very different flow
    across = layout(4, 2, axis=0)  # channels blocking the flow
    assert pl.report(across, h)[0]["pressure_drop_kPa"] > 50 * rs["pressure_drop_kPa"]


def test_more_channels_lower_pressure_drop_and_weight_tradeoff():
    pl = CoolingPlate()
    a, b = pl.report(layout(3, 2), h)[0], pl.report(layout(6, 2), h)[0]
    assert b["pressure_drop_kPa"] < a["pressure_drop_kPa"]  # more parallel channels: lower resistance
    assert b["relative_weight"] < a["relative_weight"]  # ... and less metal (same width each)


def test_gradient_is_finite_and_matches_finite_difference():
    """Generic (non-symmetric) layout: upwind advection has a kink at exactly zero velocity, so symmetric layouts are avoided."""
    pl = CoolingPlate()
    g = torch.Generator().manual_seed(4)
    k0 = torch.exp(1.2 * torch.randn(1, H, H, generator=g)).clamp(0.05, 30.0)
    k = k0.clone().requires_grad_(True)
    r = pl.solve(k, h)
    obj = r["smooth_max_temperature_C"].sum() + r["pressure_drop_kPa"].sum()
    obj.backward()
    assert torch.isfinite(k.grad).all() and k.grad.abs().max() > 0
    for j in ((0, 9, 7), (0, 15, 12)):
        eps = 1e-5
        k2 = k0.clone()
        k2[j] += eps
        r2 = pl.solve(k2, h)
        fd = ((r2["smooth_max_temperature_C"].sum() + r2["pressure_drop_kPa"].sum()) - obj.detach()) / eps
        assert float(k.grad[j]) == pytest.approx(float(fd), rel=1e-4, abs=1e-7)


def make_candidate(k, rep="model", dtype=torch.float32):
    p = DarcyPhysics().solve(k, h)
    return (
        CandidateDesign("cooling_plate", p.to(dtype), "physical", h).to_model()
        if rep == "model"
        else CandidateDesign("cooling_plate", p.to(dtype), "physical", h)
    )


def test_optimizer_repairs_a_violating_design_and_restores_flow_consistency():
    pl = CoolingPlate()
    cand = make_candidate(layout(2, 2))
    before = pl.report(cand.to_physical().tensor[:, 0].double(), h)[0]
    assert not before["constraints_satisfied"]
    out, info = optimize_layout(pl, cand, LayoutConfig(steps=150))
    after = info["report_after"][0]
    assert after["constraints_satisfied"] and after["flow_residual"] < 1e-8
    assert after["max_temperature_C"] <= pl.p.t_target_c and after["pressure_drop_kPa"] <= pl.p.dp_target_kpa
    assert out.representation == "model" and out.tensor.dtype == torch.float32 and out.tensor.shape == cand.tensor.shape
    assert not any(info["rejected"]) and info["report_before"][0]["max_temperature_C"] == pytest.approx(
        before["max_temperature_C"]
    )
    # the returned pressure channel is the exact flow solution of the returned layout
    phys = out.to_physical().tensor.double()
    assert DarcyPhysics().loss(phys, h).max() < 1e-9


def test_optimizer_rejects_when_correction_exceeds_cap_and_returns_input_untouched():
    pl = CoolingPlate()
    cand = make_candidate(layout(2, 2))
    out, info = optimize_layout(pl, cand, LayoutConfig(steps=40, max_correction_rms=1e-4))
    assert info["rejected"] == [True] and info["reject_reason"] == ["excessive_correction"]
    assert torch.equal(out.tensor, cand.tensor)


def test_infeasible_targets_are_reported_not_hidden():
    P = dataclasses.replace(CoolingParams(), t_target_c=26.0)  # below the coolant-bulk floor
    pl = CoolingPlate(P)
    out, info = optimize_layout(pl, make_candidate(layout(4, 2)), LayoutConfig(steps=30))
    assert info["report_after"][0]["constraints_satisfied"] is False


def test_structural_minimum_is_enforced_and_reported():
    pl = CoolingPlate()
    thin = layout(8, 2)  # weight ~0.33, above the 0.25 default -> should be fine
    r = pl.report(thin, h)[0]
    assert r["structure_ok"] and r["relative_weight"] >= pl.p.min_solid_fraction
    strict = CoolingPlate(dataclasses.replace(CoolingParams(), min_solid_fraction=0.5))
    r2 = strict.report(thin, h)[0]
    assert not r2["structure_ok"] and not r2["constraints_satisfied"]
    # gradient descent on a per-pixel layout is a local method: from a very thin-channel start it may not
    # reach a global optimum in a fixed budget, but the structural penalty must still pull weight upward
    # and the report must never claim success it did not reach (no silently-passing false positive).
    out, info = optimize_layout(strict, make_candidate(thin), LayoutConfig(steps=150))
    after = info["report_after"][0]
    assert after["relative_weight"] > r2["relative_weight"]
    assert after["structure_ok"] == (after["relative_weight"] >= strict.p.min_solid_fraction)
    assert after["constraints_satisfied"] is False or after["structure_ok"]
    # from a starting point already close to the limit, the linear penalty does reach the target
    near = layout(6, 2)  # weight 0.5, exactly at the strict limit
    out2, info2 = optimize_layout(strict, make_candidate(near), LayoutConfig(steps=150))
    after2 = info2["report_after"][0]
    assert after2["structure_ok"] and after2["relative_weight"] >= strict.p.min_solid_fraction * (1 - 1e-6)
    with pytest.raises(PhysicsError):
        CoolingParams(min_solid_fraction=1.0)


def test_optimizer_input_validation():
    pl = CoolingPlate()
    with pytest.raises(ContractError):
        optimize_layout(pl, CandidateDesign("laplace_heat", torch.rand(1, 1, 8, 8), "model", 0.1))
    with pytest.raises(ContractError):
        pl.solve(torch.ones(1, 4, 4), 0.3)
    with pytest.raises(ContractError):
        pl.solve(torch.ones(4, 4), 0.3)
    with pytest.raises(PhysicsError):
        CoolingPlate(bc=DarcyBC(0.0, 1.0))
    with pytest.raises(PhysicsError):
        CoolingParams(q_pump=0)
    with pytest.raises(PhysicsError):
        CoolingParams(t_target_c=10.0)


def test_stl_export_is_watertight_positive_and_scaled(tmp_path):
    k = layout(4, 2)[0]
    info = plate_to_stl(k, tmp_path / "plate.stl", layers=3)
    P = CoolingParams()
    solid_cells = int((k <= P.channel_threshold).sum())
    assert info["watertight"] and info["positive_volume"] and info["components"] >= 1
    cell = P.plate_length_m * 1e3 / (H - 1)
    assert info["volume_mm3"] == pytest.approx(solid_cells * cell * cell * P.thickness_m * 1e3, rel=1e-6)
    t = trimesh.load(tmp_path / "plate.stl")
    assert t.is_watertight and t.volume > 0
    with pytest.raises(PhysicsError):
        plate_to_stl(torch.full((H, H), 50.0), tmp_path / "x.stl")
    with pytest.raises(ContractError):
        plate_to_stl(torch.ones(2, H, H), tmp_path / "x.stl")


def test_solver_input_guards():
    pl = CoolingPlate()
    bad = torch.ones(1, H, H)
    bad[0, 2, 2] = 0.0
    with pytest.raises(PhysicsError):
        pl.solve(bad, h)
    bad[0, 2, 2] = float("nan")
    with pytest.raises(PhysicsError):
        pl.solve(bad, h)
    with pytest.raises(ContractError):
        pl.solve(torch.ones(1, 80, 80), 1 / 79)
