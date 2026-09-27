import json

import numpy as np
import pytest
import torch

from squad1.applications import CoolingPlate, LayoutConfig, optimize_layout
from squad1.conditioning import BoundarySegment, BoundarySpec, SourceRegion
from squad1.contracts import ChannelNormalizer, CondSpec
from squad1.errors import ConditioningError, ContractError, NonFiniteError
from squad1.generation import DiffusionScheduler, DiT
from squad1.physics import DarcyPhysics
from squad1.pipeline import E2EConfig, Squad1Pipeline, build_toy_generator, darcy_bc_from_boundary, summarize
from squad1.projection import PCFMPipeline

SIZE = 16
H = 1 / (SIZE - 1)


@pytest.fixture(scope="module")
def toy():
    model, sched, hist = build_toy_generator("darcy", SIZE, train_steps=250, n_train=96, seed=0, hidden=48, depth=2)
    return model, sched, hist


def test_toy_generator_trains(toy):
    _, _, hist = toy
    assert hist["loss"][-1] < 0.8 * hist["loss"][0]


@pytest.mark.slow
def test_end_to_end_unconditioned_generation_projection_and_package(toy, tmp_path):
    model, sched, _ = toy
    pkg = Squad1Pipeline(model, sched, config=E2EConfig(domain="darcy", n=4, steps=15, seed=3)).run()
    r = pkg.result
    # raw generator samples are imperfect; the projected ones satisfy the physics
    assert r.residual_rms_before.min() > 1e-4
    assert r.all_converged and not r.any_rejected
    phys = pkg.final.to_physical().tensor.double()
    assert DarcyPhysics().residual_rms(phys, H).max() < 1e-4  # float32 storage floor
    assert r.residual_rms_after.max() < 1e-3 * r.residual_rms_before.min()
    s = summarize(pkg)
    assert s["converged"] == 4 and s["rejected"] == 0
    # trained generator is much closer to the manifold than unstructured noise
    noise = ChannelNormalizer.for_domain("darcy").to_physical(torch.rand(4, 2, SIZE, SIZE) * 2 - 1).double()
    assert pkg.candidate_raw.tensor.shape == (4, 2, SIZE, SIZE)
    assert float(r.residual_rms_before.mean()) < float(DarcyPhysics().residual_rms(noise, H).mean())
    # package: JSON contract + tensors + provenance
    path = pkg.save(tmp_path / "pkg")
    d = json.loads(path.read_text())
    assert d["schema_version"] == "squad1_design_package_v1" and d["accepted"] == [True] * 4
    assert d["channels"] == ["k", "p"] and d["provenance"]["seed"] == 3 and d["provenance"]["config_hash"]
    assert np.load(tmp_path / "pkg" / "design_final.npy").shape == (4, 2, SIZE, SIZE)


def test_pipeline_is_deterministic_for_a_seed(toy):
    model, sched, _ = toy
    cfg = E2EConfig(n=2, steps=6, seed=11)
    a = Squad1Pipeline(model, sched, config=cfg).run()
    b = Squad1Pipeline(model, sched, config=cfg).run()
    assert torch.equal(a.candidate_raw.tensor, b.candidate_raw.tensor) and torch.equal(a.final.tensor, b.final.tensor)


def test_boundary_conditions_flow_from_spec_into_physics(toy):
    model, sched, _ = toy
    spec = BoundarySpec(
        (
            BoundarySegment("x_min", "dirichlet", 0.8),
            BoundarySegment("x_max", "dirichlet", 0.2),
            BoundarySegment("y_min", "neumann", 0.0),
            BoundarySegment("y_max", "neumann", 0.0),
        )
    )
    assert darcy_bc_from_boundary(spec) == {"p_left": 0.8, "p_right": 0.2}
    assert darcy_bc_from_boundary(None) == {}
    partial = BoundarySpec((BoundarySegment("x_min", "dirichlet", 0.8, 0.0, 0.5),))
    assert darcy_bc_from_boundary(partial) == {}
    pkg = Squad1Pipeline(model, sched, config=E2EConfig(n=2, steps=6, seed=1)).run(boundary=spec)
    assert pkg.final.boundary == {"p_left": 0.8, "p_right": 0.2}
    p = pkg.final.to_physical().tensor[:, 1]
    assert torch.allclose(p[:, 0], torch.full((2, SIZE), 0.8), atol=1e-4) and torch.allclose(
        p[:, -1], torch.full((2, SIZE), 0.2), atol=1e-4
    )


def test_rejection_path_keeps_input_and_flags(toy):
    model, sched, _ = toy
    pkg = Squad1Pipeline(model, sched, config=E2EConfig(n=2, steps=6, seed=2, max_correction_rms=1e-6)).run()
    assert pkg.result.rejected.all() and torch.equal(pkg.final.tensor, pkg.candidate_raw.tensor)
    assert json.loads(json.dumps(pkg.to_dict(), default=str))["accepted"] == [False, False]


def test_quality_gate_reports_raw_residual(toy):
    model, sched, _ = toy
    pkg = Squad1Pipeline(model, sched, config=E2EConfig(n=3, steps=6, max_residual_rms_gate=1e-12)).run()
    assert pkg.quality["ok"] == [False] * 3 and len(pkg.quality["residual_rms"]) == 3


def test_other_projection_methods_via_config(toy):
    model, sched, _ = toy
    pkg = Squad1Pipeline(
        model, sched, config=E2EConfig(n=2, steps=6, method="gradient", projector_config={"max_iters": 40})
    ).run()
    assert pkg.result.method == "gradient" and (pkg.result.loss_after <= pkg.result.loss_before).all()


# ------------------------------------------------------------------- conditioned generator
ELEMS = tuple(f"E{i}" for i in range(6))
SPEC = CondSpec(ELEMS, ("t0", "t1"), ("K", "W/mK"), ("T_op",), ("K",))
BSPEC = BoundarySpec(
    (
        BoundarySegment("x_min", "dirichlet", 1.0),
        BoundarySegment("x_max", "dirichlet", 0.0),
        BoundarySegment("y_min", "neumann", 0.0),
    ),
    (SourceRegion(0.4, 0.6, 0.4, 0.6, 1.0),),
)
ITEM = {
    "composition": {"E0": 0.6, "E3": 0.4},
    "inferred": {"t0": 300.0, "t1": 12.0},
    "target": {"T_op": 500.0},
}


def conditioned_model(**kw):
    torch.manual_seed(0)
    m = DiT(SIZE, 2, 2, hidden_size=32, depth=2, num_heads=4, cond_dim=SPEC.dv, field_channels=8, **kw)
    for p in m.parameters():
        if p.ndim > 1:
            torch.nn.init.normal_(p, std=0.03)
    return m


def test_conditioned_pipeline_end_to_end():
    pipe = Squad1Pipeline(
        conditioned_model(), DiffusionScheduler(50), SPEC, E2EConfig(n=2, steps=5, seed=4, cfg_scale=1.5)
    )
    pkg = pipe.run([ITEM, {**ITEM, "target": {"T_op": 700.0}}], BSPEC)
    assert pkg.final.batch_size == 2
    assert pkg.final.tensor.abs().max() <= 1.0 + 1e-5  # projection stays inside the registered ranges
    assert (pkg.result.loss_after <= pkg.result.loss_before).all()
    assert pkg.to_dict()["cond_spec"]["composition_order"] == list(ELEMS)
    # same pipeline accepts a pre-built (B, Dv) tensor
    tens = SPEC.encode_batch([ITEM, ITEM])
    assert pipe.run(tens, BSPEC).final.batch_size == 2


def test_frozen_interface_mismatches_fail_loudly():
    sched = DiffusionScheduler(50)
    with pytest.raises(ConditioningError):  # model built for a different Dv than the frozen CondSpec
        Squad1Pipeline(
            DiT(SIZE, 2, 2, hidden_size=32, depth=1, num_heads=4, cond_dim=SPEC.dv + 1, field_channels=8),
            sched,
            SPEC,
        )
    with pytest.raises(ConditioningError):  # conditioning model but no spec
        Squad1Pipeline(conditioned_model(), sched, None)
    with pytest.raises(ConditioningError):  # wrong number of cond_field channels
        Squad1Pipeline(
            DiT(SIZE, 2, 2, hidden_size=32, depth=1, num_heads=4, cond_dim=SPEC.dv, field_channels=3),
            sched,
            SPEC,
        )
    with pytest.raises(ContractError):  # wrong domain for the model
        Squad1Pipeline(conditioned_model(), sched, SPEC, E2EConfig(domain="navier_stokes"))
    pipe = Squad1Pipeline(conditioned_model(), sched, SPEC, E2EConfig(n=1, steps=3))
    with pytest.raises(ConditioningError):
        pipe.run(None, BSPEC)
    with pytest.raises(ConditioningError):
        pipe.run([ITEM], None)
    with pytest.raises(NonFiniteError):
        pipe.run([{**ITEM, "inferred": {"t0": float("nan"), "t1": 1.0}}], BSPEC)
    with pytest.raises(ConditioningError):
        pipe.run([ITEM, ITEM], [BSPEC])  # batch of boundaries != batch of conditions


# --------------------------------------------------------------- domain library + cooling plate
def test_domain_library_generator_projection_roundtrip():
    model, sched, _ = build_toy_generator("laplace_heat", 12, train_steps=60, n_train=24, seed=1, hidden=32, depth=1)
    pkg = Squad1Pipeline(
        model, sched, config=E2EConfig(domain="laplace_heat", n=3, steps=5, method="gauss_newton")
    ).run()
    assert pkg.result.residual_rms_after.max() < 1e-3 * pkg.result.residual_rms_before.min()


def test_cooling_plate_flow_from_generator_to_repaired_design(toy):
    model, sched, _ = toy
    pkg = Squad1Pipeline(model, sched, config=E2EConfig(domain="cooling_plate", n=2, steps=6, seed=5)).run()
    plate = CoolingPlate()
    fixed, info = optimize_layout(plate, pkg.final, LayoutConfig(steps=30))
    assert fixed.domain == "cooling_plate" and fixed.tensor.shape == pkg.final.tensor.shape
    # whatever the generator produced, the returned design is flow-consistent and reports honest constraint flags
    phys = fixed.to_physical().tensor.double()
    assert DarcyPhysics().loss(phys, H).max() < 1e-8
    assert all(isinstance(r["constraints_satisfied"], bool) for r in info["report_after"])
    # PCFM alone (flow physics) accepts the same design without changing the layout channel much
    res = PCFMPipeline(method="solve").project(fixed)
    assert res.correction_rms.max() < 1e-3


def test_unused_conditioning_is_rejected_not_ignored(toy):
    model, sched, _ = toy
    with pytest.raises(ConditioningError):
        Squad1Pipeline(model, sched, config=E2EConfig(n=1, steps=3)).run([ITEM])
