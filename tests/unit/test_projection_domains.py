"""Every registered physics domain goes through the same hard-projection pipeline."""

import numpy as np
import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from squad1.contracts import CandidateDesign, CondSpec
from squad1.encoding import FormulaTokenizer
from squad1.geometry import mesh_stats, remove_edge_contacts, voxels_to_mesh
from squad1.physics import BiotPhysics, get_physics
from squad1.physics.domains import PHYSICS_CLASSES
from squad1.projection import PCFMPipeline


@pytest.mark.parametrize("name", sorted(PHYSICS_CLASSES))
def test_each_library_domain_projects_perturbed_samples_to_the_manifold(name):
    P = get_physics(name)
    size = 14
    h = 1 / (size - 1)
    clean = P.sample(2, size, seed=1)
    torch.manual_seed(0)
    bad = clean + 0.01 * torch.randn_like(clean) * (clean.abs().amax(dim=(2, 3), keepdim=True) + 0.1)
    c = CandidateDesign(name, bad, "physical", h)
    r = PCFMPipeline(physics=P, max_iters=40).project(c)
    assert not r.rejected.any()
    assert (r.loss_after <= r.loss_before).all()
    assert r.residual_rms_after.max() < 1e-2 * r.residual_rms_before.min() or r.residual_rms_after.max() < 1e-6
    assert torch.isfinite(r.projected.tensor).all()


def test_biot_candidates_project_through_the_same_pipeline():
    size = 12
    h = 1 / (size - 1)
    g = torch.linspace(0, 1, size, dtype=torch.float64)
    p = (1 - g).view(1, size, 1).expand(1, size, size)
    exact = torch.stack(
        [torch.ones(1, size, size, dtype=torch.float64), p, torch.zeros_like(p), torch.zeros_like(p)], dim=1
    )
    torch.manual_seed(1)
    bad = exact.clone()
    bad[:, 2:] += 0.03 * torch.randn_like(bad[:, 2:])
    bad[:, 1] += 0.02 * torch.randn_like(bad[:, 1])
    bad[:, 0] *= 1 + 0.05 * torch.rand_like(bad[:, 0])
    c = CandidateDesign("darcy_biot", bad.float(), "physical", h)
    r = PCFMPipeline(max_iters=30).project(c)
    assert r.residual_rms_after.max() < 1e-2 * r.residual_rms_before.min() and not r.rejected.any()
    phys = BiotPhysics()
    x = r.projected.to_physical().tensor.double()
    assert phys.loss(x, h).max() < phys.loss(bad, h).max()


# -------------------------------------------------------------------- property tests
@given(st.lists(st.booleans(), min_size=27, max_size=27))
@settings(max_examples=60, deadline=None)
def test_voxel_mesh_is_closed_and_positive_for_any_occupancy(bits):
    occ = np.array(bits, dtype=bool).reshape(3, 3, 3)
    if not occ.any():
        return
    m = voxels_to_mesh(occ)
    s = mesh_stats(m)
    assert s.watertight and s.volume == pytest.approx(remove_edge_contacts(occ).sum())


@given(
    st.lists(st.floats(0, 1, allow_nan=False), min_size=3, max_size=3),
    st.floats(-1e3, 1e3, allow_nan=False),
    st.floats(-1e3, 1e3, allow_nan=False),
)
@settings(max_examples=60, deadline=None)
def test_condspec_roundtrip_property(fracs, a, b):
    total = sum(fracs)
    if total < 1e-6:
        return
    comp = {f"E{i}": f / total for i, f in enumerate(fracs)}
    spec = CondSpec(("E0", "E1", "E2"), ("a",), ("u",), ("b",), ("u",))
    d = spec.decode(spec.encode(comp, {"a": a}, {"b": b}))
    assert d["inferred"]["a"] == pytest.approx(a, rel=1e-5, abs=1e-3) and d["target"]["b"] == pytest.approx(
        b, rel=1e-5, abs=1e-3
    )
    assert sum(d["composition"].values()) == pytest.approx(1.0, abs=1e-5)


@given(st.lists(st.sampled_from(["Hf", "C", "Si", "Zr", "B", "2", "0.5", "-", "(", ")"]), min_size=1, max_size=12))
@settings(max_examples=80, deadline=None)
def test_formula_tokenizer_roundtrip_property(parts):
    text = "".join(parts)
    tok = FormulaTokenizer()
    try:
        toks = tok.tokenize(text)
    except Exception:
        return
    assert "".join(toks) == text
    tok.build_vocab([text])
    assert tok.decode(tok.encode(text, max_length=64)) == text
