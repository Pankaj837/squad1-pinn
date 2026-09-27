import json

import numpy as np
import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from squad1.contracts import (
    CandidateDesign,
    ChannelNormalizer,
    ChannelSpec,
    CondSpec,
    DomainSpec,
    available_domains,
    get_domain,
    register_domain,
)
from squad1.errors import ConditioningError, ContractError, NonFiniteError, NormalizationError


# ------------------------------------------------------------------ channels / normaliser
def test_channelspec_validation():
    with pytest.raises(NormalizationError):
        ChannelSpec("x", 1.0, 1.0)
    with pytest.raises(NormalizationError):
        ChannelSpec("x", 0.0, 1.0, "log")
    with pytest.raises(NormalizationError):
        ChannelSpec("x", 0.0, 1.0, "weird")
    with pytest.raises(NormalizationError):
        ChannelSpec("", 0.0, 1.0)
    with pytest.raises(NormalizationError):
        ChannelSpec("x", float("nan"), 1.0)


def test_domain_registry():
    d = get_domain("darcy")
    assert d.names == ("k", "p") and d.n_channels == 2 and d.index("p") == 1
    with pytest.raises(ContractError):
        d.index("z")
    with pytest.raises(ContractError):
        get_domain("nope")
    with pytest.raises(ContractError):
        DomainSpec("dup", (ChannelSpec("a", 0, 1), ChannelSpec("a", 0, 1)))
    with pytest.raises(ContractError):
        DomainSpec("empty", ())
    with pytest.raises(ContractError):
        register_domain(DomainSpec("darcy", (ChannelSpec("q", 0, 1),)))
    assert {"darcy", "darcy_biot", "cooling_plate", "navier_stokes", "laplace_heat"} <= set(available_domains())


def test_normalizer_channel_axis_names_and_values():
    n = ChannelNormalizer([ChannelSpec("a", 0.0, 10.0), ChannelSpec("b", 1.0, 100.0, "log")])
    x = torch.zeros(1, 2, 4, 4)  # model 0  ->  a = 5 ; b = 10 (geometric mean)
    ph = n.to_physical(x)
    assert torch.allclose(ph[:, 0], torch.full((1, 4, 4), 5.0)) and torch.allclose(
        ph[:, 1], torch.full((1, 4, 4), 10.0)
    )
    edge = torch.full((1, 2, 2, 2), -1.0)
    assert torch.allclose(n.to_physical(edge)[0, :, 0, 0], torch.tensor([0.0, 1.0]))
    assert torch.allclose(n.to_physical(-edge)[0, :, 0, 0], torch.tensor([10.0, 100.0]), rtol=1e-5)


@given(st.lists(st.floats(-1, 1, allow_nan=False, width=32), min_size=8, max_size=8))
@settings(max_examples=60, deadline=None)
def test_normalizer_roundtrip_property(vals):
    n = ChannelNormalizer.for_domain("darcy")
    x = torch.tensor(vals, dtype=torch.float64).view(1, 2, 2, 2)
    assert torch.allclose(n.to_model(n.to_physical(x)), x, atol=1e-9)


def test_normalizer_dtype_device_autograd_preserved():
    n = ChannelNormalizer.for_domain("darcy")
    x = torch.rand(2, 2, 4, 4, dtype=torch.float32, requires_grad=True)
    y = n.to_physical(x)
    assert y.dtype == torch.float32 and y.requires_grad
    y.sum().backward()
    assert torch.isfinite(x.grad).all()


def test_normalizer_errors():
    n = ChannelNormalizer.for_domain("darcy")
    with pytest.raises(ContractError):
        n.to_physical(torch.zeros(1, 3, 4, 4))
    with pytest.raises(ContractError):
        n.to_physical(torch.zeros(1, 2, 4, 4, dtype=torch.long))
    with pytest.raises(NormalizationError):
        ChannelNormalizer([])
    assert n.clip_model(torch.full((1, 2, 2, 2), 5.0)).max() == 1.0


def test_to_model_log_channel_handles_nonpositive_without_nan():
    n = ChannelNormalizer.for_domain("darcy")
    out = n.to_model(torch.zeros(1, 2, 2, 2))
    assert torch.isfinite(out).all()


# ------------------------------------------------------------------------ candidate
def test_candidate_representation_conversion_roundtrip_and_replace():
    x = torch.rand(2, 2, 6, 6) * 2 - 1
    c = CandidateDesign("darcy", x, "model", 0.2)
    assert c.batch_size == 2 and c.to_model() is c
    back = c.to_physical().to_model()
    assert torch.allclose(back.tensor, x, atol=1e-5)
    assert c.replace(h=0.5).h == 0.5 and c.h == 0.2


def test_candidate_min_size_and_dtype():
    with pytest.raises(ContractError):
        CandidateDesign("darcy", torch.zeros(1, 2, 2, 2), "model", 1.0)
    with pytest.raises(ContractError):
        CandidateDesign("darcy", torch.zeros(1, 2, 8, 8, dtype=torch.long), "model", 1.0)
    with pytest.raises(NonFiniteError):
        CandidateDesign("darcy", torch.full((1, 2, 8, 8), float("inf")), "model", 1.0)


# ---------------------------------------------------------------------------- cond
def make_spec(**kw):
    base = dict(
        composition_order=("Al", "Cu", "Mg"),
        inferred_order=("theta0", "theta1"),
        inferred_units=("K", "W/mK"),
        target_order=("T_max",),
        target_units=("K",),
    )
    base.update(kw)
    return CondSpec(**base)


def test_condspec_encode_decode_roundtrip():
    spec = make_spec()
    v = spec.encode({"Al": 0.7, "Cu": 0.3}, {"theta0": 300.0, "theta1": 12.5}, {"T_max": 900.0})
    assert v.shape == (spec.dv,) == (6,) and v.dtype == torch.float32
    d = spec.decode(v)
    assert d["composition"]["Al"] == pytest.approx(0.7) and d["composition"]["Mg"] == 0.0
    assert d["inferred"]["theta1"] == pytest.approx(12.5) and d["target"]["T_max"] == pytest.approx(900.0)
    assert spec.groups == {"composition": 3, "inferred": 2, "target": 1}


def test_condspec_validation_errors():
    spec = make_spec()
    ok = ({"Al": 1.0}, {"theta0": 1.0, "theta1": 2.0}, {"T_max": 3.0})
    spec.encode(*ok)
    with pytest.raises(ConditioningError):
        spec.encode({"Al": 0.5}, *ok[1:])  # does not sum to 1
    with pytest.raises(ConditioningError):
        spec.encode({"Al": -0.1, "Cu": 1.1}, *ok[1:])  # negative
    with pytest.raises(ConditioningError):
        spec.encode({"Zz": 1.0}, *ok[1:])  # unknown element
    with pytest.raises(ConditioningError):
        spec.encode(ok[0], {"theta0": 1.0}, ok[2])  # missing
    with pytest.raises(ConditioningError):
        spec.encode(ok[0], {"theta0": 1.0, "theta1": 2.0, "x": 3}, ok[2])  # extra
    with pytest.raises(NonFiniteError):
        spec.encode(ok[0], {"theta0": float("nan"), "theta1": 2.0}, ok[2])
    with pytest.raises(NonFiniteError):
        spec.encode({"Al": float("nan")}, *ok[1:])
    with pytest.raises(ConditioningError):
        spec.encode(ok[0], {"theta0": 3e6, "theta1": 2.0}, ok[2])  # raw un-normalised
    with pytest.raises(ConditioningError):
        make_spec(composition_order=("Al", "Al"))
    with pytest.raises(ConditioningError):
        make_spec(inferred_units=("K",))
    with pytest.raises(ConditioningError):
        make_spec(mean=(0.0,) * 6)
    with pytest.raises(ConditioningError):
        make_spec(mean=(0.0,) * 5, std=(1.0,) * 5)
    with pytest.raises(ConditioningError):
        make_spec(mean=(0.0,) * 6, std=(0.0,) * 6)


def test_condspec_normalisation_fit_allows_large_raw_values_and_roundtrips():
    spec = make_spec()
    rng = np.random.default_rng(0)
    inf = rng.normal([5e5, 1e4], [1e5, 2e3], size=(50, 2))
    tgt = rng.normal([900.0], [50.0], size=(50, 1))
    fitted = spec.with_normalization(inf, tgt)
    assert fitted.is_standardised
    v = fitted.encode({"Al": 1.0}, {"theta0": 6e5, "theta1": 1.1e4}, {"T_max": 950.0})
    assert v[:3].tolist() == [1.0, 0.0, 0.0] and v.abs().max() <= 2.5
    back = fitted.decode(v)
    assert back["inferred"]["theta0"] == pytest.approx(6e5, rel=1e-5)
    assert CondSpec.from_dict(json.loads(fitted.to_json())) == fitted
    with pytest.raises(ConditioningError):
        spec.with_normalization(inf[:1], tgt[:1])
    with pytest.raises(ConditioningError):
        spec.with_normalization(inf[:, :1], tgt)


def test_condspec_batch_and_tensor_checks():
    spec = make_spec()
    item = {"composition": {"Al": 1.0}, "inferred": {"theta0": 1.0, "theta1": 2.0}, "target": {"T_max": 3.0}}
    b = spec.encode_batch([item, item])
    assert spec.check_tensor(b) is b and b.shape == (2, 6)
    with pytest.raises(ConditioningError):
        spec.encode_batch([])
    with pytest.raises(ConditioningError):
        spec.check_tensor(torch.zeros(2, 5))
    with pytest.raises(ConditioningError):
        spec.check_tensor(torch.zeros(6))
    with pytest.raises(ConditioningError):
        spec.check_tensor(torch.zeros(2, 6, dtype=torch.long))
    with pytest.raises(NonFiniteError):
        spec.check_tensor(torch.full((2, 6), float("nan")))
    with pytest.raises(ConditioningError):
        spec.decode(torch.zeros(5))


def test_empty_batch_rejected():
    with pytest.raises(ContractError):
        CandidateDesign("darcy", torch.zeros(0, 2, 8, 8), "model", 0.1)
