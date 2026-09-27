"""Recompute the headline verification numbers quoted in docs/VERIFICATION.md  ->  docs/evidence.json

Usage: python scripts/collect_evidence.py [--skip-pinn]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from squad1.applications import CoolingPlate, LayoutConfig, optimize_layout
from squad1.contracts import CandidateDesign
from squad1.geometry import TopologyProblem, analyse, build_lattice, build_solid, optimize
from squad1.physics import DarcyPhysics, get_physics
from squad1.physics.darcy import darcy_residual, solve_pressure
from squad1.physics.domains import PHYSICS_CLASSES
from squad1.projection import get_projector

ROOT = Path(__file__).resolve().parents[1]


def darcy_analytic() -> dict:
    H = 24
    h = 1 / (H - 1)
    xs = torch.linspace(0, 1, H, dtype=torch.float64)
    X, _ = torch.meshgrid(xs, xs, indexing="ij")
    one = torch.ones(1, H, H, dtype=torch.float64)
    return {
        "linear_p_max_abs_residual": float(darcy_residual(one, (1 - X)[None], h).abs().max()),
        "p_eq_x2_mean_residual (exact 2)": float(darcy_residual(one, (X**2)[None], h).mean()),
        "variable_k_exact_solution_max_scaled_residual": float(
            (darcy_residual((1 + X)[None], (1 - torch.log(1 + X) / np.log(2))[None], h) * h * h).abs().max()
        ),
    }


def domains() -> dict:
    out = {}
    for name, cls in sorted(PHYSICS_CLASSES.items()):
        P = cls()
        x = P.sample(4, 24, seed=0).double()
        rnd = torch.randn_like(x) * x.std()
        out[name] = {
            "data_residual_rms": float(P.residual_rms(x, 1 / 23).max()),
            "noise_residual_rms": float(P.residual_rms(rnd, 1 / 23).min()),
        }
    return out


def projection_comparison() -> dict:
    H = 16
    h = 1 / (H - 1)
    g = torch.Generator().manual_seed(0)
    k = torch.exp(0.6 * torch.randn(4, H, H, generator=g, dtype=torch.float64))
    k = torch.nn.functional.avg_pool2d(k[:, None], 3, 1, 1)[:, 0]
    truth = torch.stack([k, solve_pressure(k, h)], 1)
    rows = {}
    for noise in (0.02, 0.1, 0.3):
        bad = truth.clone()
        bad[:, 1] += noise * torch.randn(4, H, H, generator=g, dtype=torch.float64)
        phys = DarcyPhysics()
        row = {"loss_before": float(phys.loss(bad, h).mean())}
        for name, kw in (
            ("gradient", {"max_iters": 300}),
            ("residual_weighted", {"max_iters": 300}),
            ("gauss_newton", {}),
            ("solve", {}),
        ):
            t0 = time.perf_counter()
            x, info = get_projector(name, **kw).project(bad, phys, h)
            row[name] = {
                "loss_after": float(info["loss_after"].mean()),
                "rel_l2_to_truth_p": float(
                    ((x[:, 1] - truth[:, 1]).flatten(1).norm(dim=1) / truth[:, 1].flatten(1).norm(dim=1)).mean()
                ),
                "seconds": time.perf_counter() - t0,
            }
        rows[str(noise)] = row
    return rows


def cooling() -> dict:
    H = 24
    h = 1 / (H - 1)
    pl = CoolingPlate()
    k = torch.full((1, H, H), 0.02, dtype=torch.float64)
    for c in (5, 17):
        k[:, :, c : c + 2] = 50.0
    cand = CandidateDesign("cooling_plate", DarcyPhysics().solve(k, h).float(), "physical", h).to_model()
    _out, info = optimize_layout(pl, cand, LayoutConfig(steps=150))
    perm = torch.randperm(H * H, generator=torch.Generator().manual_seed(0))
    shuffled = layout4(H)
    straight = layout4(H, shuffled=False)
    return {
        "energy_balance_error_random_layouts_max": float(
            pl.solve(torch.exp(1.5 * torch.randn(3, H, H, dtype=torch.float64)).clamp(0.02, 50), h)[
                "energy_balance_error"
            ].max()
        ),
        "before": info["report_before"][0],
        "after": info["report_after"][0],
        "same_area_different_layout": {
            "straight": pl.report(straight, h)[0],
            "shuffled_same_pixels": pl.report(shuffled, h)[0],
        },
        "_perm_checksum": int(perm.sum()),
    }


def layout4(H: int, shuffled: bool = True) -> torch.Tensor:
    k = torch.full((1, H, H), 0.02, dtype=torch.float64)
    for i in range(4):
        c = round((i + 0.5) * H / 4 - 1)
        k[:, :, c : c + 2] = 50.0
    if shuffled:
        perm = torch.randperm(H * H, generator=torch.Generator().manual_seed(0))
        k = k.reshape(-1)[perm].reshape(1, H, H)
    return k


def geometry() -> dict:
    g = build_lattice("bcc_cube", 2, 2, 2, 5.0)
    _, rep, st = build_solid(g, 0.4, 0.1)
    naive = float(np.pi * 0.4**2 * g.total_length / g.volume)
    p = TopologyProblem(nelx=60, nely=20, volfrac=0.5, penal=3.0, rmin=1.5, case="mbb")
    r = optimize(p, iters=200)
    q = TopologyProblem(nelx=60, nely=10, case="cantilever", volfrac=0.5, rmin=1.2)
    c, _ = analyse(q, np.ones((10, 60)))
    L, hh = 60.0, 10.0
    beam = L**3 / (3 * hh**3 / 12) + L / (5 / 6 * (1 / (2 * 1.3)) * hh)
    return {
        "lattice_bcc_cube_2x2x2": {
            "unique_struts": len(g.struts),
            "relative_density_union": rep.relative_density,
            "naive_sum_of_cylinders": naive,
            "components": rep.components,
            "watertight": st.watertight,
        },
        "mbb_60x20_vf0.5": {
            "compliance": r.compliance[-1],
            "iterations": r.iterations,
            "converged": r.converged,
            "volume": r.volume,
            "grayness": r.grayness,
        },
        "cantilever_vs_timoshenko_beam_ratio": c / beam,
    }


def pinn() -> dict:
    from squad1.pinn import Heat1D, TrainerConfig, train

    r = train(Heat1D(), cfg=TrainerConfig(adam_steps=2500, lbfgs_steps=15, seed=0, log_every=500))
    return {
        "heat1d_rel_l2_vs_exact": r.rel_l2,
        "seconds": r.seconds,
        "note": "original framework path on the same problem: rel-L2 0.970 with 'PASSED VALIDATION'",
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-pinn", action="store_true")
    ap.add_argument("--out", default=str(ROOT / "docs" / "evidence.json"))
    a = ap.parse_args()
    ev = {
        "darcy_analytic": darcy_analytic(),
        "domain_library_residuals": domains(),
        "projection_comparison_16x16": projection_comparison(),
        "cooling_plate": cooling(),
        "geometry": geometry(),
    }
    if not a.skip_pinn:
        ev["pinn"] = pinn()
    ev["environment"] = {"torch": torch.__version__, "numpy": np.__version__}
    Path(a.out).write_text(json.dumps(ev, indent=2, default=str), encoding="utf-8")
    print(f"wrote {a.out}")
    # silence unused import warnings for optional helpers kept for interactive use
    _ = get_physics


if __name__ == "__main__":
    main()
