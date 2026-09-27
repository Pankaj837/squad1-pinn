"""GPU / device benchmark suite (G1-G8) — emits one JSON document per run.

``run_suite(smoke=True)`` uses tiny sizes so it also runs (and is tested) on CPU; on a GPU machine call
``squad1 gpu-suite --out results.json`` for the full sizes. Every experiment is wrapped: a failure is recorded as
``status: "error"`` with the exception text instead of aborting the whole run.
"""

from __future__ import annotations

import functools
import json
import time
import traceback
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch

from squad1.contracts.candidate import CandidateDesign
from squad1.contracts.channels import ChannelNormalizer
from squad1.generation.data import darcy_fields
from squad1.generation.diffusion import DiffusionScheduler
from squad1.generation.dit import DiT
from squad1.generation.sampler import sample
from squad1.physics import DarcyPhysics
from squad1.pinn import Heat1D, TrainerConfig, train
from squad1.projection import PCFMPipeline, get_projector
from squad1.utils.seed import provenance, seed_everything


def _sync(dev: torch.device) -> None:
    if dev.type == "cuda":  # pragma: no cover - GPU only
        torch.cuda.synchronize(dev)


def _timed(dev: torch.device, fn: Callable[[], Any], repeat: int = 3) -> tuple[Any, float]:
    out = fn()  # warm-up
    _sync(dev)
    t0 = time.perf_counter()
    for _ in range(repeat):
        out = fn()
    _sync(dev)
    return out, (time.perf_counter() - t0) / repeat


def _peak_mb(dev: torch.device) -> float | None:
    return torch.cuda.max_memory_allocated(dev) / 2**20 if dev.type == "cuda" else None  # pragma: no cover


def _reset_mem(dev: torch.device) -> None:
    if dev.type == "cuda":  # pragma: no cover
        torch.cuda.reset_peak_memory_stats(dev)


def _darcy_batch(n: int, size: int, dev: torch.device, dtype: torch.dtype) -> torch.Tensor:
    return darcy_fields(n, size, seed=0).to(dev, dtype)


# ------------------------------------------------------------------------------------------------- G1
def g1_physics_throughput(
    dev: torch.device, sizes: list[int], batches: list[int], dtypes: list[torch.dtype]
) -> dict[str, Any]:
    rows = []
    phys = DarcyPhysics()
    for s in sizes:
        for b in batches:
            for dt in dtypes:
                x = _batch_random(b, s, dev, dt)
                h = 1.0 / (s - 1)
                _reset_mem(dev)
                _, t_loss = _timed(dev, functools.partial(phys.loss, x, h))
                _, t_grad = _timed(dev, functools.partial(phys.gradient, x, h))
                rows.append(
                    {
                        "size": s,
                        "batch": b,
                        "dtype": str(dt).split(".")[-1],
                        "loss_ms": 1e3 * t_loss,
                        "grad_ms": 1e3 * t_grad,
                        "ms_per_sample": 1e3 * t_grad / b,
                        "peak_mb": _peak_mb(dev),
                    }
                )
    return {"rows": rows}


def _batch_random(n: int, size: int, dev: torch.device, dtype: torch.dtype) -> torch.Tensor:
    g = torch.Generator().manual_seed(0)
    k = torch.exp(0.5 * torch.randn(n, 1, size, size, generator=g))
    p = torch.rand(n, 1, size, size, generator=g)
    return torch.cat([k, p], 1).to(dev, dtype)


# ------------------------------------------------------------------------------------------------- G2
def g2_projection(dev: torch.device, size: int, batch: int) -> dict[str, Any]:
    phys = DarcyPhysics()
    h = 1.0 / (size - 1)
    x0 = _darcy_batch(batch, size, dev, torch.float64)
    x0[:, 1] += 0.1 * torch.randn_like(x0[:, 1])
    rows = []
    for name, cfg in (
        ("gauss_newton_dense", ("gauss_newton", {"linear_solver": "dense"})),
        ("gauss_newton_cg", ("gauss_newton", {"linear_solver": "cg", "cg_tol": 1e-8, "cg_max_iter": 4000})),
        ("gradient_300", ("gradient", {"max_iters": 300})),
    ):
        try:
            _reset_mem(dev)
            proj = get_projector(cfg[0], **cfg[1])
            t0 = time.perf_counter()
            _, info = proj.project(x0, phys, h)
            _sync(dev)
            rows.append(
                {
                    "method": name,
                    "seconds": time.perf_counter() - t0,
                    "loss_after_max": float(info["loss_after"].max()),
                    "iterations": info.get("iterations"),
                    "peak_mb": _peak_mb(dev),
                    "status": "ok",
                }
            )
        except Exception as exc:
            rows.append({"method": name, "status": "error", "error": f"{type(exc).__name__}: {exc}"})
    return {"size": size, "batch": batch, "loss_before_min": float(phys.loss(x0, h).min()), "rows": rows}


# ------------------------------------------------------------------------------------------------- G3
def g3_precision(dev: torch.device, size: int, batch: int) -> dict[str, Any]:
    phys = DarcyPhysics()
    h = 1.0 / (size - 1)
    x64 = _darcy_batch(batch, size, dev, torch.float64)
    x64[:, 1] += 0.1 * torch.randn_like(x64[:, 1])
    rows = []
    for dt in (torch.float64, torch.float32):
        x, _ = get_projector("gauss_newton").project(x64.to(dt), phys, h)
        rows.append(
            {
                "compute_dtype": str(dt).split(".")[-1],
                "residual_rms_float64_eval": float(phys.residual_rms(x.double(), h).max()),
            }
        )
    with torch.autocast(device_type=dev.type, dtype=torch.bfloat16):
        x_ref = x64.float()
        r_bf = phys.loss(x_ref, h)
    r_32 = phys.loss(x64.float(), h)
    return {
        "rows": rows,
        "bf16_vs_fp32_loss_rel_diff": float(((r_bf.float() - r_32).abs() / r_32.abs().clamp_min(1e-30)).max()),
    }


# ------------------------------------------------------------------------------------------------- G4
def g4_encoder(dev: torch.device, batch: int, embed: int, seq: int, dv: int = 42) -> dict[str, Any]:
    from squad1.encoding import ConditioningToTokens

    m = ConditioningToTokens(
        {"composition": 30, "inferred": max(dv - 34, 1), "target": 4},
        vocab_size=64,
        embed_dim=embed,
        max_seq_len=seq,
        num_heads=8 if embed % 8 == 0 else 4,
        num_layers=2,
        ff_dim=2 * embed,
    ).to(dev)
    x = torch.randn(batch, m.ctx.dv, device=dev)
    _reset_mem(dev)

    def step() -> None:
        m.zero_grad()
        m(x)[2].sum().backward()

    _, t = _timed(dev, step, repeat=2)
    return {
        "batch": batch,
        "embed": embed,
        "seq": seq,
        "dv": m.ctx.dv,
        "fwd_bwd_ms": 1e3 * t,
        "peak_mb": _peak_mb(dev),
        "params_m": sum(p.numel() for p in m.parameters()) / 1e6,
    }


# ------------------------------------------------------------------------------------------------- G5
def g5_pinn_seeds(seeds: list[int], steps: int) -> dict[str, Any]:
    errs = []
    for s in seeds:
        r = train(Heat1D(), cfg=TrainerConfig(adam_steps=steps, lbfgs_steps=0, seed=s, log_every=max(steps // 2, 1)))
        errs.append(r.rel_l2)
    t = torch.tensor(errs, dtype=torch.float64)
    return {
        "seeds": seeds,
        "steps": steps,
        "rel_l2": errs,
        "mean": float(t.mean()),
        "std": float(t.std(unbiased=False)),
    }


# ------------------------------------------------------------------------------------------------- G6
def g6_sampling(dev: torch.device, size: int, batch: int, steps: int) -> dict[str, Any]:
    seed_everything(0)
    model = DiT(size, 2, 2, hidden_size=128, depth=4, num_heads=4).to(dev)
    sched = DiffusionScheduler(100)
    _reset_mem(dev)
    out, t = _timed(dev, lambda: sample(model, sched, batch, steps=steps, seed=0), repeat=1)
    return {
        "size": size,
        "batch": batch,
        "ddim_steps": steps,
        "seconds": t,
        "peak_mb": _peak_mb(dev),
        "finite": bool(torch.isfinite(out).all()),
        "note": "random-init DiT: timing/memory only, not sample quality",
    }


# ------------------------------------------------------------------------------------------------- G7
def g7_determinism(dev: torch.device, size: int) -> dict[str, Any]:
    model = DiT(size, 2, 2, hidden_size=32, depth=2, num_heads=4).to(dev)
    for p in model.parameters():
        if p.ndim > 1:
            torch.nn.init.normal_(p, std=0.05)
    sched = DiffusionScheduler(50)
    a = sample(model, sched, 2, steps=5, seed=1)
    b = sample(model, sched, 2, steps=5, seed=1)
    proj = PCFMPipeline()
    cand = CandidateDesign("darcy", a, "model", 1.0 / (size - 1))
    p1, p2 = proj.project(cand).projected.tensor, proj.project(cand).projected.tensor
    return {
        "sampling_max_abs_diff": float((a - b).abs().max()),
        "projection_max_abs_diff": float((p1 - p2).abs().max()),
    }


# ------------------------------------------------------------------------------------------------- G8
def g8_e2e(size: int, train_steps: int) -> dict[str, Any]:
    from squad1.pipeline.e2e import E2EConfig, Squad1Pipeline, build_toy_generator, summarize

    model, sched, hist = build_toy_generator(
        "darcy", size, train_steps=train_steps, n_train=48, seed=0, hidden=32, depth=2
    )
    pkg = Squad1Pipeline(model, sched, config=E2EConfig(n=2, steps=8, seed=0)).run()
    s = summarize(pkg)
    return {
        **s,
        "train_loss_first": hist["loss"][0],
        "train_loss_last": hist["loss"][-1],
        "ok": s["converged"] == 2 and s["rejected"] == 0,
    }


# ---------------------------------------------------------------------------------------------- runner
def run_suite(out: str | Path | None = None, smoke: bool = True, device: str = "auto") -> dict[str, Any]:
    dev = torch.device(
        "cuda" if (device == "auto" and torch.cuda.is_available()) else ("cpu" if device == "auto" else device)
    )
    if smoke:
        cfgs = {
            "G1": lambda: g1_physics_throughput(dev, [16], [2], [torch.float64, torch.float32]),
            "G2": lambda: g2_projection(dev, 8, 2),
            "G3": lambda: g3_precision(dev, 16, 2),
            "G4": lambda: g4_encoder(dev, 2, 32, 16),
            "G5": lambda: g5_pinn_seeds([0, 1], 40),
            "G6": lambda: g6_sampling(dev, 16, 2, 4),
            "G7": lambda: g7_determinism(dev, 16),
            "G8": lambda: g8_e2e(12, 30),
        }
    else:  # pragma: no cover - full sizes for GPU hardware
        cfgs = {
            "G1": lambda: g1_physics_throughput(dev, [64, 128, 256], [1, 8, 64], [torch.float32, torch.float64]),
            "G2": lambda: g2_projection(dev, 32, 8),
            "G3": lambda: g3_precision(dev, 32, 8),
            "G4": lambda: g4_encoder(dev, 64, 768, 128),
            "G5": lambda: g5_pinn_seeds([0, 1, 2], 3000),
            "G6": lambda: g6_sampling(dev, 64, 8, 50),
            "G7": lambda: g7_determinism(dev, 32),
            "G8": lambda: g8_e2e(16, 300),
        }
    results: dict[str, Any] = {}
    for name, fn in cfgs.items():
        t0 = time.perf_counter()
        try:
            results[name] = {"status": "ok", "seconds": None, **fn()}
        except Exception as exc:
            results[name] = {
                "status": "error",
                "error": f"{type(exc).__name__}: {exc}",
                "trace": traceback.format_exc(limit=3),
            }
        results[name]["seconds"] = time.perf_counter() - t0
    doc = {
        "schema_version": "squad1_gpu_suite_v1",
        "smoke": smoke,
        "device": str(dev),
        "provenance": provenance({"smoke": smoke}, 0),
        "results": results,
    }
    if out is not None:
        p = Path(out)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(doc, indent=2, default=str), encoding="utf-8")
    return doc


__all__ = [
    "g1_physics_throughput",
    "g2_projection",
    "g3_precision",
    "g4_encoder",
    "g5_pinn_seeds",
    "g6_sampling",
    "g7_determinism",
    "g8_e2e",
    "run_suite",
]
_ = ChannelNormalizer  # re-exported for convenience of notebooks that time normalisation
