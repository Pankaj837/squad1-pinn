"""End-to-end Squad 1 flow:  conditioning -> generator -> quality gate -> PCFM projection -> design package.

cond items (chemistry + inferred params + targets) --CondSpec--> cond_vec (B, Dv)
boundary spec --rasterize--> cond_field (B, Cc, H, W)
DiT.sample(cond_vec, cond_field)  -> CandidateDesign (model space)
quality_check                     -> per-sample gate
PCFMPipeline.project              -> physics-valid candidate | rejected (unchanged)
DesignPackage                     -> JSON contract + .npy tensors + provenance
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

from squad1 import __version__
from squad1.conditioning.boundary import COND_FIELD_CHANNELS, BoundarySpec, cond_field_batch
from squad1.contracts.candidate import CandidateDesign, ProjectionResult
from squad1.contracts.channels import ChannelNormalizer, get_domain
from squad1.contracts.cond import CondSpec
from squad1.errors import ConditioningError, ContractError
from squad1.generation.data import make_dataset
from squad1.generation.diffusion import DiffusionScheduler
from squad1.generation.dit import DiT
from squad1.generation.quality import quality_check
from squad1.generation.sampler import generate_candidates
from squad1.generation.train import TrainConfig, train_diffusion
from squad1.projection.pipeline import PCFMPipeline, physics_for
from squad1.utils.seed import provenance, seed_everything


@dataclass
class E2EConfig:
    domain: str = "darcy"
    n: int = 4
    steps: int = 20  # DDIM steps
    cfg_scale: float = 0.0
    physics_scale: float = 0.0
    method: str = "gauss_newton"
    max_correction_rms: float | None = None
    projector_config: dict[str, Any] = field(default_factory=dict)
    seed: int = 0
    max_residual_rms_gate: float | None = None  # quality gate on the *raw* generator output; None = report only


@dataclass
class DesignPackage:
    config: E2EConfig
    candidate_raw: CandidateDesign
    result: ProjectionResult
    quality: dict[str, Any]
    cond_spec: CondSpec | None
    provenance: dict[str, Any]

    @property
    def final(self) -> CandidateDesign:
        return self.result.projected

    def to_dict(self) -> dict[str, Any]:
        r = self.result.to_dict()
        return {
            "schema_version": "squad1_design_package_v1",
            "squad1_version": __version__,
            "domain": self.final.domain,
            "representation": self.final.representation,
            "tensor_shape": list(self.final.tensor.shape),
            "channels": list(self.final.spec.names),
            "h": self.final.h,
            "config": asdict(self.config),
            "quality_gate_raw": self.quality,
            "projection": r,
            "accepted": [bool(c and not rj) for c, rj in zip(r["converged"], r["rejected"])],
            "cond_spec": None if self.cond_spec is None else self.cond_spec.to_dict(),
            "provenance": self.provenance,
        }

    def save(self, out_dir: str | Path) -> Path:
        d = Path(out_dir)
        d.mkdir(parents=True, exist_ok=True)
        np.save(d / "candidate_raw.npy", self.candidate_raw.tensor.numpy())
        np.save(d / "design_final.npy", self.final.tensor.numpy())
        (d / "package.json").write_text(json.dumps(self.to_dict(), indent=2, default=str), encoding="utf-8")
        return d / "package.json"


def darcy_bc_from_boundary(spec: BoundarySpec | None) -> dict[str, float]:
    """Extract ``p_left`` / ``p_right`` from full-side Dirichlet segments on ``x_min`` / ``x_max`` (defaults 1 / 0)."""
    bc: dict[str, float] = {}
    if spec is None:
        return bc
    for seg in spec.segments:
        if seg.kind == "dirichlet" and seg.start == 0.0 and seg.end == 1.0:
            if seg.side == "x_min":
                bc["p_left"] = float(seg.value)  # type: ignore[arg-type]
            elif seg.side == "x_max":
                bc["p_right"] = float(seg.value)  # type: ignore[arg-type]
    return bc


class Squad1Pipeline:
    def __init__(
        self,
        model: DiT,
        scheduler: DiffusionScheduler,
        cond_spec: CondSpec | None = None,
        config: E2EConfig | None = None,
    ):
        self.model, self.scheduler, self.cond_spec = model, scheduler, cond_spec
        self.cfg = config or E2EConfig()
        spec = get_domain(self.cfg.domain)
        if spec.n_channels != model.in_channels:
            raise ContractError(
                f"domain {self.cfg.domain!r} has {spec.n_channels} channels; model has {model.in_channels}"
            )
        if cond_spec is not None and model.cond_dim != cond_spec.dv:
            raise ConditioningError(
                f"model cond_dim {model.cond_dim} != CondSpec.dv {cond_spec.dv} (frozen interface mismatch)"
            )
        if cond_spec is None and model.cond_dim:
            raise ConditioningError("model expects conditioning but no CondSpec was given")
        if model.field_channels not in (0, len(COND_FIELD_CHANNELS)):
            raise ConditioningError(
                f"model field_channels={model.field_channels}; the boundary contract has Cc={len(COND_FIELD_CHANNELS)}"
            )

    def run(
        self,
        cond_items: Sequence[Mapping[str, Mapping[str, float]]] | torch.Tensor | None = None,
        boundary: BoundarySpec | Sequence[BoundarySpec] | None = None,
    ) -> DesignPackage:
        cfg, model = self.cfg, self.model
        seed_everything(cfg.seed)
        cond_vec = None
        if not model.cond_dim and cond_items is not None:
            raise ConditioningError("model was built with cond_dim=0: the given conditioning would be ignored")
        if model.cond_dim:
            if cond_items is None:
                raise ConditioningError("this model needs conditioning: pass cond_items (dicts) or a (B, Dv) tensor")
            if isinstance(cond_items, torch.Tensor):
                cond_vec = self.cond_spec.check_tensor(cond_items) if self.cond_spec else cond_items
            else:
                cond_vec = self.cond_spec.encode_batch(cond_items)  # type: ignore[union-attr]
            n = cond_vec.shape[0]
        else:
            n = cfg.n
        cond_field = None
        if model.field_channels:
            if boundary is None:
                raise ConditioningError("this model needs a cond_field: pass a BoundarySpec")
            cond_field = cond_field_batch(boundary, model.img_size, n if isinstance(boundary, BoundarySpec) else None)
            if cond_field.shape[0] != n:
                raise ConditioningError(f"boundary batch {cond_field.shape[0]} != n {n}")
        first = boundary if isinstance(boundary, BoundarySpec) else (boundary[0] if boundary else None)
        bc_meta = darcy_bc_from_boundary(first) if cfg.domain in ("darcy", "cooling_plate", "darcy_biot") else {}
        h = 1.0 / (model.img_size - 1)
        spec = get_domain(cfg.domain)
        physics = physics_for(
            CandidateDesign(
                cfg.domain,
                torch.zeros(1, spec.n_channels, model.img_size, model.img_size),
                "model",
                h,
                boundary=bc_meta,
            )
        )
        guided = cfg.physics_scale > 0
        cand = generate_candidates(
            model,
            self.scheduler,
            cfg.domain,
            n,
            h,
            boundary=bc_meta,
            seed=cfg.seed,
            steps=cfg.steps,
            cond_vec=cond_vec,
            cond_field=cond_field,
            cfg_scale=cfg.cfg_scale,
            physics=physics if guided else None,
            normalizer=ChannelNormalizer(spec.channels) if guided else None,
            physics_scale=cfg.physics_scale,
        )
        gate = quality_check(cand, physics, max_residual_rms=cfg.max_residual_rms_gate)
        pipe = PCFMPipeline(cfg.method, cfg.max_correction_rms, physics=physics, **cfg.projector_config)
        res = pipe.project(cand)
        prov = provenance(
            {"e2e": asdict(cfg), "cond": None if self.cond_spec is None else self.cond_spec.to_dict()},
            cfg.seed,
        )
        return DesignPackage(cfg, cand, res, gate, self.cond_spec, prov)


def build_toy_generator(
    domain: str = "darcy",
    size: int = 16,
    train_steps: int = 300,
    n_train: int = 128,
    seed: int = 0,
    cond_dim: int = 0,
    hidden: int = 64,
    depth: int = 3,
) -> tuple[DiT, DiffusionScheduler, dict[str, list[float]]]:
    """Train a small DiT on on-manifold synthetic data (CPU-friendly demo/CI generator; replace with the real DiT)."""
    data, _ = make_dataset(domain, n_train, size, seed=seed)
    scheduler = DiffusionScheduler(100)
    seed_everything(seed)
    model = DiT(size, 2, data.shape[1], hidden_size=hidden, depth=depth, num_heads=4, cond_dim=cond_dim)
    ema, hist = train_diffusion(
        model,
        scheduler,
        data,
        TrainConfig(steps=train_steps, batch_size=32, lr=2e-3, seed=seed, log_every=max(train_steps // 6, 1)),
    )
    return ema, scheduler, hist


def summarize(pkg: DesignPackage) -> dict[str, Any]:
    r = pkg.result
    return {
        "domain": pkg.final.domain,
        "n": pkg.final.batch_size,
        "raw_residual_rms_mean": float(r.residual_rms_before.mean()),
        "projected_residual_rms_max": float(r.residual_rms_after.max()),
        "converged": int(r.converged.sum()),
        "rejected": int(r.rejected.sum()),
        "correction_rms_mean": float(r.correction_rms.mean()),
        "method": r.method,
        "runtime_s": r.runtime_s,
    }
