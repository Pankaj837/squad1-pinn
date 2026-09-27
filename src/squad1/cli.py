"""Command line: ``squad1 info | selftest | demo | gpu-suite`` (also ``python -m squad1``)."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence

import torch

from squad1 import __version__


def _cmd_info(_args: argparse.Namespace) -> int:
    from squad1.contracts import available_domains
    from squad1.utils.seed import provenance

    print(json.dumps({**provenance(), "domains": available_domains()}, indent=2))
    return 0


def _cmd_selftest(_args: argparse.Namespace) -> int:
    """Fast end-to-end sanity checks (no data, no GPU, < 30 s)."""
    from squad1.contracts import CandidateDesign, ChannelNormalizer, CondSpec
    from squad1.encoding import FormulaTokenizer
    from squad1.geometry import build_lattice, build_solid
    from squad1.physics import DarcyPhysics
    from squad1.physics.darcy import solve_pressure
    from squad1.projection import PCFMPipeline

    checks: list[tuple[str, bool]] = []
    torch.manual_seed(0)
    n = ChannelNormalizer.for_domain("darcy")
    x = torch.rand(2, 2, 8, 8) * 2 - 1
    checks.append(("normaliser round-trip", bool(torch.allclose(n.to_model(n.to_physical(x)), x, atol=1e-5))))
    k = torch.exp(0.5 * torch.randn(2, 12, 12, dtype=torch.float64))
    p = solve_pressure(k, 1 / 11)
    checks.append(
        (
            "darcy solve has zero residual",
            float(DarcyPhysics().loss(torch.stack([k, p], 1), 1 / 11).max()) < 1e-16,
        )
    )
    bad = torch.stack([k, p + 0.1 * torch.randn_like(p)], 1).float()
    cand = CandidateDesign("darcy", bad, "physical", 1 / 11)
    res = PCFMPipeline().project(cand)
    checks.append(
        (
            "hard projection converges",
            res.all_converged and float(res.residual_rms_after.max()) < 1e-3 * float(res.residual_rms_before.min()),
        )
    )
    spec = CondSpec(("A", "B"), ("t",), ("K",), ("T",), ("K",))
    v = spec.encode({"A": 0.5, "B": 0.5}, {"t": 1.0}, {"T": 2.0})
    checks.append(("cond vector round-trip", abs(spec.decode(v)["target"]["T"] - 2.0) < 1e-6))
    tok = FormulaTokenizer().build_vocab(["HfC-SiC"])
    checks.append(("formula tokenizer keeps Hf", tok.decode(tok.encode("HfC-SiC")) == "HfC-SiC"))
    _, rep, st = build_solid(build_lattice("bcc", 2, 2, 2, 4.0), 0.6, 0.2, min_feature=0.5)
    checks.append(("lattice solid is one watertight body", rep.all_passed and st.watertight))
    for name, ok in checks:
        print(f"[{'PASS' if ok else 'FAIL'}] {name}")
    return 0 if all(ok for _, ok in checks) else 1


def _cmd_demo(args: argparse.Namespace) -> int:
    from squad1.pipeline import E2EConfig, Squad1Pipeline, build_toy_generator, summarize

    print(f"training a toy Darcy generator ({args.train_steps} steps, {args.size}x{args.size}) ...")
    model, sched, hist = build_toy_generator(
        "darcy", args.size, train_steps=args.train_steps, n_train=96, seed=args.seed, hidden=48, depth=2
    )
    pkg = Squad1Pipeline(model, sched, config=E2EConfig(n=args.n, steps=15, seed=args.seed)).run()
    path = pkg.save(args.out)
    print(
        json.dumps(
            {**summarize(pkg), "train_loss": [hist["loss"][0], hist["loss"][-1]], "package": str(path)},
            indent=2,
        )
    )
    return 0


def _cmd_gpu_suite(args: argparse.Namespace) -> int:
    from squad1.pipeline.benchmarks import run_suite

    doc = run_suite(args.out, smoke=args.smoke, device=args.device)
    bad = [k for k, v in doc["results"].items() if v["status"] != "ok"]
    print(
        f"device={doc['device']} smoke={doc['smoke']} experiments={len(doc['results'])} "
        f"errors={bad or 'none'} -> {args.out}"
    )
    return 1 if bad else 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="squad1", description=__doc__)
    p.add_argument("--version", action="version", version=f"squad1 {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("info", help="versions, device, registered domains").set_defaults(fn=_cmd_info)
    sub.add_parser("selftest", help="fast sanity checks").set_defaults(fn=_cmd_selftest)
    d = sub.add_parser("demo", help="train a toy generator, sample, project, write a design package")
    d.add_argument("--out", default="squad1_demo_out")
    d.add_argument("--size", type=int, default=16)
    d.add_argument("--n", type=int, default=4)
    d.add_argument("--train-steps", type=int, default=250)
    d.add_argument("--seed", type=int, default=0)
    d.set_defaults(fn=_cmd_demo)
    g = sub.add_parser("gpu-suite", help="benchmark suite G1-G8 -> JSON")
    g.add_argument("--out", default="gpu_suite_results.json")
    g.add_argument("--device", default="auto")
    g.add_argument("--smoke", action="store_true", help="tiny sizes (CPU friendly); default is full size")
    g.set_defaults(fn=_cmd_gpu_suite)
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.fn(args))


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
