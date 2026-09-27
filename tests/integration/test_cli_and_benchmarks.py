import json

import pytest

from squad1.cli import build_parser, main
from squad1.pipeline import benchmarks as bm


def test_info_and_selftest(capsys):
    assert main(["info"]) == 0
    info = json.loads(capsys.readouterr().out)
    assert "darcy" in info["domains"] and info["squad1_version"] == "1.0.0"
    assert main(["selftest"]) == 0
    out = capsys.readouterr().out
    assert out.count("[PASS]") == 6 and "[FAIL]" not in out


def test_parser_rejects_unknown_command():
    with pytest.raises(SystemExit):
        build_parser().parse_args(["nope"])
    with pytest.raises(SystemExit):
        main(["--version"])


@pytest.mark.slow
def test_demo_writes_a_valid_package(tmp_path, capsys):
    rc = main(["demo", "--out", str(tmp_path / "d"), "--size", "12", "--n", "2", "--train-steps", "40"])
    assert rc == 0
    d = json.loads((tmp_path / "d" / "package.json").read_text())
    assert d["schema_version"] == "squad1_design_package_v1" and len(d["accepted"]) == 2


@pytest.mark.slow
def test_gpu_suite_smoke_runs_all_experiments_on_cpu(tmp_path):
    out = tmp_path / "r.json"
    assert main(["gpu-suite", "--smoke", "--out", str(out), "--device", "cpu"]) == 0
    doc = json.loads(out.read_text())
    assert doc["schema_version"] == "squad1_gpu_suite_v1" and set(doc["results"]) == {f"G{i}" for i in range(1, 9)}
    assert all(v["status"] == "ok" for v in doc["results"].values())
    r = doc["results"]
    assert r["G2"]["rows"][0]["loss_after_max"] < 1e-10 and r["G2"]["rows"][1]["loss_after_max"] < 1e-10
    assert r["G7"]["sampling_max_abs_diff"] == 0 and r["G7"]["projection_max_abs_diff"] == 0
    assert r["G8"]["ok"] and r["G3"]["rows"][0]["compute_dtype"] == "float64"


def test_single_experiments_return_expected_shapes():
    import torch

    dev = torch.device("cpu")
    g1 = bm.g1_physics_throughput(dev, [12], [2], [torch.float32])
    assert g1["rows"][0]["ms_per_sample"] > 0
    g4 = bm.g4_encoder(dev, 2, 32, 8)
    assert g4["params_m"] > 0 and g4["dv"] == 42
    g6 = bm.g6_sampling(dev, 8, 2, 3)
    assert g6["finite"]
