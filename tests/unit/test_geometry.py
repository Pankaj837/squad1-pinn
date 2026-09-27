import numpy as np
import pandas as pd
import pytest
import trimesh

from squad1.errors import ConditioningError, ContractError
from squad1.geometry import (
    TopologyProblem,
    analyse,
    build_lattice,
    build_solid,
    element_stiffness,
    load_material,
    mesh_stats,
    optimize,
    rasterize,
    read_binary_stl,
    remove_edge_contacts,
    voxels_to_mesh,
    write_binary_stl,
)


# ---------------------------------------------------------------------- voxel mesh
def test_single_voxel_is_unit_cube():
    m = voxels_to_mesh(np.ones((1, 1, 1), bool))
    st = mesh_stats(m)
    assert (st.n_vertices, st.n_faces, st.watertight, st.components) == (8, 12, True, 1)
    assert st.volume == pytest.approx(1.0)


def test_volume_matches_voxel_count_scaling_and_orientation():
    rng = np.random.default_rng(0)
    occ = rng.random((6, 5, 4)) > 0.45
    m = voxels_to_mesh(occ, spacing=0.5)
    st = mesh_stats(m)
    filled = remove_edge_contacts(occ)
    assert st.volume == pytest.approx(filled.sum() * 0.125) and st.volume > 0 and st.watertight


def test_edge_contact_is_repaired_to_manifold():
    occ = np.zeros((2, 2, 1), bool)
    occ[0, 0, 0] = occ[1, 1, 0] = True  # two voxels touching only along an edge
    st = mesh_stats(voxels_to_mesh(occ))
    assert st.watertight and st.components == 1


def test_agrees_with_trimesh_on_random_solid():
    rng = np.random.default_rng(1)
    occ = np.pad(rng.random((7, 7, 7)) > 0.4, 1)
    m = voxels_to_mesh(occ)
    t = trimesh.Trimesh(m.vertices, m.faces, process=False)
    st = mesh_stats(m)
    assert t.is_watertight == st.watertight and t.volume == pytest.approx(st.volume) and t.is_winding_consistent


def test_stl_roundtrip_and_errors(tmp_path):
    m = voxels_to_mesh(np.ones((3, 2, 2), bool), 0.25)
    p = write_binary_stl(tmp_path / "a" / "x.stl", m)
    back = read_binary_stl(p)
    assert mesh_stats(back).volume == pytest.approx(mesh_stats(m).volume, rel=1e-5) and mesh_stats(back).watertight
    t = trimesh.load(p)
    assert t.is_watertight and t.volume > 0
    p.write_bytes(p.read_bytes()[:-3])
    with pytest.raises(ContractError):
        read_binary_stl(p)
    with pytest.raises(ContractError):
        voxels_to_mesh(np.zeros((2, 2, 2), bool))
    with pytest.raises(ContractError):
        voxels_to_mesh(np.ones((2, 2), bool))


# ----------------------------------------------------------------------- lattice
@pytest.mark.parametrize(
    ("kind", "struts_per_cell_unique_3x3x3"),
    [("simple_cubic", 144), ("bcc", 216), ("bcc_cube", 360), ("octet", 0)],
)
def test_graph_is_deduplicated(kind, struts_per_cell_unique_3x3x3):
    g = build_lattice(kind, 3, 3, 3, 5.0)
    assert len(np.unique(g.struts, axis=0)) == len(g.struts)
    if struts_per_cell_unique_3x3x3:
        assert len(g.struts) == struts_per_cell_unique_3x3x3
    assert g.volume == pytest.approx(15.0**3)


def test_octet_counts_match_theory():
    g = build_lattice("octet", 1, 1, 1, 1.0)
    assert len(g.struts) == 36 and len(g.nodes) == 14
    L = np.linalg.norm(g.nodes[g.struts[:, 0]] - g.nodes[g.struts[:, 1]], axis=1)
    assert np.allclose(L, np.sqrt(2) / 2)


def test_unknown_cell_and_bad_sizes():
    with pytest.raises(ContractError):
        build_lattice("nope", 1, 1, 1, 1.0)
    with pytest.raises(ContractError):
        build_lattice("bcc", 0, 1, 1, 1.0)
    with pytest.raises(ContractError):
        rasterize(build_lattice("bcc", 1, 1, 1, 1.0), 0.0, 0.1)


def test_density_counts_the_union_not_the_sum_of_cylinders():
    g = build_lattice("bcc_cube", 2, 2, 2, 5.0)
    assert len(g.struts) == 118 and len(g.nodes) == 27 + 8  # 54 cube edges + 64 half-diagonals; corners + centres
    r, v = 0.4, 0.1
    _, rep, _ = build_solid(g, r, v, min_feature=0.3)
    naive = np.pi * r**2 * g.total_length / g.volume
    assert rep.n_struts == 118 and rep.n_nodes == 35
    assert 0.8 * naive < rep.relative_density < naive  # joint overlaps removed, nothing else


def test_solid_is_single_watertight_manifold_and_passes_checks():
    g = build_lattice("bcc", 2, 2, 2, 4.0)
    mesh, rep, st = build_solid(g, 0.6, 0.2, min_feature=0.5)
    assert rep.all_passed and st.components == 1 and st.watertight and st.volume > 0
    t = trimesh.Trimesh(mesh.vertices, mesh.faces, process=False)
    assert t.is_watertight and t.volume > 0


def test_checks_fail_when_they_should():
    g = build_lattice("bcc", 2, 2, 2, 4.0)
    _, thin, _ = build_solid(g, 0.1, 0.05, min_feature=0.3)  # printable diameter too small
    assert not thin.min_feature_ok and not thin.all_passed
    _, dense, _ = build_solid(g, 1.6, 0.4, min_feature=0.3, density_range=(0.02, 0.10))
    assert not dense.density_in_range


def test_disconnected_lattice_is_detected():
    g = build_lattice("simple_cubic", 1, 1, 1, 10.0)
    g.struts = g.struts[:3][:1]  # keep a single strut, drop the rest -> still one component
    _, rep, _ = build_solid(g, 0.8, 0.2)
    assert rep.components == 1
    g2 = build_lattice("simple_cubic", 1, 1, 1, 10.0)
    g2.struts = np.array([g2.struts[0], g2.struts[-1]])  # two far-apart struts
    _, rep2, _ = build_solid(g2, 0.8, 0.2)
    assert rep2.components == 2 and not rep2.all_passed


# ---------------------------------------------------------------------- topology
def test_element_stiffness_properties():
    K = element_stiffness(0.3)
    assert np.allclose(K, K.T)
    w = np.linalg.eigvalsh(K)
    assert (w > -1e-12).all() and (np.abs(w) < 1e-10).sum() == 3  # 3 rigid-body modes
    # known value: K11 for nu = 0.3, E = 1
    assert K[0, 0] == pytest.approx(1 / (1 - 0.09) * (0.5 - 0.3 / 6), rel=1e-9)


def test_solid_cantilever_matches_beam_theory():
    nelx, nely = 60, 10
    p = TopologyProblem(nelx=nelx, nely=nely, case="cantilever", volfrac=0.5, E0=1.0, nu=0.3, load=1.0, rmin=1.2)
    c, _ = analyse(p, np.ones((nely, nelx)))
    L, h = float(nelx), float(nely)
    I = h**3 / 12
    delta = L**3 / (3 * 1.0 * I) + L / (5 / 6 * (1.0 / (2 * 1.3)) * h)  # bending + shear (Timoshenko)
    assert c == pytest.approx(delta, rel=0.12)  # compliance = P * delta with P = 1


def test_compliance_scales_inversely_with_modulus_and_linearly_with_load_squared():
    x = np.full((10, 30), 0.6)
    c1, _ = analyse(TopologyProblem(nelx=30, nely=10, case="mbb", E0=1.0, load=1.0), x)
    c2, _ = analyse(TopologyProblem(nelx=30, nely=10, case="mbb", E0=200.0, load=1.0), x)
    c3, _ = analyse(TopologyProblem(nelx=30, nely=10, case="mbb", E0=1.0, load=3.0), x)
    assert c2 == pytest.approx(c1 / 200.0, rel=1e-6) and c3 == pytest.approx(9 * c1, rel=1e-6)


@pytest.mark.slow
def test_mbb_optimisation_converges_to_a_binary_truss_like_design():
    p = TopologyProblem(nelx=60, nely=20, volfrac=0.5, penal=3.0, rmin=1.5, case="mbb")
    r = optimize(p, iters=200)
    assert r.compliance[-1] < 0.5 * r.compliance[0] and r.converged
    assert r.compliance[-1] == pytest.approx(
        218.8, rel=0.03
    )  # published top88 benchmark range (60x20, vf 0.5, density filter)
    assert abs(r.volume - 0.5) < 2e-3
    assert r.grayness < 0.35 and r.density.min() >= 0.0 and r.density.max() <= 1.0
    assert np.all(np.diff(r.compliance[10:]) < 0.02 * r.compliance[10])  # (near-)monotone after warm-up


def test_cantilever_optimisation_short_run_reduces_compliance():
    p = TopologyProblem(nelx=30, nely=15, volfrac=0.4, case="cantilever", rmin=1.3)
    r = optimize(p, iters=25)
    assert r.compliance[-1] < r.compliance[0] and abs(r.volume - 0.4) < 2e-3


def test_problem_validation_and_bcs():
    for bad in (
        dict(nelx=2),
        dict(volfrac=0.99),
        dict(penal=0.5),
        dict(case="x"),
        dict(nu=0.6),
        dict(load=0.0),
        dict(nelx=100, nely=4),
        dict(E0=-1),
    ):
        with pytest.raises(ContractError):
            TopologyProblem(**bad)
    p = TopologyProblem(nelx=60, nely=20, case="mbb")
    fixed, f = p.boundary_conditions()
    assert len(fixed) == 21 + 1  # symmetry edge (all ux) + roller
    assert f[2 * p.node(0, 20) + 1] == -1.0 and f.sum() == -1.0
    fixed_c, fc = TopologyProblem(nelx=20, nely=10, case="cantilever").boundary_conditions()
    assert len(fixed_c) == 2 * 11 and fc.sum() == -1.0


def test_load_material(tmp_path):
    df = pd.DataFrame(
        {
            "material_id": ["a", "b", "c", "d"],
            "formula_pretty": ["Al", "Fe", "Bad", "SiC"],
            "E": [70.0, 200.0, -5.0, 410.0],
            "nu": [0.33, 0.29, 0.3, 0.17],
            "density_g_cm3": [2.7, 7.8, 1.0, 3.2],
        }
    )
    assert load_material(df)["formula"] == "SiC"  # stiffest valid
    assert load_material(df, target_E_gpa=190)["formula"] == "Fe"
    assert load_material(df, formula="Al")["E_gpa"] == 70.0
    df.to_parquet(tmp_path / "m.parquet")
    df.to_csv(tmp_path / "m.csv", index=False)
    assert load_material(tmp_path / "m.parquet")["formula"] == load_material(tmp_path / "m.csv")["formula"]
    with pytest.raises(ConditioningError):
        load_material(df, formula="nope")
    with pytest.raises(ConditioningError):
        load_material(df.drop(columns=["E"]))
    with pytest.raises(ConditioningError):
        load_material(df.assign(E=-1.0))
    with pytest.raises(FileNotFoundError):
        load_material(tmp_path / "missing.parquet")
