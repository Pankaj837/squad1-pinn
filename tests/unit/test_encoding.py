import pytest
import torch

from squad1.contracts import CondSpec
from squad1.encoding import (
    ELEMENTS,
    ConditioningTokenizer,
    ConditioningToTokens,
    FormulaTokenizer,
    SmilesTokenizer,
    token_accuracy,
    token_loss,
    train_tokens,
)
from squad1.errors import ConditioningError, ContractError, NonFiniteError

G = {"composition": 30, "inferred": 8, "target": 4}


# --------------------------------------------------------------------- tokenizers
@pytest.fixture(scope="module")
def smiles():
    return SmilesTokenizer().build_vocab(["CCO", "C1=CC=CC=C1", "CC(=O)O", "C(=O)Cl", "[Hf]", "[Si]", "c1ccccc1Br"])


@pytest.mark.parametrize("s", ["CCO", "C1=CC=CC=C1", "CC(=O)O", "C(=O)Cl", "[Hf]", "[Si]", "c1ccccc1Br"])
def test_smiles_roundtrip(smiles, s):
    assert smiles.decode(smiles.encode(s)) == s


@pytest.mark.parametrize("bad", ["Hf", "Ti", "Zr", "Mo", "HfC", "Ta", "W", "CCX", "C C"])
def test_smiles_raises_instead_of_silently_dropping(bad):
    with pytest.raises(ConditioningError):
        SmilesTokenizer().tokenize(bad)


def test_smiles_bracket_atoms_and_two_letter_organics():
    t = SmilesTokenizer()
    assert t.tokenize("[Hf]C(Cl)Br") == ["[Hf]", "C", "(", "Cl", ")", "Br"]


@pytest.fixture(scope="module")
def formula():
    return FormulaTokenizer().build_vocab(["HfC-SiC", "Al2Cu0.5Mg", "ZrB2/SiC", "Ti(C:N)"])


@pytest.mark.parametrize("s", ["HfC-SiC", "Al2Cu0.5Mg", "ZrB2/SiC", "Ti(C:N)"])
def test_formula_roundtrip(formula, s):
    assert formula.decode(formula.encode(s)) == s


def test_formula_element_matching_is_longest_first_and_validated():
    t = FormulaTokenizer()
    assert t.tokenize("CoNbTa") == ["Co", "Nb", "Ta"] and t.tokenize("HfC") == ["Hf", "C"]
    assert t.tokenize("Al2O3") == ["Al", "2", "O", "3"] and t.tokenize("Cu0.5") == ["Cu", "0.5"]
    for bad in ("Xx", "hfc", "HfC_SiC", "Hf C"):
        with pytest.raises(ConditioningError):
            t.tokenize(bad)
    assert len(ELEMENTS) == 103 and len(set(ELEMENTS)) == 103


def test_encode_details_and_errors(smiles):
    ids = smiles.encode("CCO", max_length=8)
    assert len(ids) == 8 and ids[0] == smiles.token_to_id["<SOS>"] and ids[-1] == smiles.pad_id
    ids = smiles.encode("C" * 300, max_length=16)
    assert len(ids) == 16 and ids[-1] == smiles.token_to_id["<EOS>"]  # <EOS> survives truncation
    ids = smiles.encode("CCO")
    junk = [*ids[:5], smiles.token_to_id["C"], smiles.token_to_id["C"], *ids[5:]]
    assert smiles.decode(junk) == "CCO"  # decode stops at <EOS>
    assert smiles.decode([9999, smiles.token_to_id["C"]]) == "C"  # unknown ids ignored
    with pytest.raises(ContractError):
        smiles.encode("CCO", max_length=2)
    with pytest.raises(ConditioningError):
        SmilesTokenizer().encode("CCO")
    with pytest.raises(ConditioningError):
        SmilesTokenizer().build_vocab([])
    with pytest.raises(ConditioningError):
        smiles.tokenize("")
    unk = smiles.encode("CCS")  # S known? not in vocab -> <UNK>
    assert smiles.token_to_id["<UNK>"] in unk


def test_vocab_persistence(smiles, tmp_path):
    p = tmp_path / "v.json"
    smiles.save(p)
    back = SmilesTokenizer.load(p)
    assert back.token_to_id == smiles.token_to_id and back.encode("CCO") == smiles.encode("CCO")
    with pytest.raises(ConditioningError):
        FormulaTokenizer.load(p)
    with pytest.raises(ConditioningError):
        SmilesTokenizer.from_dict({"kind": "smiles", "tokens": ["A", "B"]})


# ------------------------------------------------------------------------ encoder
@pytest.fixture(scope="module")
def model():
    torch.manual_seed(0)
    return ConditioningToTokens(
        G, vocab_size=20, embed_dim=64, max_seq_len=24, num_heads=4, num_layers=2, ff_dim=128
    ).eval()


def test_shapes_and_seq_len(model):
    e, i, lg = model(torch.randn(3, 42))
    assert e.shape == (3, 24, 64) and i.shape == (3, 24) and lg.shape == (3, 24, 20)
    assert model(torch.randn(2, 42), seq_len=10)[0].shape == (2, 10, 64)
    with pytest.raises(ContractError):
        model(torch.randn(2, 42), seq_len=25)
    with pytest.raises(ContractError):
        model(torch.randn(2, 42), seq_len=0)


def test_attention_is_not_degenerate(model):
    *_, w = model(torch.randn(2, 42), return_attn=True)
    assert w.shape == (2, 24, 42)
    assert torch.allclose(w.sum(-1), torch.ones(2, 24), atol=1e-4)
    assert w.max() < 0.99 and w.std(dim=-1).mean() > 0


def test_guards_nan_wrong_dv_and_bad_config(model):
    x = torch.randn(2, 42)
    x[0, 3] = float("nan")
    with pytest.raises(NonFiniteError):
        model(x)
    with pytest.raises(ContractError):
        model(torch.randn(2, 41))
    with pytest.raises(ContractError):
        model(torch.randn(42))
    with pytest.raises(ContractError):
        ConditioningToTokens(G, 10, embed_dim=63, num_heads=4)
    with pytest.raises(ContractError):
        ConditioningTokenizer({}, 8)


def test_batch_independence_and_conditioning_sensitivity(model):
    x = torch.randn(4, 42)
    assert torch.allclose(model(x)[0][1], model(x[1:2])[0][0], atol=1e-4)
    assert (model(torch.zeros(1, 42))[0] - model(torch.ones(1, 42))[0]).abs().mean() > 1e-3


def test_from_spec_and_backward():
    spec = CondSpec(tuple(f"E{i}" for i in range(6)), ("a", "b"), ("K", "m"), ("t",), ("K",))
    m = ConditioningToTokens.from_spec(spec, 12, embed_dim=32, max_seq_len=8, num_heads=4, num_layers=1, ff_dim=64)
    _, _, lg = m(torch.randn(2, spec.dv))
    lg.sum().backward()
    assert all(p.grad is not None for p in m.parameters())


def test_loss_ignores_pad_and_shape_check():
    lg = torch.zeros(1, 4, 5)
    tgt = torch.tensor([[1, 2, 0, 0]])
    full = token_loss(lg, tgt, pad_id=0)
    assert full.item() == pytest.approx(torch.log(torch.tensor(5.0)).item(), rel=1e-5)
    assert token_accuracy(lg, tgt, 0) in (0.0, 0.5, 1.0)
    with pytest.raises(ContractError):
        token_loss(lg, tgt[:, :3], 0)


def test_model_can_learn_to_map_conditions_to_strings():
    """Trainability check: memorise 6 (cond_vec -> formula) pairs."""
    strings = ["HfC-SiC", "ZrB2-SiC", "Al2O3", "TiC-TiN", "SiC", "HfB2"]
    tok = FormulaTokenizer().build_vocab(strings)
    ids = torch.tensor([tok.encode(s, max_length=16) for s in strings])
    torch.manual_seed(1)
    cond = torch.randn(len(strings), 42)
    m = ConditioningToTokens(
        G, tok.vocab_size, embed_dim=64, max_seq_len=16, num_heads=4, num_layers=2, ff_dim=128, dropout=0.0
    )
    hist = train_tokens(m, cond, ids, tok.pad_id, steps=250, lr=3e-3, seed=1)
    assert hist["loss"][-1] < 0.25 * hist["loss"][0] and hist["acc"][-1] > 0.95
    with torch.no_grad():
        pred = m(cond, seq_len=16)[1]
    assert [tok.decode(p) for p in pred] == strings
    with pytest.raises(ContractError):
        train_tokens(m, cond[:3], ids, tok.pad_id, steps=1)
