"""Strict tokenizers: SMILES (molecules) and chemical formulas (ceramics / alloys / composites).

Both are *strict*: any character that cannot be tokenised raises (the earlier ``re.findall`` implementation
silently dropped e.g. ``Hf``/``Ti``/``Zr``). Both keep ``<EOS>`` on truncation, stop decoding at ``<EOS>`` and
can be saved/loaded (vocabulary persistence is part of reproducibility).
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

from squad1.errors import ConditioningError, ContractError

SPECIAL_TOKENS = ("<PAD>", "<UNK>", "<SOS>", "<EOS>")

# fmt: off
ELEMENTS = (
    "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne", "Na", "Mg",
    "Al", "Si", "P", "S", "Cl", "Ar", "K", "Ca", "Sc", "Ti", "V", "Cr",
    "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Ge", "As", "Se", "Br", "Kr",
    "Rb", "Sr", "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd",
    "In", "Sn", "Sb", "Te", "I", "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd",
    "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu", "Hf",
    "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg", "Tl", "Pb", "Bi", "Po",
    "At", "Rn", "Fr", "Ra", "Ac", "Th", "Pa", "U", "Np", "Pu", "Am", "Cm",
    "Bk", "Cf", "Es", "Fm", "Md", "No", "Lr",
)
# fmt: on


class _Tokenizer:
    kind = "base"
    _regex: re.Pattern[str]

    def __init__(self) -> None:
        self.token_to_id: dict[str, int] = {}
        self.id_to_token: dict[int, str] = {}

    # ---------------------------------------------------------------- tokenise
    def tokenize(self, text: str) -> list[str]:
        if not isinstance(text, str) or not text:
            raise ConditioningError("text must be a non-empty string")
        toks: list[str] = []
        pos = 0
        for m in self._regex.finditer(text):
            if m.start() != pos:
                raise ConditioningError(f"untokenisable text {text[pos : m.start()]!r} at position {pos} in {text!r}")
            toks.append(m.group(0))
            pos = m.end()
        if pos != len(text):
            raise ConditioningError(f"untokenisable text {text[pos:]!r} at position {pos} in {text!r}")
        return toks

    # -------------------------------------------------------------- vocabulary
    def build_vocab(self, texts: Iterable[str]) -> _Tokenizer:
        vocab = sorted({t for s in texts for t in self.tokenize(s)})
        if not vocab:
            raise ConditioningError("cannot build a vocabulary from an empty corpus")
        all_tokens = [*SPECIAL_TOKENS, *vocab]
        self.token_to_id = {t: i for i, t in enumerate(all_tokens)}
        self.id_to_token = {i: t for t, i in self.token_to_id.items()}
        return self

    @property
    def vocab_size(self) -> int:
        return len(self.token_to_id)

    @property
    def pad_id(self) -> int:
        return self._id("<PAD>")

    def _id(self, tok: str) -> int:
        if not self.token_to_id:
            raise ConditioningError("vocabulary is empty: call build_vocab(...) first")
        return self.token_to_id[tok]

    # ------------------------------------------------------------ encode/decode
    def encode(self, text: str, max_length: int = 128) -> list[int]:
        if max_length < 3:
            raise ContractError("max_length must be >= 3 (<SOS>, at least one token, <EOS>)")
        unk = self._id("<UNK>")
        ids = [self._id("<SOS>"), *(self.token_to_id.get(t, unk) for t in self.tokenize(text))]
        ids = [*ids[: max_length - 1], self._id("<EOS>")]  # <EOS> always survives truncation
        return ids + [self.pad_id] * (max_length - len(ids))

    def decode(self, ids: Sequence[int]) -> str:
        eos = self._id("<EOS>")
        out: list[str] = []
        for i in ids:
            i = int(i)
            if i == eos:
                break
            tok = self.id_to_token.get(i)
            if tok is None or tok in SPECIAL_TOKENS:
                continue
            out.append(tok)
        return "".join(out)

    # ------------------------------------------------------------- persistence
    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "tokens": [self.id_to_token[i] for i in range(self.vocab_size)]}

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=1), encoding="utf-8")

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> _Tokenizer:
        if d.get("kind") != cls.kind:
            raise ConditioningError(f"vocabulary kind {d.get('kind')!r} does not match {cls.kind!r}")
        tokens = list(d["tokens"])
        if tuple(tokens[: len(SPECIAL_TOKENS)]) != SPECIAL_TOKENS:
            raise ConditioningError("vocabulary must start with the special tokens")
        tok = cls()
        tok.token_to_id = {t: i for i, t in enumerate(tokens)}
        tok.id_to_token = dict(enumerate(tokens))
        return tok

    @classmethod
    def load(cls, path: str | Path) -> _Tokenizer:
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


class SmilesTokenizer(_Tokenizer):
    """SMILES: bracket atoms ``[Hf]``, organic subset (``Br``/``Cl`` before single letters), bonds, rings, branches."""

    kind = "smiles"
    _regex = re.compile(r"(\[[^\]]+\]|Br|Cl|[BCNOPSFIbcnosp]|\(|\)|=|#|-|\+|\\|/|:|\.|%[0-9]{2}|[0-9]|@@?|~|\*|\$)")


class FormulaTokenizer(_Tokenizer):
    """Chemical formulas such as ``HfC-SiC`` or ``Al2Cu0.5Mg``: element symbols, numbers, ``()`` and ``- / : .``.

    Element symbols are matched longest-first against the periodic table, so ``Co`` is cobalt (not C + o) and a
    made-up symbol such as ``Xx`` raises.
    """

    kind = "formula"
    _els = sorted(ELEMENTS, key=lambda e: (-len(e), e))
    _regex = re.compile("(" + "|".join(_els) + r"|\d+(?:\.\d+)?|\(|\)|-|/|:|\.)")
