"""Documentation must not rot: README code runs; every doc link/path that names a module exists."""

import importlib
import pathlib
import re

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]


def _python_blocks(path: pathlib.Path) -> list[str]:
    return re.findall(r"```python\n(.*?)```", path.read_text(encoding="utf-8"), flags=re.S)


def test_readme_quick_example_runs():
    blocks = _python_blocks(ROOT / "README.md")
    assert blocks, "README has no python example"
    ns: dict = {}
    exec(compile(blocks[0], "README.md", "exec"), ns)
    result = ns["result"]
    assert result.converged.all()


def test_documented_modules_exist():
    text = (ROOT / "README.md").read_text(encoding="utf-8") + (ROOT / "docs" / "INTERFACES.md").read_text(
        encoding="utf-8"
    )
    names = set(re.findall(r"`(squad1\.[a-z_]+)`", text))
    assert {"squad1.contracts", "squad1.physics", "squad1.projection", "squad1.generation"} <= names
    for n in names:
        importlib.import_module(n)


@pytest.mark.parametrize("doc", ["INTERFACES.md", "INTEGRATION.md", "DECISIONS.md", "LITERATURE.md"])
def test_docs_present_and_nonempty(doc):
    assert len((ROOT / "docs" / doc).read_text(encoding="utf-8")) > 500


def test_documented_api_names_are_importable():
    text = (ROOT / "docs" / "INTEGRATION.md").read_text(encoding="utf-8")
    for mod, name in re.findall(r"from (squad1[\w.]*) import ([\w, ]+)", text):
        m = importlib.import_module(mod)
        for n in [x.strip() for x in name.split(",")]:
            assert hasattr(m, n), f"{mod}.{n} documented but missing"
