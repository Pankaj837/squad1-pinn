"""Injection-safe compiler for loss/rule strings (replaces ``sympify`` + ``exec`` of untrusted text).

Only numbers, allow-listed variable names, ``+ - * / **``, unary minus/plus and allow-listed functions are accepted.
Attribute access, subscripts, comprehensions, lambdas, calls to anything else, keyword arguments and strings are all
rejected *before* anything is evaluated. The accepted AST is compiled and evaluated with empty builtins.

Use this for rule strings that may originate from an LLM or a user, e.g.
``"max(0, cost - budget) + P_escape * impact"``.
"""

from __future__ import annotations

import ast
from collections.abc import Callable, Collection

import torch

from squad1.errors import UnsafeExpressionError

MAX_LENGTH = 500

_FUNCS: dict[str, Callable[..., torch.Tensor]] = {
    "max": torch.maximum,
    "min": torch.minimum,
    "exp": torch.exp,
    "log": torch.log,
    "sqrt": torch.sqrt,
    "abs": torch.abs,
    "relu": torch.relu,
    "tanh": torch.tanh,
    "sin": torch.sin,
    "cos": torch.cos,
}
_ARITY = {"max": 2, "min": 2}
_BIN = (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow)


def _check(node: ast.AST, allowed: Collection[str], used: set[str]) -> None:
    if isinstance(node, ast.Expression):
        return _check(node.body, allowed, used)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            raise UnsafeExpressionError(f"only numeric constants are allowed, got {node.value!r}")
        return None
    if isinstance(node, ast.Name):
        if node.id not in allowed:
            raise UnsafeExpressionError(f"unknown variable {node.id!r}")
        used.add(node.id)
        return None
    if isinstance(node, ast.BinOp) and isinstance(node.op, _BIN):
        _check(node.left, allowed, used)
        return _check(node.right, allowed, used)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        return _check(node.operand, allowed, used)
    if isinstance(node, ast.Call):
        if not (isinstance(node.func, ast.Name) and node.func.id in _FUNCS) or node.keywords:
            raise UnsafeExpressionError("only calls to the allow-listed functions (positional args) are allowed")
        if len(node.args) != _ARITY.get(node.func.id, 1):
            raise UnsafeExpressionError(f"{node.func.id}() takes {_ARITY.get(node.func.id, 1)} argument(s)")
        for a in node.args:
            _check(a, allowed, used)
        return None
    raise UnsafeExpressionError(f"disallowed syntax: {type(node).__name__}")


def compile_rule(expression: str, allowed_variables: Collection[str]) -> Callable[..., torch.Tensor]:
    """Return ``f(**tensors) -> scalar tensor`` (mean of the expression)."""
    if not isinstance(expression, str) or not expression.strip():
        raise UnsafeExpressionError("expression must be a non-empty string")
    if len(expression) > MAX_LENGTH:
        raise UnsafeExpressionError(f"expression longer than {MAX_LENGTH} characters")
    try:
        tree = ast.parse(expression.strip(), mode="eval")
    except SyntaxError as exc:
        raise UnsafeExpressionError(f"invalid expression: {exc.msg}") from exc
    used: set[str] = set()
    _check(tree, allowed_variables, used)
    code = compile(tree, "<rule>", "eval")

    def fn(**kw: torch.Tensor | float) -> torch.Tensor:
        missing = used - set(kw)
        if missing:
            raise TypeError(f"missing variables {sorted(missing)}")
        env: dict[str, object] = {k: torch.as_tensor(v, dtype=torch.float32) for k, v in kw.items() if k in used}
        for name, f in _FUNCS.items():
            env[name] = lambda *a, _f=f: _f(*[torch.as_tensor(x, dtype=torch.float32) for x in a])
        out = eval(code, {"__builtins__": {}}, env)
        return torch.as_tensor(out, dtype=torch.float32).mean()

    return fn
