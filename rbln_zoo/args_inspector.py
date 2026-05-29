"""Static extraction of argparse flags from a script without importing it."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ArgSpec:
    flags: tuple[str, ...]
    type: str | None = None
    default: Any = None
    choices: tuple[Any, ...] | None = None
    help: str | None = None
    required: bool = False
    action: str | None = None
    nargs: str | None = None


def extract_arg_specs(script_path: Path) -> list[ArgSpec]:
    try:
        source = script_path.read_text()
    except OSError:
        return []
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []

    specs: list[ArgSpec] = []
    for node in ast.walk(tree):
        spec = _spec_from_call(node)
        if spec is not None:
            specs.append(spec)
    return specs


def _spec_from_call(node: ast.AST) -> ArgSpec | None:
    if not (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "add_argument"
    ):
        return None

    flags = tuple(
        a.value
        for a in node.args
        if isinstance(a, ast.Constant) and isinstance(a.value, str)
    )
    if not flags:
        return None

    kw = {k.arg: _literal(k.value) for k in node.keywords if k.arg}
    choices = kw.get("choices")
    return ArgSpec(
        flags=flags,
        type=_as_str(kw.get("type")),
        default=kw.get("default"),
        choices=tuple(choices) if isinstance(choices, (list, tuple)) else None,
        help=kw.get("help"),
        required=bool(kw.get("required", False)),
        action=_as_str(kw.get("action")),
        nargs=_as_str(kw.get("nargs")),
    )


def _literal(node: ast.AST) -> Any:
    try:
        return ast.literal_eval(node)
    except (ValueError, SyntaxError, TypeError):
        pass
    if isinstance(node, ast.Name):
        return node.id
    try:
        return ast.unparse(node)
    except Exception:
        return None


def _as_str(value: Any) -> str | None:
    return None if value is None else str(value)
