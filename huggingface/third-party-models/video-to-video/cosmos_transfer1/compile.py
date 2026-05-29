import argparse
import importlib.util
import os
import sys
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType
from typing import Any, Sequence

import yaml

PRESET_CONFIG_REL_PATH = "configs/preset.yaml"
IMPL_PREFIX = "compile_"
IMPL_SUFFIX = "_impl.py"
CKPT_DIR_ENV = "RBLN_CKPT_DIR"

BASE_DIR = Path(__file__).resolve().parent


def _discover_impl_map(base_dir: Path) -> dict[str, Path]:
    mapping: dict[str, Path] = {}
    for path in sorted(base_dir.glob(f"{IMPL_PREFIX}*{IMPL_SUFFIX}")):
        name = path.name
        mode = name[len(IMPL_PREFIX) : -len(IMPL_SUFFIX)]
        if mode:
            mapping[mode] = path
    return mapping


SCRIPT_MAP: dict[str, Path] = _discover_impl_map(BASE_DIR)


def _load_preset_config(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    with path.open("r", encoding="utf-8") as f:
        cfg: dict[str, Any] = dict(yaml.safe_load(f) or {})
    if ckpt := os.environ.get(CKPT_DIR_ENV):
        cfg.setdefault("defaults", {})["checkpoint_dir"] = ckpt
    return cfg


PRESET_CFG: dict[str, Any] = _load_preset_config(BASE_DIR / PRESET_CONFIG_REL_PATH)


def _preset_for(model_name: str) -> tuple[str, list[str]]:
    group: dict[str, Any] = PRESET_CFG["compile"]
    entry: dict[str, Any] = group[model_name]

    mode: str = entry["mode"]

    merged: dict[str, Any] = {}
    merged.update(PRESET_CFG.get("defaults", {}))
    merged.update(entry)

    merged.pop("mode", None)
    flags = merged.pop("flags", ())

    args: list[str] = []
    for flag in flags:
        args.append(f"--{flag}")

    for key, value in merged.items():
        if value is None:
            continue
        if isinstance(value, bool):
            if value:
                args.append(f"--{key}")
        else:
            args.extend([f"--{key}", str(value)])

    return mode, args


def _available_presets() -> list[str]:
    compile_group = PRESET_CFG.get("compile", {})
    return sorted(compile_group)


def parsing_argument(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        nargs="?",
        help=f"preset name from {PRESET_CONFIG_REL_PATH}",
    )
    parsed_argv = list(argv) if argv is not None else None
    return parser.parse_args(parsed_argv)


def import_module_from_path(path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(path.stem, str(path))
    if not spec or not spec.loader:
        raise ImportError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@contextmanager
def _override_argv(new_argv: list[str]):
    original = list(sys.argv)
    try:
        sys.argv = new_argv
        yield
    finally:
        sys.argv = original


def _invoke_child_main(
    script_path: Path,
    forwarded_args: list[str],
    invoked_token: str,
) -> int:
    module = import_module_from_path(script_path)
    main_fn = getattr(module, "main", None)
    if not callable(main_fn):
        raise AttributeError(f"{script_path.name} has no callable main()")

    cli_name = Path(__file__).name
    prog = f"python {cli_name} {invoked_token}"

    with _override_argv([prog, *forwarded_args]):
        try:
            result = main_fn()
        except SystemExit as e:
            code = e.code
            return code if isinstance(code, int) else 1
        return result if isinstance(result, int) else 0


def _print_presets_usage(header: str) -> None:
    print(header, file=sys.stderr)
    presets = _available_presets()
    if presets:
        print(f"available presets in {PRESET_CONFIG_REL_PATH}:", file=sys.stderr)
        for name in presets:
            print(f"  - {name}", file=sys.stderr)


def main(argv: Sequence[str] | None = None) -> int:
    ns = parsing_argument(argv)
    model_name: str | None = ns.model_name

    if model_name is None:
        cli_name = Path(__file__).name
        _print_presets_usage(f"usage: python {cli_name} --model_name <name>")
        return 2

    try:
        mode, preset_args = _preset_for(model_name)
    except KeyError:
        _print_presets_usage(f"unknown preset '{model_name}'")
        return 2

    script_path = SCRIPT_MAP.get(mode)
    if script_path is None:
        raise SystemExit(
            f"no script for mode '{mode}' (expected {IMPL_PREFIX}{mode}{IMPL_SUFFIX})"
        )

    return _invoke_child_main(script_path, preset_args, model_name)


if __name__ == "__main__":
    raise SystemExit(main())
