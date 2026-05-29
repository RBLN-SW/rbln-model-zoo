from __future__ import annotations

import os
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib  # type: ignore[no-redef]


ALLOWED_KEYS: frozenset[str] = frozenset({"rbln_home"})


def config_path() -> Path:
    base = os.environ.get("XDG_CONFIG_HOME") or str(Path.home() / ".config")
    return Path(base) / "rbln-zoo" / "config.toml"


def load() -> dict[str, str]:
    p = config_path()
    if not p.is_file():
        return {}
    try:
        data = tomllib.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return {k: str(v) for k, v in data.items() if k in ALLOWED_KEYS}


def _dump(data: dict[str, str]) -> str:
    return "".join(f'{k} = "{v}"\n' for k, v in sorted(data.items()))


def set_(key: str, value: str) -> None:
    if key not in ALLOWED_KEYS:
        raise ValueError(f"unknown key {key!r}; allowed: {sorted(ALLOWED_KEYS)}")
    data = load()
    data[key] = value
    p = config_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(_dump(data), encoding="utf-8")


def unset(key: str) -> bool:
    if key not in ALLOWED_KEYS:
        raise ValueError(f"unknown key {key!r}; allowed: {sorted(ALLOWED_KEYS)}")
    data = load()
    if key not in data:
        return False
    del data[key]
    p = config_path()
    if data:
        p.write_text(_dump(data), encoding="utf-8")
    elif p.exists():
        p.unlink()
    return True


def get(key: str) -> str | None:
    if key not in ALLOWED_KEYS:
        raise ValueError(f"unknown key {key!r}; allowed: {sorted(ALLOWED_KEYS)}")
    return load().get(key)
