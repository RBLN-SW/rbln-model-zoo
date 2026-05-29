"""Model registry: discovery, filtering, and card-type resolution."""

from __future__ import annotations

import json
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml

MODEL_ROOTS = (
    "huggingface",
    "pytorch",
    "pytorch_dynamo",
    "tensorflow",
    "cpp",
    "serving",
    "vllm",
)
CLI_SEARCH_ROOTS = (
    "huggingface",
    "pytorch",
    "pytorch_dynamo",
    "tensorflow",
)
COMPILE_ONLY_ROOTS = ("cpp", "serving", "vllm")

GIT_DIRNAME = ".git"
SKIP_DIRS = frozenset({".venv", "__pycache__", "node_modules", GIT_DIRNAME})

# Script stems / filenames (single source for discovery + CLI)
STEM_COMPILE = "compile"
STEM_INFERENCE = "inference"
STEM_MAIN = "main"
COMPILE_PY = f"{STEM_COMPILE}.py"
INFERENCE_PY = f"{STEM_INFERENCE}.py"
MAIN_PY = f"{STEM_MAIN}.py"
ENTRY_SCRIPTS = (COMPILE_PY, MAIN_PY)

GIT_REV_TIMEOUT_SEC = 10
FIND_TIMEOUT_SEC = 30

# model_registry.yaml top-level keys
RK_CARDS = "cards"
RK_DEFAULT_CARDS = "default_cards"
RK_OVERRIDES = "overrides"

# On-disk cache payload keys
CK_FINGERPRINT = "fingerprint"
CK_MODELS = "models"


REPO_ROOT: Path = Path(__file__).resolve().parent.parent
REGISTRY_PATH: Path = REPO_ROOT / "model_registry.yaml"
DISCOVER_CACHE_DIR: Path = REPO_ROOT / ".rbln_zoo_cache"
DISCOVER_CACHE_FILE: Path = DISCOVER_CACHE_DIR / "discovered_models.json"

REQ_FILE = "requirements.txt"

_DEFAULT_REGISTRY: dict[str, Any] = {
    RK_CARDS: {},
    RK_DEFAULT_CARDS: ["CA"],
    RK_OVERRIDES: {},
}


@dataclass(frozen=True, slots=True)
class ModelEntry:
    path: str
    framework: str
    task: str
    name: str
    cards: tuple[str, ...] = ()
    has_compile: bool = False
    has_inference: bool = False
    has_main: bool = False


def load_registry(path: Path | None = None) -> dict[str, Any]:
    if path is None:
        path = REGISTRY_PATH
    if not path.exists():
        return {**_DEFAULT_REGISTRY}
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    for key, default in _DEFAULT_REGISTRY.items():
        data.setdefault(key, default)
    if data[RK_OVERRIDES] is None:
        data[RK_OVERRIDES] = {}
    return data


def _parse_model_path(rel: str) -> tuple[str, str, str]:
    hf = MODEL_ROOTS[0]
    parts = rel.split("/")
    framework = parts[0] if parts else ""
    if framework == hf and len(parts) > 1:
        framework = f"{hf}/{parts[1]}"

    if framework.startswith(f"{hf}/"):
        task = parts[2] if len(parts) > 2 else ""
        name = "/".join(parts[3:]) if len(parts) > 3 else ""
    else:
        task = parts[1] if len(parts) > 1 else ""
        name = "/".join(parts[2:]) if len(parts) > 2 else ""

    return framework, task, name or parts[-1]


def _cache_fingerprint() -> str:
    parts: list[str] = []
    try:
        git_hash = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            text=True,
            timeout=GIT_REV_TIMEOUT_SEC,
            stderr=subprocess.DEVNULL,
        ).strip()
        parts.append(f"git:{git_hash}")
    except (
        subprocess.CalledProcessError,
        FileNotFoundError,
        subprocess.TimeoutExpired,
    ):
        # Not a git checkout (e.g. COPYed into a Docker image) — fall back to
        # the registry stat below as the sole fingerprint input.
        parts.append("git:none")
    if REGISTRY_PATH.exists():
        st = REGISTRY_PATH.stat()
        parts.append(f"reg:{st.st_mtime_ns}:{st.st_size}")
    return "|".join(parts)


def _is_nested_in_submodule(model_dir: Path) -> bool:
    """True when *model_dir* sits inside a git-submodule root (but is not the root itself)."""
    md = model_dir.resolve()
    root = REPO_ROOT.resolve()
    if md == root:
        return False
    for cur in md.parents:
        if cur == root:
            break
        if (cur / GIT_DIRNAME).is_file():
            return md != cur
    return False


def _model_entry_from_dict(d: dict[str, Any]) -> ModelEntry:
    return ModelEntry(**{**d, "cards": tuple(d["cards"])})


def _or_join(flag: str, values: list[str]) -> list[str]:
    """Build ``[flag, v1, -o, flag, v2, ...]`` for ``find``."""
    parts: list[str] = []
    for v in values:
        if parts:
            parts.append("-o")
        parts.extend([flag, v])
    return parts


def _find_scripts() -> list[Path]:
    roots = [REPO_ROOT / n for n in MODEL_ROOTS if (REPO_ROOT / n).exists()]
    if not roots:
        return []

    prune = _or_join("-path", [f"*/{d}/*" for d in sorted(SKIP_DIRS)])
    names = _or_join("-name", list(ENTRY_SCRIPTS))

    out = subprocess.check_output(
        [
            "find",
            *map(str, roots),
            "(",
            *prune,
            ")",
            "-prune",
            "-o",
            "(",
            *names,
            ")",
            "-print",
        ],
        text=True,
        timeout=FIND_TIMEOUT_SEC,
    )
    return [Path(line) for raw in out.splitlines() if (line := raw.strip())]


def _paths_to_model_entries(
    registry: dict[str, Any], paths: list[Path]
) -> list[ModelEntry]:
    default_cards = registry[RK_DEFAULT_CARDS]
    overrides = registry[RK_OVERRIDES]
    seen: set[str] = set()
    models: list[ModelEntry] = []

    for p in sorted(paths, key=lambda x: x.as_posix()):
        model_dir = p.parent
        if _is_nested_in_submodule(model_dir):
            continue
        rel = model_dir.relative_to(REPO_ROOT).as_posix()
        if rel in seen:
            continue
        seen.add(rel)

        framework, task, name = _parse_model_path(rel)
        override = overrides.get(rel, {})
        cards = tuple(override.get("cards", default_cards))

        models.append(
            ModelEntry(
                path=rel,
                framework=framework,
                task=task,
                name=name,
                cards=cards,
                has_compile=(model_dir / COMPILE_PY).exists(),
                has_inference=(model_dir / INFERENCE_PY).exists(),
                has_main=(model_dir / MAIN_PY).exists(),
            )
        )

    return sorted(models, key=lambda m: m.path)


def discover_models(registry: dict[str, Any]) -> list[ModelEntry]:
    return _paths_to_model_entries(registry, _find_scripts())


def discover_models_cached(
    registry: dict[str, Any],
    *,
    refresh: bool = False,
) -> list[ModelEntry]:
    fp = _cache_fingerprint()
    if not refresh and DISCOVER_CACHE_FILE.exists():
        data = json.loads(DISCOVER_CACHE_FILE.read_text(encoding="utf-8"))
        if data[CK_FINGERPRINT] == fp:
            return [_model_entry_from_dict(m) for m in data[CK_MODELS]]

    models = discover_models(registry)
    DISCOVER_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    DISCOVER_CACHE_FILE.write_text(
        json.dumps({CK_FINGERPRINT: fp, CK_MODELS: [asdict(m) for m in models]}),
        encoding="utf-8",
    )
    return models


def resolve_card(registry: dict[str, Any], name: str) -> str | None:
    """Resolve a card name or alias to its primary card name.

    Each entry in ``registry['cards']`` may declare an ``aliases``
    list (e.g. ``CA`` accepts ``RBLN-CA02``/``CA12``/``CA22``/``CA25``).
    Matching is case-insensitive on both the primary name and aliases.

    Returns the primary card name on hit, ``None`` if the input is
    not a known card or alias.
    """
    if not name:
        return None
    norm = name.strip().upper()
    for primary, info in registry.get(RK_CARDS, {}).items():
        if primary.upper() == norm:
            return primary
        aliases = info.get("aliases", []) if isinstance(info, dict) else []
        if any(str(a).strip().upper() == norm for a in aliases):
            return primary
    return None


def filter_models(
    models: list[ModelEntry],
    *,
    card: str | None = None,
    framework: str | None = None,
    task: str | None = None,
    search: str | None = None,
) -> list[ModelEntry]:
    result = models
    if card:
        target = card.lower()
        result = [m for m in result if any(c.lower() == target for c in m.cards)]
    if framework:
        target = framework.lower().rstrip("/")
        if not target:
            return []
        result = [
            m
            for m in result
            if m.framework.lower() == target
            or m.framework.lower().startswith(f"{target}/")
        ]
    if task:
        target = task.lower()
        result = [m for m in result if m.task.lower() == target]
    if search is not None:
        q = search.lower()
        if not q:
            return []
        result = [m for m in result if q in m.path.lower()]
    return result
